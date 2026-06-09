"""
QNetwork, ReplayBuffer, OfflineDQNTrainer — Double DQN with CQL regularisation.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
N_ACTIONS = len(ACTIONS)
CHECKPOINT_DIR = Path("rl/checkpoints")


# ─────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, n_features: int = 44, n_actions: int = N_ACTIONS,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    @classmethod
    def from_log_file(cls, path: str | Path,
                      reward_key: str = "reward_5d_sharpe") -> "ReplayBuffer":
        buf = cls()
        p = Path(path)
        if not p.exists():
            return buf
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reward = rec.get(reward_key)
                if reward is None:
                    continue
                vec = rec.get("feature_vector")
                if not vec or len(vec) == 0:
                    continue
                action_str = rec.get("recommendation", "HOLD")
                # Map to action index
                if action_str in ACTIONS:
                    action_idx = ACTIONS.index(action_str)
                elif "STRONG_BUY" in action_str:
                    action_idx = 0
                elif "STRONG_SELL" in action_str:
                    action_idx = 4
                elif "BUY" in action_str:
                    action_idx = 1
                elif "SELL" in action_str:
                    action_idx = 3
                else:
                    action_idx = 2
                buf._records.append({
                    "state": np.array(vec, dtype=np.float32),
                    "action": action_idx,
                    "reward": float(reward),
                })
        return buf

    def __len__(self) -> int:
        return len(self._records)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self._records, min(batch_size, len(self._records)))
        states = torch.tensor(np.stack([b["state"] for b in batch]), dtype=torch.float32)
        actions = torch.tensor([b["action"] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)
        return states, actions, rewards

    def temporal_split(self, val_frac: float = 0.2) -> Tuple["ReplayBuffer", "ReplayBuffer"]:
        """Chronological split — never shuffle (prevents future leakage)."""
        n = len(self._records)
        split = max(1, int(n * (1 - val_frac)))
        train_buf = ReplayBuffer()
        val_buf = ReplayBuffer()
        train_buf._records = self._records[:split]
        val_buf._records = self._records[split:]
        return train_buf, val_buf


# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
class OfflineDQNTrainer:
    def __init__(
        self,
        n_features: int = 44,
        lr: float = 3e-4,
        cql_alpha: float = 1.0,
        batch_size: int = 64,
        patience: int = 40,
        target_update_tau: float = 0.005,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cql_alpha = cql_alpha
        self.batch_size = batch_size
        self.patience = patience
        self.tau = target_update_tau
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.online_net = QNetwork(n_features).to(self.device)
        self.target_net = QNetwork(n_features).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.online_net.parameters(), lr=lr)

    def train(self, buffer: ReplayBuffer, n_epochs: int = 300,
              val_frac: float = 0.2) -> Dict[str, Any]:
        if len(buffer) < 10:
            raise ValueError(f"Too few labelled records ({len(buffer)}). Need ≥10.")

        train_buf, val_buf = buffer.temporal_split(val_frac)
        best_val_loss = float("inf")
        patience_counter = 0
        history: List[Dict] = []

        for epoch in range(1, n_epochs + 1):
            self.online_net.train()
            train_loss = self._train_epoch(train_buf)
            val_loss = self._val_epoch(val_buf) if len(val_buf) > 0 else train_loss
            self._soft_update_target()

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stop at epoch {epoch} (best val_loss={best_val_loss:.6f})")
                    break

            if epoch % 50 == 0:
                print(f"Epoch {epoch:4d} | train={train_loss:.6f} | val={val_loss:.6f}")

        return {"best_val_loss": best_val_loss, "epochs_run": len(history), "history": history}

    def _train_epoch(self, buf: ReplayBuffer) -> float:
        if len(buf) == 0:
            return 0.0
        states, actions, rewards = buf.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)

        q_vals = self.online_net(states)
        q_taken = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

        td_loss = nn.functional.mse_loss(q_taken, rewards)

        # CQL: push down Q for actions not in training distribution
        logsumexp = torch.logsumexp(q_vals, dim=1)
        cql_loss = (logsumexp - q_taken).mean()

        loss = td_loss + self.cql_alpha * cql_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    def _val_epoch(self, buf: ReplayBuffer) -> float:
        if len(buf) == 0:
            return 0.0
        self.online_net.eval()
        with torch.no_grad():
            states, actions, rewards = buf.sample(len(buf))
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)

            q_vals = self.online_net(states)
            q_taken = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)
            td_loss = nn.functional.mse_loss(q_taken, rewards)

            logsumexp = torch.logsumexp(q_vals, dim=1)
            cql_loss = (logsumexp - q_taken).mean()
            loss = td_loss + self.cql_alpha * cql_loss
        return float(loss.item())

    def _soft_update_target(self) -> None:
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1 - self.tau) * tp.data)

    def _save_checkpoint(self) -> None:
        path = self.checkpoint_dir / "best.pt"
        torch.save(self.online_net.state_dict(), path)

    def evaluate(self, buffer: ReplayBuffer) -> Dict[str, Any]:
        """Report per-action accuracy and mean Q-values from the saved checkpoint."""
        checkpoint = self.checkpoint_dir / "best.pt"
        if checkpoint.exists():
            self.online_net.load_state_dict(
                torch.load(checkpoint, map_location=self.device)
            )
        self.online_net.eval()
        if len(buffer) == 0:
            return {"error": "empty buffer"}

        states, actions, rewards = buffer.sample(len(buffer))
        states = states.to(self.device)
        with torch.no_grad():
            q_vals = self.online_net(states)
            predicted = q_vals.argmax(dim=1).cpu()

        correct = int((predicted == actions).sum().item())
        accuracy = correct / len(actions)

        mean_q = q_vals.mean(dim=0).cpu().tolist()
        return {
            "accuracy": round(accuracy, 4),
            "n_samples": len(actions),
            "mean_q_per_action": {ACTIONS[i]: round(mean_q[i], 4) for i in range(N_ACTIONS)},
        }
