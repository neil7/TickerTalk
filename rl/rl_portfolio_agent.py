"""
RLPortfolioDecider — drop-in inference wrapper; no-op until best.pt exists.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ACTIONS = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
CHECKPOINT = Path("rl/checkpoints/best.pt")


class RLPortfolioDecider:
    def __init__(self, checkpoint_path: Optional[Path] = None, n_features: int = 44):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT
        self.n_features = n_features
        self._net: Optional[Any] = None
        self.is_trained = False
        self._try_load()

    def _try_load(self) -> None:
        try:
            from rl.dqn_policy import QNetwork
            net = QNetwork(self.n_features)
            net.load_state_dict(
                torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            )
            net.eval()
            self._net = net
            self.is_trained = True
        except Exception:
            self._net = None
            self.is_trained = False

    def reload(self) -> None:
        """Hot-reload the checkpoint after a nightly training run."""
        self._try_load()

    def decide(
        self,
        feature_vector: Any,
        fallback_scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Returns a recommendation dict.  Falls back to rule-based logic if
        no trained checkpoint is present.
        """
        if self.is_trained and self._net is not None:
            vec = np.array(feature_vector, dtype=np.float32)
            tensor = torch.tensor(vec).unsqueeze(0)
            with torch.no_grad():
                q_vals = self._net(tensor).squeeze(0)
            action_idx = int(q_vals.argmax().item())
            action = ACTIONS[action_idx]
            q_list = q_vals.tolist()
            # Simple conviction: spread between best and second-best Q
            sorted_q = sorted(q_list, reverse=True)
            spread = sorted_q[0] - sorted_q[1] if len(sorted_q) > 1 else 0.0
            conviction = min(10.0, max(1.0, 5.0 + spread * 5.0))
            return {
                "recommendation": action,
                "conviction": round(conviction, 1),
                "source": "rl_policy",
                "q_values": {ACTIONS[i]: round(q_list[i], 4) for i in range(len(ACTIONS))},
            }

        # Fallback to composite score logic
        if fallback_scores:
            composite = fallback_scores.get("composite_score", 0)
            conviction = fallback_scores.get("conviction", 5)
            if composite >= 1:
                action = "STRONG_BUY"
            elif composite >= 0.3:
                action = "BUY"
            elif composite <= -1:
                action = "STRONG_SELL"
            elif composite <= -0.3:
                action = "SELL"
            else:
                action = "HOLD"
            return {
                "recommendation": action,
                "conviction": round(conviction, 1),
                "source": "rule_based_fallback",
            }

        return {"recommendation": "HOLD", "conviction": 5.0, "source": "default"}
