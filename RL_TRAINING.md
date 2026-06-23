# RL Training Guide — TickTalker

Offline reinforcement learning layer that learns from its own recommendations.
Policy replaces the rule-based `_determine_action()` once trained.

---

## How it works

```
python main.py AAPL          # 1. run analysis → logs record to JSONL
                              # (repeat daily, diverse tickers)

fetch_and_label_outcomes()   # 2. after 5+ days: fetch actual price outcomes
                              # fills reward_5d_sharpe, reward_calmar, etc.

python -m rl.train           # 3. after 50+ labelled records: train QNetwork
                              # saves rl/checkpoints/best.pt

wire RLPortfolioDecider      # 4. activate live policy (see CLAUDE.md step 3)
```

**Algorithm:** Double DQN + Conservative Q-Learning (CQL).  
Off-policy → learns from logs, no live market access needed.  
Discrete 5-class output: `STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL`

**Network:** `Input(44) → Linear(256) → LayerNorm → ReLU → Dropout(0.2) → Linear(128) → LayerNorm → ReLU → Dropout(0.2) → Linear(64) → ReLU → Linear(5)`

---

## Step 1 — Collect training data

Every `python main.py <TICKER>` call appends one record to `logs/recommendations.jsonl`.

**Tips for better data:**
- Run different tickers daily — tech, energy, finance, small-cap
- Cover different market conditions (bull / bear / sideways / high-vol)
- Each run = one training sample — more is always better

**Data thresholds:**

| Records | Status |
|---------|--------|
| < 10 | Training blocked |
| 10–50 | Trains but overfits — wiring test only |
| 50–200 | Practical minimum for real signal |
| 200–2000 | Sweet spot |
| 2000+ | Consider graduating to IQL |

---

## Step 2 — Label outcomes

Run after at least 5 trading days have passed since your earliest unlabelled record.

```bash
python -c "
from rl.recommendation_logger import RecommendationLogger
logger = RecommendationLogger()
n = logger.fetch_and_label_outcomes()
print(f'Labelled {n} new records')
print(logger.stats())
"
```

**What it does:** downloads 15 days of price history via yfinance for each unlabelled record, computes:

| Reward field | Formula | Use when |
|---|---|---|
| `reward_5d_sharpe` | `direction × return_5d` | default |
| `reward_calmar` | `direction × return_5d / max_drawdown_5d` | risk-adjusted |
| `reward_10d_sharpe` | `direction × return_10d` | less noise, medium-term |
| `reward_rr_weighted` | `reward_5d_sharpe × min(R/R / 2, 1.5)` | weights by entry quality |
| `reward_regime_adj` | `reward_5d_sharpe × 0.5` if regime mismatch | regime-aware (future) |

`direction = +1` for BUY actions, `-1` for SELL, `0` for HOLD.

**Check reward variance before training:**
```python
from rl.recommendation_logger import RecommendationLogger
stats = RecommendationLogger().stats()
print(stats["reward_stats"])
# Need std > 0.005 — if near zero, data is too flat
```

**Automate nightly (cron):**
```bash
# crontab -e
0 18 * * 1-5 cd /path/to/tick_talker && python -c "from rl.recommendation_logger import RecommendationLogger; RecommendationLogger().fetch_and_label_outcomes()"
```

---

## Step 3 — Train the policy

```bash
# Sensible defaults (start here)
python -m rl.train

# Full control
python -m rl.train \
  --log      logs/recommendations.jsonl \
  --epochs   500 \
  --alpha    1.0 \
  --lr       3e-4 \
  --batch    64 \
  --patience 40

# Evaluate saved checkpoint without retraining
python -m rl.train --eval-only
```

**Reward function options** — pass via `--reward`:

```bash
python -m rl.train --reward reward_5d_sharpe    # default
python -m rl.train --reward reward_calmar       # penalises drawdown
python -m rl.train --reward reward_10d_sharpe   # less noise
python -m rl.train --reward reward_rr_weighted  # rewards good R/R entries (recommended)
```

Checkpoint saved to `rl/checkpoints/best.pt`.

---

## Hyperparameter reference

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `--alpha` | `1.0` | CQL regularisation. Increase to 2–5 if policy over-buys/sells. |
| `--lr` | `3e-4` | Reduce to `1e-4` if < 100 records. |
| `--batch` | `64` | Reduce to `32` if < 100 records. |
| `--patience` | `40` | Epochs without val improvement before early stop. |
| `--epochs` | `300` | Max training epochs (early stop usually triggers first). |

---

## Diagnosing a bad policy

**Policy always outputs STRONG_BUY:**
```bash
python -m rl.train --alpha 3.0   # CQL alpha too low
```

**Policy always outputs HOLD:**
```bash
# Check reward variance — must be > 0.005
python -c "from rl.recommendation_logger import RecommendationLogger; print(RecommendationLogger().stats()['reward_stats'])"
# If std near zero: not enough market diversity. Run more tickers.
# If std OK: alpha too high — try --alpha 0.5
```

**Evaluation shows low accuracy:**  
Normal for < 100 records. Policy is still better than random if it directionally correlates with reward. Use `--eval-only` to inspect per-action Q-values:
```bash
python -m rl.train --eval-only
# mean_q_per_action: check that BUY/STRONG_BUY Q-values are higher
# on days when market actually went up
```

---

## Step 4 — Activate the live policy

Only do this after `rl/checkpoints/best.pt` exists.

In `workflow.py` → `StockAnalysisWorkflow.__init__`, add:
```python
from rl.rl_portfolio_agent import RLPortfolioDecider
self.rl_decider = RLPortfolioDecider()  # no-op until checkpoint exists
```

In `_portfolio_manager_node`, after `state = self.portfolio_agent.analyze(state)`:
```python
if self.rl_decider.is_trained:
    sv = state.get("signal_vector", {})
    vec = sv.get("vector")
    if vec is not None:
        rl_action = self.rl_decider.decide(
            vec,
            fallback_scores=state.get("final_recommendation", {}).get("scores"),
        )
        state["final_recommendation"]["action"] = rl_action
```

**Hot-reload after nightly training** (no restart needed):
```python
self.rl_decider.reload()
```

---

## Feature vector quick reference

The 44-dim state input the Q-network sees:

| Group | Indices | What |
|-------|---------|------|
| Technical indicators | 0–9 | RSI, Stoch-K, BB position/width, MACD hist, SMA%, momentum, volume ratio, ATR |
| Technical signals | 10–14 | Trend, RSI signal, MACD signal, OBV, S/R position |
| Risk metrics | 15–22 | Ann. vol, max drawdown, VaR 95, Sharpe, win rate, G/L ratio, risk level, vol regime |
| Sentiment | 23–27 | News sentiment, count, political impact, LLM bias, positive ratio |
| Reddit | 28–31 | Available flag, retail sentiment, score, WSB trending |
| Fundamental | 32–37 | P/E, PEG, 52-week position, valuation, beta, P/E assessment |
| Portfolio scores | 38–40 | Technical score, sentiment score, composite score |
| Market context | 41–43 | Log price, market cap tier, momentum divergence |

**Important:** Never reorder `FEATURE_NAMES` in `agents/signal_extractor.py` without bumping `FEATURE_VERSION` and deleting `rl/checkpoints/best.pt`. The Q-network is a function of feature *positions*, not names.

---

## Graduation path

When you have 2000+ labelled records across diverse regimes:

1. **Implicit Q-Learning (IQL)** — more principled offline RL, drop-in for `OfflineDQNTrainer`
2. **PPO + backtesting simulator** — build `gym.Env` wrapper around yfinance historical data, use DQN checkpoint as initialisation
3. **Continuous position sizing** — action space `[0,1]` portfolio fraction, switch to Discrete SAC or TD3+BC

---

## File locations

| File | Purpose |
|------|---------|
| `rl/recommendation_logger.py` | Writes JSONL, labels outcomes via yfinance |
| `rl/dqn_policy.py` | QNetwork, ReplayBuffer, OfflineDQNTrainer |
| `rl/rl_portfolio_agent.py` | Live inference wrapper |
| `rl/train.py` | CLI training script |
| `agents/signal_extractor.py` | AgentState → 44-dim vector |
| `logs/recommendations.jsonl` | Training data (auto-created, gitignored) |
| `rl/checkpoints/best.pt` | Trained policy (gitignored) |
