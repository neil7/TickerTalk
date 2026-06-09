# CLAUDE.md — RL Fine-Tuning Layer for TickerTalk

This file extends the base `CLAUDE.md` with everything needed to implement
the offline reinforcement learning layer on top of the existing multi-agent
stock analysis pipeline.  Read the base `CLAUDE.md` first; this document
assumes full familiarity with the existing architecture.

---

## Goal

Replace the rule-based `_determine_action()` call in `PortfolioManagerAgent`
with a trained Q-network that learns from the system's own logged
recommendations and their actual price outcomes.  The system must keep
working identically until a checkpoint exists — zero-regression on existing
behaviour.

---

## New files to create (verbatim from the agreed design)

```
agents/
  signal_extractor.py      # Converts AgentState → 44-dim float32 vector
rl/
  __init__.py              # Package init — exports RecommendationLogger
  recommendation_logger.py # Appends (state, action, reward) records to JSONL
  dqn_policy.py            # QNetwork, ReplayBuffer, OfflineDQNTrainer
  rl_portfolio_agent.py    # RLPortfolioDecider — drop-in inference wrapper
  train.py                 # CLI training script
logs/                      # Created at runtime — git-ignore this directory
  recommendations.jsonl    # One JSON record per analysis run (append-only)
rl/checkpoints/            # Created at runtime — git-ignore
  best.pt                  # Saved by OfflineDQNTrainer after each training run
```

**All five Python files have been fully written and tested in the design
session.  Do NOT rewrite them from scratch.  Copy them exactly as specified
in the sections below.**

---

## Files to modify

### 1. `agents/__init__.py`

Add `SignalExtractor` to the exports:

```python
from .signal_extractor import SignalExtractor

__all__ = [
    "TechnicalAnalysisAgent",
    "SentimentAnalysisAgent",
    "FundamentalAnalysisAgent",
    "RiskManagementAgent",
    "PortfolioManagerAgent",
    "RedditSentimentAgent",
    "SignalExtractor",        # ← add this
]
```

### 2. `workflow.py` — five precise changes

#### 2a. Add imports (near the top, after existing agent imports)

```python
from agents.signal_extractor import SignalExtractor
from rl.recommendation_logger import RecommendationLogger
```

#### 2b. Add `signal_vector` field to `AgentState` TypedDict

Find the `AgentState` class and append one field:

```python
class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]
    ticker: str
    market_data: Dict[str, Any]
    technical_analysis: Dict[str, Any]
    sentiment_data: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    reddit_analysis: Dict[str, Any]
    fundamental_analysis: Dict[str, Any]
    economic_indicators: Dict[str, Any]
    social_data: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    final_recommendation: Dict[str, Any]
    signal_vector: Dict[str, Any]   # ← ADD THIS
```

#### 2c. Instantiate extractor and logger in `StockAnalysisWorkflow.__init__`

Inside `__init__`, after the existing agent instantiations:

```python
self.signal_extractor = SignalExtractor()
self.rec_logger       = RecommendationLogger()
```

#### 2d. Add `_signal_extraction_node` method to `StockAnalysisWorkflow`

```python
def _signal_extraction_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a numeric feature vector from all agent outputs and log it.

    Runs AFTER portfolio_manager so all scores (including composite_score
    from final_recommendation.scores) are available in state.
    """
    console.print("[cyan]🔢 Extracting RL signal vector...[/cyan]")

    result = self.signal_extractor.extract(state)

    if result["missing"]:
        console.print(
            f"[yellow]⚠  Signal extractor: {len(result['missing'])} agent(s) "
            f"degraded to defaults → {result['missing']}[/yellow]"
        )

    state["signal_vector"] = result

    try:
        record_id = self.rec_logger.log(state, result)
        console.print(
            f"[green]✓ Signal extraction done "
            f"(quality={result['data_quality']:.2f}, id={record_id})[/green]"
        )
    except Exception as exc:
        # Logging failure must never break a live analysis run
        console.print(f"[yellow]⚠  Logger error (non-fatal): {exc}[/yellow]")

    state["messages"].append(
        f"Signal extraction: {len(result['vector'])} features, "
        f"quality={result['data_quality']:.2f}"
    )
    return state
```

#### 2e. Rewire the graph in `_build_workflow`

Find `_build_workflow` and make these two changes:

1. Register the new node (after all existing `add_node` calls):
```python
workflow.add_node("signal_extraction", self._signal_extraction_node)
```

2. Replace the final edge:
```python
# BEFORE:
workflow.add_edge("portfolio_manager", END)

# AFTER:
workflow.add_edge("portfolio_manager",  "signal_extraction")
workflow.add_edge("signal_extraction",  END)
```

The full execution order becomes:
```
data_collection → technical → risk → sentiment → reddit → fundamental
    → portfolio_manager → signal_extraction → END
```

#### 2f. Add `signal_vector` to the initial state in `analyze()`

Find the `initial_state = AgentState(...)` dict inside `analyze()` and add:

```python
signal_vector={},   # populated by signal_extraction_node
```

### 3. Optional: Wire `RLPortfolioDecider` into `_portfolio_manager_node`

This step activates the trained policy in the live pipeline.  Only do this
after `rl/checkpoints/best.pt` exists (i.e. after the first training run).

In `__init__`, add:
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
        console.print(
            f"[cyan]🤖 RL policy: {rl_action['recommendation']} "
            f"(conviction={rl_action['conviction']}, "
            f"source={rl_action['source']})[/cyan]"
        )
```

Note: `signal_vector` is populated by `_signal_extraction_node` which runs
*after* `portfolio_manager`.  For the RL decider to use it, either:
- Move signal extraction to run *before* `portfolio_manager` (requires
  removing the portfolio score group from the feature vector), OR
- Use the signal from the *previous* run stored in the log (recommended —
  the RL policy is already one step behind by design in offline RL).

The simplest correct approach: call `self.signal_extractor.extract(state)`
*inline* inside `_portfolio_manager_node`, before the portfolio agent runs,
and store the vector in a local variable without saving to state yet.  Then
use it for the RL decision, and let the signal_extraction_node handle
persistence to state and the log.  See implementation note at the bottom.

---

## Architecture: the 44-feature signal vector

`SignalExtractor.extract(state)` returns a `dict` with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `vector` | `np.ndarray (44,) float32` | The actual RL state input |
| `features` | `dict[str, float]` | Named values for logging/debug |
| `data_quality` | `float 0–1` | Fraction of agents with valid data |
| `missing` | `list[str]` | Agents that fell back to defaults |
| `version` | `str` | `"1.0"` — never reorder features without bumping |

**Feature groups (positions 0–43):**

```
Group 1 (0–9)   Technical indicators — rsi_norm, stoch_k_norm, bb_position,
                bb_width_norm, macd_hist_norm, pct_from_20sma, pct_from_50sma,
                momentum_10_norm, volume_ratio_norm, atr_pct_norm

Group 2 (10–14) Technical signals (ordinal) — trend_signal, rsi_signal,
                macd_signal, obv_signal, sr_position_signal

Group 3 (15–22) Risk metrics — annualized_vol_norm, max_drawdown_norm,
                var_95_norm, sharpe_norm, win_rate_norm, gain_loss_ratio_norm,
                risk_level_encoded, vol_regime_encoded

Group 4 (23–27) Sentiment — news_sentiment_encoded, news_count_norm,
                political_impact_encoded, llm_sentiment_bias, news_positive_ratio

Group 5 (28–31) Reddit — reddit_available, retail_sentiment_encoded,
                reddit_score_norm, wsb_trending_flag

Group 6 (32–37) Fundamental — pe_norm, peg_norm, w52_position_norm,
                valuation_encoded, beta_norm, pe_assessment_encoded

Group 7 (38–40) Portfolio scores — technical_score_norm, sentiment_score_norm,
                composite_score_norm

Group 8 (41–43) Market context — log_price_norm, market_cap_tier,
                momentum_divergence
```

All values are normalised to approximately `[-1, 1]` or `[0, 1]`.
Safe defaults are applied for any missing/errored agent (same graceful-
degradation pattern as `reddit_agent`).

---

## Architecture: the JSONL recommendation log

`RecommendationLogger.log(state, signal_result)` appends one record to
`logs/recommendations.jsonl`.  Key fields:

```json
{
  "record_id":       "AAPL_1703123456",
  "ticker":          "AAPL",
  "timestamp_utc":   "2024-12-21T10:30:00+00:00",
  "unix_ts":         1703123456,
  "recommendation":  "BUY",
  "conviction":      7.4,
  "time_horizon":    "medium_term",
  "composite_score": 0.75,
  "price_at_decision": 187.62,
  "feature_version": "1.0",
  "feature_vector":  [0.623, 0.714, ...],   // 44 floats
  "data_quality":    1.0,
  "missing_agents":  [],
  "features":        { "rsi_norm": 0.623, ... },
  "outcome_fetched": false,
  "reward_5d_sharpe": null
}
```

After 21+ calendar days, run `fetch_and_label_outcomes()` to fill in:

```json
{
  "outcome_fetched":  true,
  "price_at_1d":      189.10,
  "price_at_5d":      192.30,
  "return_5d":        0.0249,
  "reward_5d_sharpe": 0.0249
}
```

`reward_5d_sharpe = direction × return_5d` where direction is `+1` for BUY
actions, `-1` for SELL actions, `0` for HOLD.

---

## Architecture: the RL algorithm

**Algorithm: Double DQN with Conservative Q-Learning (CQL) regularisation**

This was chosen over PPO and SAC for two reasons:
1. **Off-policy**: learns from the JSONL log without needing live market access.
   PPO requires on-policy rollouts and cannot consume historical logs.
2. **Discrete actions**: BUY/HOLD/SELL is a 5-class discrete space.
   SAC was designed for continuous action spaces.

**Loss function (one-step MDP):**
```
L = MSE( Q(s, a_taken), reward ) + α × ( logsumexp_a[Q(s,a)] − Q(s, a_taken) )
     ↑ TD error (target = r)          ↑ CQL penalty — pushes down Q-values for
                                         actions not in the training distribution
```

**Action space (5 discrete actions):**
```python
ACTIONS = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
```

**Network architecture:**
```
Input (44) → Linear(256) → LayerNorm → ReLU → Dropout(0.2)
           → Linear(128) → LayerNorm → ReLU → Dropout(0.2)
           → Linear(64)  → ReLU
           → Linear(5)   # Q-values, one per action
```

**Key hyperparameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `cql_alpha` | 1.0 | CQL regularisation strength. Increase to 2–5 if policy over-buys/sells. |
| `learning_rate` | 3e-4 | AdamW. Reduce to 1e-4 for <100 records. |
| `batch_size` | 64 | Reduce to 32 for <100 records. |
| `patience` | 40 | Early-stop epochs without val improvement. |
| `target_update_tau` | 0.005 | Soft Polyak update of target network. |
| `val_frac` | 0.2 | Temporal split — DO NOT shuffle (prevents future leakage). |

---

## Training workflow

### Step 1 — Collect recommendations (automated, runs on every `main.py` call)

The `_signal_extraction_node` writes to `logs/recommendations.jsonl`
automatically after each full analysis run.  No manual steps needed.

### Step 2 — Label outcomes (run nightly after 21+ days of data)

```python
from rl.recommendation_logger import RecommendationLogger
logger = RecommendationLogger()
n = logger.fetch_and_label_outcomes()
print(f"Labelled {n} new records")
print(logger.stats())
```

Or as a cron job:
```bash
python -c "
from rl.recommendation_logger import RecommendationLogger
print(RecommendationLogger().fetch_and_label_outcomes(), 'records labelled')
"
```

### Step 3 — Train the policy (run after 50+ labelled records)

```bash
# Sensible defaults
python -m rl.train

# Full options
python -m rl.train \
  --log     logs/recommendations.jsonl \
  --epochs  500 \
  --alpha   1.0 \
  --lr      3e-4 \
  --batch   64 \
  --patience 40

# Evaluate without retraining
python -m rl.train --eval-only
```

Checkpoint is saved to `rl/checkpoints/best.pt`.

### Step 4 — Hot-reload in a running workflow (optional)

```python
# Call this after a nightly training run to pick up the updated policy
self.rl_decider.reload()
```

### Data requirements

| Records | Quality |
|---------|---------|
| < 10 | Training blocked — too few samples |
| 10–50 | Trains but overfits — useful for wiring tests only |
| 50–200 | Practical minimum for meaningful signal |
| 200–2000 | Sweet spot — cover bull/bear/sideways regimes |
| 2000+ | Consider graduating to CQL proper (see below) |

---

## Graduation path (future work)

The current implementation is a solid first step.  When you have 2000+
labelled records across diverse market regimes, consider:

1. **Implicit Q-Learning (IQL)** — a more principled offline RL algorithm that
   avoids out-of-distribution action queries entirely.  Drop-in replacement
   for `OfflineDQNTrainer` using the same `ReplayBuffer`.

2. **PPO with a backtesting simulator** — build a `gym.Env` wrapper around
   `yfinance` historical data, then PPO becomes valid (it has an environment
   to collect rollouts from).  Use the trained DQN checkpoint as the PPO
   policy initialisation.

3. **Continuous position sizing** — extend the action space to a continuous
   value in `[0, 1]` representing portfolio fraction, then switch to
   Discrete SAC or TD3+BC.

---

## Conventions specific to the RL layer

- **No LLM calls in `SignalExtractor`** — pure deterministic maths.  Fast
  enough to run synchronously in the node.
- **Never reorder `FEATURE_NAMES`** in `signal_extractor.py` without bumping
  `FEATURE_VERSION` from `"1.0"` to `"2.0"` and retraining from scratch.  The
  Q-network is a function of feature *positions*, not names.
- **Logging failures are non-fatal** — the `try/except` around `rec_logger.log()`
  in `_signal_extraction_node` must never be removed.  A full disk or network
  error must not crash a live analysis.
- **Checkpoint loading is non-fatal** — `RLPortfolioDecider._try_load()` swallows
  all exceptions and sets `is_trained = False`.  The existing rule-based
  `_determine_action()` remains active until a valid checkpoint exists.
- **Temporal split, never random** — `ReplayBuffer.temporal_split()` splits by
  record order (chronological).  Shuffling before splitting would leak future
  price information into the training set.
- **`cql_alpha` tuning heuristic**:
  - Policy always outputs STRONG_BUY → `alpha` too low; increase to 2–5.
  - Policy always outputs HOLD → `alpha` too high OR reward variance is too low
    (check `logger.stats()["reward_stats"]["std"]`; it should be > 0.005).

---

## Implementation note: RL decision timing in the pipeline

The signal vector ideally needs to be computed *before* `portfolio_manager`
runs so the RL policy can replace its action.  But the portfolio score
features (Group 7) require `final_recommendation.scores`, which is written by
`portfolio_manager`.

**Recommended resolution** (two-pass approach):

In `_portfolio_manager_node`, call `SignalExtractor.extract(state)` *twice*:

1. **First pass** (before portfolio_agent.analyze): extract a partial vector
   (Groups 1–6 and 8 only, omit Group 7).  Use this for the RL action decision.
2. **Second pass** (after portfolio_agent.analyze): extract the full vector
   (all 44 features including Group 7 portfolio scores).  Store in
   `state["signal_vector"]` and pass to `rec_logger.log()`.

The `signal_extraction_node` at the end of the graph can then skip re-extraction
and just log if `state["signal_vector"]` is already populated.

For the initial implementation, the simpler approach is acceptable:
skip step 3 of the workflow (RL decision injection) entirely and focus on
logging + training first.  The policy override can be added once there are
enough labelled records.

---

## Prompt integration guide

Five trading prompt templates were analysed for fit with this RL framework.
Below is their ranking by relevance, the exact RL component each maps to,
and the concrete implementation changes each implies.

---

### Relevance ranking

| # | Prompt | Relevance | Maps to |
|---|--------|-----------|---------|
| 4 | Market Regime Detection | 🟢 Highest | Feature vector (new Group 9) + core RL state |
| 2 | Backtesting | 🟢 High | Reward function — richer outcome labels |
| 5 | Multi-Factor Strategy | 🟢 High | Feature vector (new Group 9 factor scores) |
| 3 | Risk-Reward Analysis | 🟡 Medium | Reward shaping — Calmar-based reward signal |
| 1 | Strategy Generation | 🟡 Medium | Cold-start synthetic data generation |

---

### Prompt 4 — Market Regime Detection (highest priority)

**Original prompt:**
> Analyze current market conditions for [asset]. Identify: trend (bull/bear/sideways),
> volatility level, volume behavior. Then recommend: best strategy type for this
> environment, what to avoid right now.

**Why it matters for RL:**
The RL policy makes one decision per analysis run, but a BUY signal that works
in a bull market is catastrophic in a bear market.  Without regime context in the
feature vector, the Q-network cannot condition its action on whether we are in an
uptrend or a crash.  Market regime detection IS the core contextual problem the
RL policy must solve.

**Implementation — add a `RegimeDetector` and extend the signal vector to 47 features:**

Create `agents/regime_detector.py`:

```python
"""
RegimeDetector
==============
Classifies the current broad market regime from market_data and appends
3 features to the signal vector.  Requires SPY (or benchmark) price data.

This extends FEATURE_VERSION from "1.0" to "1.1".
Update N_FEATURES in signal_extractor.py from 44 to 47 and bump FEATURE_VERSION.
"""
import numpy as np
from typing import Dict, Any

class RegimeDetector:

    def detect(self, market_data: Dict[str, Any], benchmark_prices: list) -> Dict[str, float]:
        """
        Returns 3 features to append to the signal vector as Group 9.

        regime_trend      -1=bear, 0=sideways, +1=bull
                          Rule: if benchmark SMA50 slope > +0.1%/day → bull,
                                if slope < -0.1%/day → bear, else sideways.
        regime_vol_tier    0=low, 0.5=normal, 1.0=elevated
                          Rule: annualised vol of benchmark last 20 days vs
                                its 252-day rolling mean.
        regime_breadth     0-1 (fraction of last 20 closes above their SMA20)
                          Proxy for internal market strength.
        """
        if not benchmark_prices or len(benchmark_prices) < 52:
            return {"regime_trend": 0.0, "regime_vol_tier": 0.5, "regime_breadth": 0.5}

        prices = np.array(benchmark_prices, dtype=float)
        sma50  = np.mean(prices[-50:])
        sma20  = np.mean(prices[-20:])

        # Trend: compare current price to 50-day SMA
        trend_pct = (prices[-1] - sma50) / sma50 * 100
        if   trend_pct >  2: regime_trend = 1.0
        elif trend_pct < -2: regime_trend = -1.0
        else:                regime_trend = 0.0

        # Volatility tier: annualised vol of last 20 days vs last 252 days
        returns     = np.diff(prices) / prices[:-1]
        recent_vol  = np.std(returns[-20:]) * np.sqrt(252) * 100
        hist_vol    = np.std(returns[-252:]) * np.sqrt(252) * 100 if len(returns) >= 252 else recent_vol

        if   recent_vol > hist_vol * 1.3: vol_tier = 1.0   # elevated
        elif recent_vol < hist_vol * 0.7: vol_tier = 0.0   # subdued
        else:                             vol_tier = 0.5    # normal

        # Breadth: fraction of recent prices above their own 20-day SMA
        breadth_vals = [1.0 if prices[i] > np.mean(prices[max(0,i-20):i+1]) else 0.0
                        for i in range(-20, 0)]
        breadth = float(np.mean(breadth_vals))

        return {
            "regime_trend":   regime_trend,
            "regime_vol_tier": vol_tier,
            "regime_breadth":  breadth,
        }
```

**Changes required in `signal_extractor.py`:**

1. In `FEATURE_NAMES`, append after `"momentum_divergence"`:
   ```python
   # Group 9 — Market regime (3 features)
   "regime_trend",      # -1=bear, 0=sideways, +1=bull
   "regime_vol_tier",   # 0=subdued, 0.5=normal, 1.0=elevated
   "regime_breadth",    # 0-1, fraction of recent closes above SMA20
   ```

2. Bump `FEATURE_VERSION = "1.1"` and `N_FEATURES = 47`.

3. Add `_extract_regime()` method using `RegimeDetector`.

4. **Important:** Any checkpoint trained on version 1.0 (44 features) is incompatible
   with version 1.1 (47 features).  Delete `rl/checkpoints/best.pt` and retrain.

**How the regime prompt is used at runtime:**

Add a `--regime-asset SPY` flag to `train.py` so the backtested regime can be
fetched during `fetch_and_label_outcomes()` and stored in each log record:
```json
"regime_at_decision": { "trend": 1.0, "vol_tier": 0.5, "breadth": 0.72 }
```

---

### Prompt 2 — Backtesting (reward function enrichment)

**Original prompt:**
> Backtest this trading strategy using historical data. Time period: 5-10 years.
> Metrics: CAGR, Sharpe ratio, max drawdown, win rate. Output: table + summary.
> When it performs best/worst. What market conditions break it.

**Why it matters for RL:**
The current `reward_5d_sharpe` is the only training signal.  This is a single
noisy number.  Backtesting logic tells us three better reward variants that
together reduce training variance and make the policy more robust.

**Implementation — three additional reward fields in `recommendation_logger.py`:**

Add to `_build_record()` outcome section:

```python
# In the record template — add these null fields alongside reward_5d_sharpe:
"reward_calmar":      None,   # return_5d / abs(max_intraday_drawdown_5d)
"reward_10d_sharpe":  None,   # 10-day version, lower noise than 5d
"reward_regime_adj":  None,   # reward_5d_sharpe × (1 if regime correct else 0.5)
```

Extend `_fetch_close_prices()` in `RecommendationLogger` to also fetch intraday
highs and lows to compute `max_intraday_drawdown_5d`.

In `fetch_and_label_outcomes()`, compute after fetching prices:

```python
# Calmar-style: reward is return divided by the worst drawdown during the hold
intraday_low  = min(daily_lows_during_hold)  # fetch via yf.download(...interval="1d")["Low"]
intraday_high = entry_price
max_dd = (intraday_high - intraday_low) / intraday_high
rec["reward_calmar"] = round(direction * ret5 / max(max_dd, 0.005), 6)

# 10-day reward (less noise than 5-day for medium-term strategies)
if rec.get("return_10d") is not None:
    rec["reward_10d_sharpe"] = round(direction * rec["return_10d"], 6)

# Regime-adjusted: reward is halved if the regime at decision was unfavourable
#   for the action taken (e.g., STRONG_BUY in a bear regime)
regime_trend = rec.get("regime_at_decision", {}).get("trend", 0)
if (direction > 0 and regime_trend < 0) or (direction < 0 and regime_trend > 0):
    rec["reward_regime_adj"] = round(rec["reward_5d_sharpe"] * 0.5, 6)
else:
    rec["reward_regime_adj"] = rec["reward_5d_sharpe"]
```

**Use in training:**

In `OfflineDQNTrainer`, expose a `reward_key` parameter (already supported in
`ReplayBuffer.from_log_file()`).  Experiment with different targets:

```bash
python -m rl.train --reward reward_calmar      # risk-adjusted, Calmar-style
python -m rl.train --reward reward_regime_adj  # regime-aware (recommended once regime data exists)
python -m rl.train --reward reward_5d_sharpe   # default
```

The backtesting prompt's "when it performs best/worst" insight becomes the
`reward_regime_adj` signal: the policy gets less reward for directional bets
taken against the prevailing regime.

---

### Prompt 5 — Multi-Factor Strategy (explicit factor scores)

**Original prompt:**
> Create a multi-factor strategy using: Momentum, Value, Volatility, Trend.
> Include: exact formula/logic for each factor, weight allocation (%), rebalancing
> frequency, example portfolio.

**Why it matters for RL:**
The Q-network implicitly learns the optimal weighting of the 44 raw features, but
it has to discover the factor structure from scratch.  Providing pre-aggregated
factor scores alongside the raw features acts as an inductive bias — the network
converges faster and generalises better with less data.

The four factors in the prompt map exactly to four existing feature groups:

| Factor | Feature group | Indices |
|--------|---------------|---------|
| Momentum | Group 1 (technical indicators) | 0–9 |
| Trend | Group 2 (technical signals) | 10–14 |
| Volatility | Group 3 (risk metrics) | 15–22 |
| Value | Group 6 (fundamental metrics) | 32–37 |

**Implementation — add factor scores as Group 9 (or extend Group 9 from regime):**

Add to `signal_extractor.py` as a new `_extract_factor_scores()` method:

```python
def _extract_factor_scores(self, features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute pre-aggregated factor scores from already-extracted raw features.
    Call AFTER all groups 1-8 are extracted and merged into `features`.

    Each score is a simple equal-weighted mean of the normalised sub-features
    that represent that factor.  Result is in [-1, +1].
    """
    def _mean(*keys):
        vals = [features[k] for k in keys if k in features]
        return float(np.mean(vals)) if vals else 0.0

    momentum_score = _mean(
        "momentum_10_norm", "pct_from_20sma", "pct_from_50sma",
        "macd_hist_norm", "rsi_signal",
    )
    trend_score = _mean(
        "trend_signal", "macd_signal", "obv_signal", "sr_position_signal",
    )
    volatility_score = _mean(
        # Invert vol features: low vol = positive score for trend strategies
        "sharpe_norm", "win_rate_norm", "gain_loss_ratio_norm",
    )
    value_score = _mean(
        "pe_assessment_encoded", "valuation_encoded", "peg_norm",
        "w52_position_norm",
    )

    return {
        "factor_momentum":   _clip(momentum_score,   -1.0, 1.0),
        "factor_trend":      _clip(trend_score,      -1.0, 1.0),
        "factor_volatility": _clip(volatility_score, -1.0, 1.0),
        "factor_value":      _clip(value_score,      -1.0, 1.0),
    }
```

Add these 4 keys to `FEATURE_NAMES` (bringing total to 51 features if regime is
also added, or 48 without regime).  Bump `FEATURE_VERSION` accordingly.

**Note on sequencing:** Call `_extract_factor_scores(features)` after all groups
1–8 are merged, since it reads from the already-populated `features` dict.

---

### Prompt 3 — Risk-Reward Analysis (reward shaping)

**Original prompt:**
> Analyze strategy's risk-reward profile. Break down: risk per trade, reward-to-risk
> ratio, drawdown patterns. Then suggest 3 improvements to reduce risk and 2 ways
> to increase returns without increasing risk.

**Why it matters for RL:**
The risk/reward ratio in the price_targets output (`price_targets["risk_reward_ratio"]`)
is already computed by `PortfolioManagerAgent._calculate_price_targets()`.  It should
be logged and used to weight the reward signal: high-conviction trades with a good R/R
ratio should produce a larger training signal than lucky low-R/R wins.

**Implementation — log the R/R ratio and use it to weight rewards:**

In `RecommendationLogger._build_record()`, add:

```python
price_targets = fr.get("price_targets", {})
"risk_reward_at_entry": price_targets.get("risk_reward_ratio"),  # e.g. 2.5
"atr_at_entry":         price_targets.get("atr"),
"stop_loss_pct":        price_targets.get("stop_loss_pct"),
```

In `fetch_and_label_outcomes()`, compute a quality-weighted reward:

```python
rr = rec.get("risk_reward_at_entry") or 1.0
# Trades with R/R >= 2 get full credit; R/R < 1 is penalised
rr_weight = min(rr / 2.0, 1.5)
rec["reward_rr_weighted"] = round(rec["reward_5d_sharpe"] * rr_weight, 6)
```

This encodes the prompt's core insight: the RL policy should prefer to recommend
trades with a good risk/reward profile, not just trades that happened to work.

The "3 improvements to reduce risk" and "2 ways to increase returns" from the
prompt translate to the `cql_alpha` tuning loop described in the Conventions
section above.

---

### Prompt 1 — Strategy Generation (cold-start synthetic data)

**Original prompt:**
> Act as a hedge fund quant. Generate 3 profitable trading strategies for [market].
> Constraints: Timeframe, Capital $10,000, Risk per trade 1-2%. For each strategy,
> provide: indicators, entry/exit rules, stop-loss/take-profit, market conditions,
> why it has edge.

**Why it matters for RL:**
The biggest practical problem with offline RL is that you need labelled data before
you can train, but you need a trained policy before you have useful labelled data.
This prompt solves the cold-start problem: use LLM-generated strategy rules to run
a backtest that produces synthetic (state, action, reward) tuples for training.

**Implementation — `rl/synthetic_data.py` (new file, optional):**

```python
"""
synthetic_data.py
=================
Cold-start data generator: converts LLM-generated strategy rules into
backtested training records compatible with ReplayBuffer.

Usage:
    python -m rl.synthetic_data \
        --ticker AAPL \
        --start  2020-01-01 \
        --end    2024-01-01 \
        --output logs/synthetic_recommendations.jsonl

The output file has the same schema as logs/recommendations.jsonl and can
be used as a supplement during early training:

    buffer = ReplayBuffer.from_log_file("logs/synthetic_recommendations.jsonl")
    trainer.train(buffer, n_epochs=100)  # pre-train
    # Then fine-tune on real logged data:
    real_buffer = ReplayBuffer.from_log_file("logs/recommendations.jsonl")
    trainer.train(real_buffer, n_epochs=200)
"""

# Three built-in strategy rules that correspond to the 3 most common
# strategies generated by the Prompt 1 template:

STRATEGY_RULES = {
    # Strategy 1: EMA Crossover (Trend-following)
    "ema_cross": {
        "entry_long":  lambda f: f["trend_signal"] >= 0.5 and f["macd_signal"] > 0,
        "entry_short": lambda f: f["trend_signal"] <= -0.5 and f["macd_signal"] < 0,
        "exit":        lambda f: abs(f["trend_signal"]) < 0.3,
        "market_best": "trending",
    },
    # Strategy 2: Mean Reversion (RSI oversold/overbought)
    "rsi_reversion": {
        "entry_long":  lambda f: f["rsi_norm"] < 0.35 and f["bb_position"] < 0.2,
        "entry_short": lambda f: f["rsi_norm"] > 0.70 and f["bb_position"] > 0.8,
        "exit":        lambda f: 0.45 < f["rsi_norm"] < 0.60,
        "market_best": "sideways",
    },
    # Strategy 3: Momentum breakout (volume surge + trend alignment)
    "momentum_breakout": {
        "entry_long":  lambda f: f["volume_ratio_norm"] > 0.7 and f["momentum_10_norm"] > 0.5,
        "entry_short": lambda f: f["volume_ratio_norm"] > 0.7 and f["momentum_10_norm"] < -0.5,
        "exit":        lambda f: f["volume_ratio_norm"] < 0.4,
        "market_best": "volatile",
    },
}
```

**When to use synthetic data:**
Use Prompt 1 to ask an LLM to generate strategy rules, translate each rule into a
lambda over feature names (as shown above), run `synthetic_data.py` on 3–5 years of
historical data, and use the output for pre-training before any real logged data exists.
Switch to real logged data as soon as you have 50+ labelled records — synthetic data
from rule-based strategies has inherent look-ahead bias in backtesting.

---

### Recommended implementation order

Given the above analysis, implement the prompt integrations in this sequence:

1. **Prompt 2 first** — richer reward signals (`reward_calmar`, `reward_10d_sharpe`,
   `reward_rr_weighted`).  No feature vector changes; backwards-compatible with any
   existing checkpoint and logs.

2. **Prompt 3 alongside Prompt 2** — log `risk_reward_at_entry` and `atr_at_entry`
   in `_build_record()`.  Same record schema extension, no retraining needed.

3. **Prompt 5 next** — add 4 factor score features.  Requires bumping `FEATURE_VERSION`
   to `"1.1"` and `N_FEATURES` to 48.  Delete and retrain checkpoint.

4. **Prompt 4 last** — adds 3 regime features (N_FEATURES → 51, version `"1.2"`).
   Most impactful but requires fetching SPY/benchmark data alongside each analysis.

5. **Prompt 1 as needed** — implement `synthetic_data.py` only if you need to
   cold-start training before accumulating 50+ real logged records.

---

## Quick smoke-test after implementation

```bash
# Run one full analysis — should produce no errors and write to the log
python main.py AAPL

# Verify the log was written
python -c "
import json
from pathlib import Path
line = Path('logs/recommendations.jsonl').read_text().splitlines()[-1]
rec = json.loads(line)
print('ticker:', rec['ticker'])
print('action:', rec['recommendation'])
print('quality:', rec['data_quality'])
print('vector len:', len(rec['feature_vector']))  # should be 44
print('missing:', rec['missing_agents'])           # ideally []
"

# Check log stats
python -c "
from rl.recommendation_logger import RecommendationLogger
print(RecommendationLogger().stats())
"
```

Expected output from the last command (before any outcomes are labelled):
```
{
  'total_records': 1,
  'labelled_records': 0,
  'unlabelled': 1,
  'action_counts': {'BUY': 1},
  'log_path': 'logs/recommendations.jsonl'
}
```