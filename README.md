# 🤖 TickTalker - Agentic AI Stock Analysis System

An advanced multi-agent AI system for stock market analysis, powered by **Claude/Gemini/Groq/Ollama** and **LangGraph** — with an offline **Reinforcement Learning fine-tuning layer** that learns from its own recommendations over time.

---

## 🧠 Architecture

### Core Multi-Agent Pipeline

```
data_collection → technical → risk → sentiment → reddit → fundamental
    → portfolio_manager → signal_extraction → END
```

| Agent | Role |
|-------|------|
| **Data Collection** | Yahoo Finance, Alpha Vantage, FRED, news, Reddit |
| **Technical Analysis** | RSI, MACD, Bollinger Bands, ATR, OBV, support/resistance |
| **Risk Management** | Volatility, VaR, Sharpe, drawdown, position sizing |
| **Sentiment Analysis** | News, White House briefings, LLM interpretation |
| **Reddit Sentiment** | WSB & retail sentiment via PRAW |
| **Fundamental Analysis** | P/E, PEG, 52-week position, valuation |
| **Portfolio Manager** | Synthesizes all → rule-based action (BUY/SELL/HOLD + price targets) |
| **Signal Extraction** | Converts state → 44-dim float32 vector, logs to JSONL |

### RL Fine-Tuning Layer (`rl/`)

Every analysis run automatically logs a record to `logs/recommendations.jsonl`. Once enough labelled records exist, an **offline Double DQN + CQL** policy replaces the rule-based `_determine_action()`.

```
logs/recommendations.jsonl   ← written after every main.py run
         ↓  (after 5+ days)
fetch_and_label_outcomes()   ← fetches actual price outcomes via yfinance
         ↓  (after 50+ labelled)
python -m rl.train           ← trains QNetwork, saves rl/checkpoints/best.pt
         ↓
RLPortfolioDecider           ← live policy replaces rule-based decisions
```

**Algorithm:** Double DQN with Conservative Q-Learning (CQL) regularisation — chosen because it learns off-policy from historical logs without needing live market access, and handles discrete 5-class action spaces (STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL).

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
git clone <repo>
cd tick_talker
pip install -r requirements.txt
```

### 2. Configure API keys

Create `.env` in project root:

```env
# LLM Provider (pick one)
LLM_PROVIDER=anthropic            # recommended
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6

# Or Gemini
LLM_PROVIDER=gemini
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash

# Or Groq (fast, free tier)
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile

# Or local Ollama
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:latest
OLLAMA_BASE_URL=http://localhost:11434

# Market data (free tiers)
ALPHA_VANTAGE_KEY=...
FRED_API_KEY=...
STOCKDATA_KEY=...

# Optional: Reddit sentiment
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=TickTalker/1.0
```

### 3. Run

```bash
# Full deep analysis — logs RL record automatically
python main.py AAPL

# Market scanner
python main.py --scan

# Day trading setups with buy/sell levels
python main.py --day-trade

# Penny stocks (>100% potential, under $5)
python main.py --penny

# Quick scan multiple tickers
python main.py --quick AAPL TSLA NVDA MSFT

# WallStreetBets trending stocks
python main.py --wsb

# All blue-chip stocks
python main.py --blue-chips

# Show LLM model config
python main.py --models
```

---

## 📈 RL Training Workflow

### Step 1 — Collect (automated)

Every `python main.py AAPL` run appends one record to `logs/recommendations.jsonl`. No manual steps.

### Step 2 — Label outcomes (after 5+ days)

```bash
python -c "
from rl.recommendation_logger import RecommendationLogger
logger = RecommendationLogger()
n = logger.fetch_and_label_outcomes()
print(f'Labelled {n} new records')
print(logger.stats())
"
```

### Step 3 — Train (after 50+ labelled records)

```bash
# Default
python -m rl.train

# With options
python -m rl.train --epochs 500 --alpha 1.0 --lr 3e-4 --batch 64

# Evaluate saved checkpoint without retraining
python -m rl.train --eval-only

# Experiment with reward functions
python -m rl.train --reward reward_calmar       # risk-adjusted
python -m rl.train --reward reward_rr_weighted  # R/R weighted
python -m rl.train --reward reward_5d_sharpe    # default
```

Checkpoint saved to `rl/checkpoints/best.pt`.

### Step 4 — Activate (after first successful training)

Follow instructions in `CLAUDE.md` → "Optional: Wire RLPortfolioDecider" to enable live policy.

### Data requirements

| Records | Status |
|---------|--------|
| < 10 | Training blocked |
| 10–50 | Trains but overfits (useful for wiring tests) |
| 50–200 | Practical minimum for signal |
| 200–2000 | Sweet spot |
| 2000+ | Consider graduating to IQL |

---

## 🔢 Signal Vector (44 features)

| Group | Indices | Features |
|-------|---------|----------|
| Technical indicators | 0–9 | RSI, Stoch-K, BB position/width, MACD hist, SMA pct, momentum, volume ratio, ATR |
| Technical signals | 10–14 | Trend, RSI signal, MACD signal, OBV, S/R position |
| Risk metrics | 15–22 | Ann. vol, max drawdown, VaR 95, Sharpe, win rate, G/L ratio, risk level, vol regime |
| Sentiment | 23–27 | News sentiment, count, political impact, LLM bias, positive ratio |
| Reddit | 28–31 | Available flag, retail sentiment, score, WSB trending |
| Fundamental | 32–37 | P/E, PEG, 52-week position, valuation, beta, P/E assessment |
| Portfolio scores | 38–40 | Technical score, sentiment score, composite score |
| Market context | 41–43 | Log price, market cap tier, momentum divergence |

---

## 🗂️ Project Structure

```
tick_talker/
├── main.py                    # CLI entrypoint
├── workflow.py                # LangGraph orchestrator
├── config.py                  # Tickers, analysis params
├── llm_config.py              # LLM provider routing
├── agents/
│   ├── technical_agent.py
│   ├── sentiment_agent.py
│   ├── reddit_agent.py
│   ├── fundamental_agent.py
│   ├── risk_agent.py
│   ├── portfolio_agent.py
│   └── signal_extractor.py    # AgentState → 44-dim vector
├── rl/
│   ├── recommendation_logger.py  # JSONL log + outcome labelling
│   ├── dqn_policy.py             # QNetwork, ReplayBuffer, OfflineDQNTrainer
│   ├── rl_portfolio_agent.py     # RLPortfolioDecider inference wrapper
│   └── train.py                  # CLI training script
├── data_sources/              # yfinance, Alpha Vantage, FRED, news, Reddit
├── scanners/                  # Day trading & penny stock scanners
└── logs/
    └── recommendations.jsonl  # RL training data (auto-created, gitignored)
```

---

## 📊 API Keys

| API | Link | Free Tier |
|-----|------|-----------|
| Alpha Vantage | https://www.alphavantage.co/support/#api-key | 25 req/day |
| FRED | https://fred.stlouisfed.org/docs/api/api_key.html | Unlimited |
| StockData.org | https://www.stockdata.org | 100 req/day |
| Reddit | https://www.reddit.com/prefs/apps | 60 req/min |
| Anthropic | https://console.anthropic.com | Pay-per-token |
| Groq | https://console.groq.com | Free tier |
| Gemini | https://aistudio.google.com/app/apikey | Free tier |

---

## ⚠️ Disclaimer

**Educational and research purposes only.** Not financial advice. No guarantee of accuracy or profitability. Always do your own research. Consult a financial advisor before investing real money.

---

## 📄 License

MIT License
