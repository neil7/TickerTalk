# 🤖 TickTalker - Agentic AI Stock Analysis System

An advanced multi-agent AI system for stock market analysis and prediction, powered by **Ollama** (local LLMs) and **LangGraph** (agent orchestration).

## 🎯 Features

### Multi-Agent Architecture
- **Data Collection Agent**: Aggregates data from stock APIs, news sources, FRED, and social media
- **Technical Analysis Agent**: RSI, MACD, Bollinger Bands, support/resistance, pattern recognition
- **Sentiment Analysis Agent**: News sentiment, White House briefings, Reddit/social analysis
- **Fundamental Analysis Agent**: P/E ratios, valuations, economic environment impact
- **Risk Management Agent**: Volatility metrics, VaR, position sizing, stop-loss recommendations
- **Portfolio Manager Agent**: Synthesizes all analyses for final recommendations

### Data Sources
| Source | Type | Cost |
|--------|------|------|
| Yahoo Finance | Stock data, fundamentals | Free |
| Alpha Vantage | Real-time data, technicals | Free tier |
| FRED (Federal Reserve) | Economic indicators | Free |
| White House RSS | Political news | Free |
| StockData.org | News with sentiment | Free tier (100/day) |
| Reddit (PRAW) | Social sentiment | Free |
| Google Search | Additional context | Free |

### Analysis Capabilities
- Individual stock deep analysis
- Market scanner for top movers
- **Penny Stock Scanner**: Finds high-potential (>100% gain) stocks under $5
- Day Trading Scanner: Identifies short-term setups with buy/sell levels
- Momentum candidate identification
- Event impact analysis
- Quick scan for multiple stocks
- Support for all US-listed stocks

## 📋 Prerequisites

1. **Python 3.9+**
2. **Ollama** - Local LLM runtime
3. **API Keys** (Free tiers available for all)

## 🚀 Quick Start

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model (in a new terminal)
ollama pull llama3.2:latest
```

### 2. Clone and Setup

```bash
cd tick_talker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file in the project root:

```env
# Required for full functionality
ALPHA_VANTAGE_KEY=your_key_here
FRED_API_KEY=your_key_here

# Optional but recommended
STOCKDATA_KEY=your_key_here
NEWS_API_KEY=your_key_here

# Optional: Reddit integration
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret

# Ollama settings (defaults shown)
OLLAMA_MODEL=llama3.2:latest
OLLAMA_BASE_URL=http://localhost:11434
```

#### Getting Free API Keys:

| API | Link | Notes |
|-----|------|-------|
| Alpha Vantage | https://www.alphavantage.co/support/#api-key | 25 requests/day free |
| FRED | https://fred.stlouisfed.org/docs/api/api_key.html | Completely free |
| StockData.org | https://www.stockdata.org | 100 requests/day free |
| NewsAPI | https://newsapi.org | 100 requests/day free |
| Reddit | https://www.reddit.com/prefs/apps | Free, 60 req/min |

#### Reddit API Setup (for WSB & Retail Sentiment)

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Fill in:
   - **Name**: TickTalker (or any name)
   - **App type**: Select "script"
   - **Redirect URI**: http://localhost
4. Click "Create App"
5. Copy your credentials:
   - **client_id**: The string under your app name
   - **client_secret**: The "secret" field
6. Add to your `.env`:
   ```
   REDDIT_CLIENT_ID=your_client_id_here
   REDDIT_CLIENT_SECRET=your_client_secret_here
   ```

### 4. Run TickTalker

```bash
# Analyze a specific stock
python main.py AAPL

# Day trading scanner (RECOMMENDED - shows buy/sell prices!)
python main.py --day-trade

# Scan market for movers
python main.py --scan

# Quick scan multiple stocks
python main.py --quick AAPL MSFT GOOGL NVDA

# Analyze event impact
python main.py --event "Fed raises interest rates" JPM BAC GS

# Scan blue chip stocks
python main.py --blue-chips

# Scan WallStreetBets for trending stocks
python main.py --wsb

# Show recommended LLM models
python main.py --models

# Interactive mode
python main.py
```

## 📚 Documentation

- **[TRADING_GUIDE.md](TRADING_GUIDE.md)** - Complete guide to understanding LONG and SHORT trades
- **[SHORT_SELLING_EXPLAINED.md](SHORT_SELLING_EXPLAINED.md)** - Detailed explanation of how short selling works
- **See these guides if you're confused about the trading recommendations!**

## 📊 Usage Examples

### Analyze a Stock
```bash
python main.py TSLA
```

Output includes:
- Market data summary
- Technical analysis with indicators
- Sentiment analysis from news
- Fundamental valuation assessment
- Risk metrics and position sizing
- Final BUY/HOLD/SELL recommendation

### Scan for Opportunities
```bash
python main.py --scan
```

Shows:
- Top gainers of the day
- Top losers of the day
- Momentum candidates
- Sector performance
- Option to deep-dive into top mover

### Penny Stock Scanner (Potential 2x Gains)
```bash
python main.py --penny
```
Finds "sleeper" penny stocks (price < $5) with high volatility and oversold conditions that have the potential to double in value.

### Quick Multi-Stock Scan
```bash
python main.py --quick AAPL MSFT AMZN GOOGL META
```

Rapid assessment of multiple stocks with:
- Current price and change
- Quick AI assessment
- Worth-analyzing verdict

## 🏗️ Architecture

```
tick_talker/
├── main.py                 # CLI entry point
├── workflow.py             # LangGraph orchestration
├── config.py               # Configuration management
├── requirements.txt        # Dependencies
│
├── data_sources/           # Data collection modules
│   ├── stock_data.py       # Stock prices, fundamentals
│   ├── news_data.py        # News aggregation
│   ├── economic_data.py    # FRED economic indicators
│   ├── social_data.py      # Reddit, trends
│   └── market_movers.py    # Top movers scanning
│
└── agents/                 # AI analysis agents
    ├── base_agent.py       # Base agent class
    ├── technical_agent.py  # Technical analysis
    ├── sentiment_agent.py  # News sentiment analysis
    ├── reddit_agent.py     # Reddit/WSB sentiment
    ├── fundamental_agent.py # Fundamental analysis
    ├── risk_agent.py       # Risk assessment
    └── portfolio_agent.py  # Final recommendations
```

### Agent Workflow

```
                    ┌─────────────────────┐
                    │   Data Collection   │
                    └──────────┬──────────┘
                               │
           ┌──────────────┬──────────────┬──────────────┐
           │              │              │              │
           ▼              ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │Technical │   │  News    │   │  Reddit  │   │Fundament │
    │ Analysis │   │Sentiment │   │Sentiment │   │ Analysis │
    └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │              │
         ▼              │              │              │
    ┌──────────┐        │              │              │
    │   Risk   │        │              │              │
    │Management│        │              │              │
    └────┬─────┘        │              │              │
         │              │              │              │
         └──────────────┴──────────────┴──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Portfolio Manager  │
                    │  (Final Decision)   │
                    └─────────────────────┘
```

## 📈 Additional Data Points to Consider

Beyond the implemented sources, consider adding:

| Category | Data Points |
|----------|-------------|
| **Options** | Unusual options activity, put/call ratios, implied volatility |
| **Institutional** | 13F filings, ETF flows, dark pool activity |
| **Insider Trading** | SEC Form 4 filings, executive transactions |
| **Commodities** | Oil, gold, copper correlations |
| **Macro** | VIX, Treasury yield curve, dollar index |
| **Alternative** | Google Trends, satellite imagery, web traffic |
| **Crypto** | Bitcoin correlation for tech stocks |
| **Congress** | STOCK Act disclosures |

## ⚠️ Limitations

### Twitter/X API
- **No free tier** available for useful access
- Basic tier: $200/month (100 posts, 10K reads)
- Recommendation: Use news APIs and Reddit as alternatives

### Truth Social
- **No official API** available
- Options: Third-party scrapers (unreliable), RSS from news sites
- Recommendation: Monitor news sources that cover Truth Social posts

## 🔧 Customization

### Change LLM Model

Edit `.env` or `config.py`:
```python
OLLAMA_MODEL=mistral:latest  # or phi3:latest, llama2:latest, etc.
```

Available models:
- `llama3.2:latest` - Best balance of quality/speed
- `mistral:latest` - Good for analysis
- `phi3:latest` - Faster, smaller
- `codellama:latest` - If adding code analysis

### Adjust Analysis Parameters

Edit `config.py`:
```python
class AnalysisConfig:
    rsi_period: int = 14
    sma_short: int = 20
    sma_long: int = 50
    max_position_size: float = 0.05  # 5% max per position
    default_stop_loss: float = 0.05  # 5% stop loss
```

### Add Custom Stocks to Monitor

Edit `config.py`:
```python
BLUE_CHIP_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    # Add your stocks here
]
```

## 📜 Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- No guarantee of accuracy or profitability
- Past performance does not indicate future results
- Always do your own research
- Consult a financial advisor for investment decisions
- Comply with all applicable securities regulations

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional data sources
- More sophisticated technical indicators
- Machine learning price prediction
- Backtesting framework
- Real-time streaming support
- Web UI dashboard

## 📄 License

MIT License - See LICENSE file for details.

---

**Built with ❤️ using Ollama, LangGraph, and Python**

