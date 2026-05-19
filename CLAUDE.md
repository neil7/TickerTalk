# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

Setup (one-time):
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Run the CLI (all entry points are `main.py`):
```bash
python main.py AAPL                       # full multi-agent analysis of one ticker
python main.py --quick AAPL MSFT NVDA     # lightweight scan (no full graph)
python main.py --scan                     # market movers + sectors + momentum
python main.py --day-trade                # DayTradingScanner (buy/sell levels)
python main.py --penny                    # PennyStockScanner (<$5, oversold)
python main.py --wsb                      # WallStreetBets trending tickers
python main.py --blue-chips               # iterate BLUE_CHIP_TICKERS from config.py
python main.py --event "Fed hikes" JPM    # event-impact analysis for one+ tickers
python main.py --models                   # print recommended LLM models
python main.py                            # interactive menu
```

There is no test suite, linter config, or build system in this repo. There is nothing to run before committing besides the CLI itself.

## Environment

A `.env` file in the repo root is loaded by `config.py` via `python-dotenv`. The only knob that changes runtime behavior across the codebase is `LLM_PROVIDER` (`ollama` | `gemini` | `groq` | `anthropic`); each provider then reads its own model + key env vars (e.g. `ANTHROPIC_API_KEY` + `ANTHROPIC_MODEL`). Data-source keys (`ALPHA_VANTAGE_KEY`, `FRED_API_KEY`, `STOCKDATA_KEY`, `NEWS_API_KEY`, `REDDIT_CLIENT_ID/SECRET`) are all optional — fetchers degrade gracefully when missing.

For Ollama, the daemon must be running locally (`ollama serve`) and the model pulled (`ollama pull llama3.2:latest`).

## Architecture

The system is a LangGraph-orchestrated multi-agent pipeline. The "big picture" requires understanding three layers that don't live in one file:

### 1. Shared state (`workflow.py`)
`AgentState` is a `TypedDict` acting as a blackboard passed through every node. Each agent reads from and writes to specific keys (`market_data`, `technical_analysis`, `sentiment_analysis`, `reddit_analysis`, `fundamental_analysis`, `risk_assessment`, `final_recommendation`). **The graph edges are sequential, not parallel** (`workflow.py:88-98`) — this is intentional to avoid LangGraph's concurrent state-reducer conflicts. If you add a node, wire it sequentially or you will get state-merge errors.

Execution order: `data_collection → technical → risk → sentiment → reddit → fundamental → portfolio_manager → END`.

### 2. Pluggable LLM backend (`agents/base_agent.py`)
Every agent inherits from `BaseAgent`, whose `__init__` reads `api_config.llm_provider` and instantiates one of `ChatAnthropic` / `ChatGroq` / `ChatGoogleGenerativeAI` / `OllamaLLM`. **On any failure (missing key, missing import) it silently falls back to Ollama.** This means a misconfigured provider does not crash — agents just quietly run on the local model. When debugging "why is the output different from expected," check which backend actually initialized.

Agents invoke the LLM via `self._invoke_llm(prompt, **kwargs)`, which builds a `prompt | llm | StrOutputParser()` LCEL chain. There is no structured-output schema; agents parse free-text responses themselves.

### 3. Two execution paths
- **Full pipeline**: `workflow.StockAnalysisWorkflow.analyze(ticker)` runs the LangGraph for a single ticker. Slow (6 LLM calls). Used by `analyze_stock()` in `main.py`.
- **Lightweight paths**: `workflow.quick_scan()` and the scanners under `scanners/` skip the graph entirely — they call the data fetchers and then `portfolio_agent.quick_assessment()` (a single LLM call) per ticker. Used for bulk scanning where running the full graph N times would be too slow/expensive.

When adding a new feature, decide which path it belongs on. Don't put bulk-scan features behind the full graph.

### Data sources (`data_sources/`)
Pure fetchers — no LLM calls. Each returns dicts/lists ready to be dropped into `AgentState`. `NewsFetcher.get_all_news_for_analysis()` is the aggregator that fans out to all news/RSS/Google sources and shapes the dict that becomes `state["sentiment_data"]`. `MarketMoversFetcher` powers `--scan`; `RedditDataFetcher` powers `--wsb` and feeds the reddit agent.

### Model recommendations (`llm_config.py`)
`RECOMMENDED_MODELS` and `USE_CASE_RECOMMENDATIONS` are reference tables surfaced by `--models`. They are not wired into agent selection — they exist to help the user pick a value for `OLLAMA_MODEL` / `GROQ_MODEL` / etc.

## Conventions worth knowing

- LLM temperature is hard-coded per provider (~0.1–0.3) for consistent financial output. Don't bump it without a reason.
- The Anthropic default in `config.py` is `claude-sonnet-4-6` — this matches the latest available Sonnet at the time of writing. When changing model IDs, also update `llm_config.RECOMMENDED_MODELS` so `--models` stays accurate.
- The Reddit agent depends on `praw` being configured; without `REDDIT_CLIENT_ID/SECRET` the reddit node sets `available: false` and other agents skip it cleanly. Preserve this graceful-degradation pattern for any new optional source.
- `BLUE_CHIP_TICKERS`, `SECTOR_ETFS`, `RSS_FEEDS`, and `FRED_INDICATORS` are global constants in `config.py` — extend them there rather than hardcoding in agents/fetchers.
