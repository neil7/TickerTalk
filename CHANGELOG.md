# Changelog

All notable changes to TickerTalk are documented here.

## [0.0.0.1] - 2026-05-18

### Added
- `.env.example` template covering all supported environment variables: LLM provider
  selection (`LLM_PROVIDER`), provider-specific keys and models for Anthropic, Ollama,
  Groq, and Gemini, plus optional data-source keys for Alpha Vantage, FRED, News API,
  Reddit, and StockData. Copy to `.env` and fill in values to get started.
- `CLAUDE.md` project guide documenting CLI entry points, environment configuration,
  architecture overview (LangGraph pipeline, agent execution order, pluggable LLM
  backend, dual execution paths), and conventions for working in this repo.
