"""
Configuration management for TickTalker AI Stock Analysis System
"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

# Load environment variables
load_dotenv()


@dataclass
class APIConfig:
    """API configuration settings"""
    # Stock Data APIs
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_KEY", "demo")
    stockdata_key: Optional[str] = os.getenv("STOCKDATA_KEY")
    
    # Economic Data
    fred_api_key: Optional[str] = os.getenv("FRED_API_KEY")
    
    # News & Social
    news_api_key: Optional[str] = os.getenv("NEWS_API_KEY")
    reddit_client_id: Optional[str] = os.getenv("REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = os.getenv("REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = os.getenv("REDDIT_USER_AGENT", "TickTalker/1.0")
    
    # LLM Provider
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")  # "ollama", "gemini", "groq", or "anthropic"
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")


@dataclass
class GeminiConfig:
    """
    Google Gemini LLM configuration
    """
    model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    temperature: float = 0.3


@dataclass
class GroqConfig:
    """
    Groq LLM configuration
    """
    model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    temperature: float = 0.3


@dataclass
class AnthropicConfig:
    """
    Anthropic Claude LLM configuration
    """
    model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    temperature: float = 0.3


@dataclass
class OllamaConfig:
    """
    Ollama LLM configuration
    
    Recommended models for trading analysis:
    - llama3.2:latest  - Best for trading (provides actionable advice)
    - mistral:7b       - Fast, good alternative
    - llama3.1:8b      - Good balance
    
    Note: Qwen models may refuse to give trading advice
    
    Install with: ollama pull llama3.2:latest
    """
    model: str = os.getenv("OLLAMA_MODEL", "llama3.2:latest")  # Best for trading
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature: float = 0.3  # Higher for more actionable advice
    num_ctx: int = 8192  # Larger context for comprehensive analysis


@dataclass
class AnalysisConfig:
    """Analysis parameters"""
    # Technical Analysis
    rsi_period: int = 14
    sma_short: int = 20
    sma_long: int = 50
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    
    # Data Fetching
    historical_days: int = 90
    news_limit: int = 15
    max_market_movers: int = 50
    
    # Risk Management
    max_position_size: float = 0.05  # 5% of portfolio
    default_stop_loss: float = 0.05  # 5% stop loss
    
    # Cache settings (in seconds)
    cache_ttl_stock: int = 300  # 5 minutes
    cache_ttl_economic: int = 86400  # 24 hours
    cache_ttl_news: int = 1800  # 30 minutes


# RSS Feed URLs
RSS_FEEDS = {
    "white_house": "https://www.whitehouse.gov/feed/",
    "white_house_briefings": "https://www.whitehouse.gov/briefing-room/feed/",
    "federal_register": "https://www.federalregister.gov/documents/current.rss",
    "sec_filings": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=include&count=40&output=atom",
}

# Economic Indicators from FRED
FRED_INDICATORS = {
    "CPIAUCSL": "Consumer Price Index (Inflation)",
    "UNRATE": "Unemployment Rate",
    "GDP": "Gross Domestic Product",
    "FEDFUNDS": "Federal Funds Rate",
    "DGS10": "10-Year Treasury Yield",
    "UMCSENT": "Consumer Sentiment",
    "VIXCLS": "VIX Volatility Index",
    "DTWEXBGS": "Trade Weighted Dollar Index",
    "PPIACO": "Producer Price Index",
    "PAYEMS": "Total Nonfarm Payrolls",
}

# Blue Chip Stocks for monitoring
BLUE_CHIP_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Tech Giants
    "NVDA", "TSLA", "AMD", "INTC",  # Semiconductors & EV
    "JPM", "BAC", "GS", "WFC",  # Banks
    "JNJ", "UNH", "PFE", "MRK",  # Healthcare
    "XOM", "CVX", "COP",  # Energy
    "WMT", "HD", "COST",  # Retail
    "V", "MA", "AXP",  # Payments
    "DIS", "NFLX",  # Entertainment
]

# Sector ETFs for market analysis
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
}

# Global config instances
api_config = APIConfig()
ollama_config = OllamaConfig()
gemini_config = GeminiConfig()
groq_config = GroqConfig()
anthropic_config = AnthropicConfig()
analysis_config = AnalysisConfig()
