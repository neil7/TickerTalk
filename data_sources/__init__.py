"""
Data Sources Module - Handles all external data fetching
"""
from .stock_data import StockDataFetcher
from .news_data import NewsFetcher
from .economic_data import EconomicDataFetcher
from .social_data import SocialDataFetcher
from .market_movers import MarketMoversFetcher
from .reddit_data import RedditDataFetcher

__all__ = [
    "StockDataFetcher",
    "NewsFetcher", 
    "EconomicDataFetcher",
    "SocialDataFetcher",
    "MarketMoversFetcher",
    "RedditDataFetcher",
]

