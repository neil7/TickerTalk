"""
News Data Fetcher
Handles news from multiple sources including White House, financial news, and RSS feeds
"""
import feedparser
import requests
import yfinance as yf
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from config import api_config, analysis_config, RSS_FEEDS


class NewsFetcher:
    """Fetches news from various sources for sentiment analysis"""
    
    def __init__(self):
        self.stockdata_key = api_config.stockdata_key
        self.news_api_key = api_config.news_api_key
        
    def get_stock_news(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Fetch stock-specific news from multiple sources
        
        Returns combined news from StockData API and Yahoo Finance
        """
        news_items = []
        
        # Try StockData.org API first
        if self.stockdata_key:
            stockdata_news = self._fetch_stockdata_news(ticker)
            news_items.extend(stockdata_news)
        
        # Fallback/supplement with Yahoo Finance news
        yahoo_news = self._fetch_yahoo_news(ticker)
        news_items.extend(yahoo_news)
        
        # Remove duplicates based on title similarity
        seen_titles = set()
        unique_news = []
        for item in news_items:
            title_lower = item.get('title', '').lower()[:50]
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_news.append(item)
        
        return unique_news[:analysis_config.news_limit]
    
    def _fetch_stockdata_news(self, ticker: str) -> List[Dict]:
        """Fetch news from StockData.org API"""
        try:
            url = "https://api.stockdata.org/v1/news/all"
            params = {
                "symbols": ticker,
                "filter_entities": "true",
                "language": "en",
                "api_token": self.stockdata_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                news = []
                for item in data.get('data', []):
                    news.append({
                        "source": "stockdata",
                        "title": item.get('title', ''),
                        "description": item.get('description', '')[:300],
                        "url": item.get('url', ''),
                        "published": item.get('published_at', ''),
                        "sentiment": item.get('sentiment', None),  # StockData provides sentiment
                        "entities": item.get('entities', []),
                    })
                return news
            return []
        except Exception as e:
            return []
    
    def _fetch_yahoo_news(self, ticker: str) -> List[Dict]:
        """Fetch news from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news if hasattr(stock, 'news') else []
            
            result = []
            for item in news[:10]:
                result.append({
                    "source": "yahoo",
                    "title": item.get('title', ''),
                    "description": item.get('summary', '')[:300] if item.get('summary') else '',
                    "url": item.get('link', ''),
                    "published": datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat() if item.get('providerPublishTime') else '',
                    "publisher": item.get('publisher', ''),
                    "thumbnail": item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', '') if item.get('thumbnail') else '',
                })
            
            return result
        except Exception as e:
            return []
    
    def get_white_house_news(self) -> List[Dict[str, Any]]:
        """
        Fetch White House press releases and briefings
        Critical for policy-related market impacts
        """
        all_news = []
        
        # Fetch from main White House feed
        main_news = self._fetch_rss_feed(
            RSS_FEEDS["white_house"], 
            "White House"
        )
        all_news.extend(main_news)
        
        # Fetch from briefing room
        briefings = self._fetch_rss_feed(
            RSS_FEEDS["white_house_briefings"],
            "White House Briefing Room"
        )
        all_news.extend(briefings)
        
        return all_news[:15]
    
    def get_federal_news(self) -> List[Dict[str, Any]]:
        """Fetch Federal Register announcements"""
        return self._fetch_rss_feed(
            RSS_FEEDS["federal_register"],
            "Federal Register"
        )[:10]
    
    def get_sec_filings(self) -> List[Dict[str, Any]]:
        """Fetch recent SEC filings"""
        return self._fetch_rss_feed(
            RSS_FEEDS["sec_filings"],
            "SEC"
        )[:10]
    
    def _fetch_rss_feed(self, url: str, source_name: str) -> List[Dict]:
        """Generic RSS feed fetcher"""
        try:
            feed = feedparser.parse(url)
            articles = []
            
            for entry in feed.entries[:15]:
                # Extract and clean summary
                summary = entry.get('summary', entry.get('description', ''))
                if summary:
                    soup = BeautifulSoup(summary, 'html.parser')
                    summary = soup.get_text()[:300]
                
                articles.append({
                    "source": source_name,
                    "title": entry.get('title', 'No title'),
                    "url": entry.get('link', ''),
                    "published": entry.get('published', entry.get('updated', '')),
                    "summary": summary,
                    "categories": [tag.term for tag in entry.get('tags', [])] if hasattr(entry, 'tags') and entry.tags else [],
                })
            
            return articles
        except Exception as e:
            return []
    
    def get_general_market_news(self) -> List[Dict[str, Any]]:
        """
        Fetch general market news from NewsAPI
        Covers broader market trends and events
        """
        if not self.news_api_key:
            return []
        
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                "category": "business",
                "country": "us",
                "apiKey": self.news_api_key,
                "pageSize": 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for item in data.get('articles', []):
                    articles.append({
                        "source": item.get('source', {}).get('name', 'Unknown'),
                        "title": item.get('title', ''),
                        "description": item.get('description', '')[:300] if item.get('description') else '',
                        "url": item.get('url', ''),
                        "published": item.get('publishedAt', ''),
                        "author": item.get('author', ''),
                    })
                
                return articles
            return []
        except Exception as e:
            return []
    
    def google_search_news(self, query: str, num_results: int = 5) -> List[str]:
        """
        Perform Google search for additional context
        Useful for finding recent news not covered by APIs
        """
        try:
            from googlesearch import search
            
            # Add time filter to query for recent results
            enhanced_query = f"{query} news {datetime.now().strftime('%Y')}"
            
            results = []
            for url in search(enhanced_query, num_results=num_results):
                results.append(url)
            
            return results
        except ImportError:
            return []
        except Exception as e:
            return []
    
    def fetch_url_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract text content from a URL
        Useful for deep analysis of specific articles
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove scripts and styles
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text(separator=' ', strip=True)
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:5000]  # Limit to 5000 chars
            
            return None
        except Exception as e:
            return None
    
    def get_all_news_for_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Aggregate all news sources for comprehensive analysis
        """
        return {
            "stock_specific": self.get_stock_news(ticker),
            "white_house": self.get_white_house_news(),
            "federal": self.get_federal_news(),
            "market_general": self.get_general_market_news(),
            "google_results": self.google_search_news(f"{ticker} stock news"),
            "fetch_time": datetime.now().isoformat(),
        }

