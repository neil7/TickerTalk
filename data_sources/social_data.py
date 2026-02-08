"""
Social Media Data Fetcher
Handles Reddit, Google Trends, and other social sentiment sources
Note: Twitter/X API requires paid subscription ($200/month)
      Truth Social has no official API
"""
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from config import api_config


class SocialDataFetcher:
    """Fetches social media sentiment data"""
    
    def __init__(self):
        self.reddit_client_id = api_config.reddit_client_id
        self.reddit_client_secret = api_config.reddit_client_secret
        self.reddit_user_agent = api_config.reddit_user_agent
        self.reddit = None
        self._init_reddit()
    
    def _init_reddit(self):
        """Initialize Reddit API client (PRAW)"""
        if self.reddit_client_id and self.reddit_client_secret:
            try:
                import praw
                self.reddit = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent
                )
            except ImportError:
                self.reddit = None
    
    def get_wallstreetbets_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch WallStreetBets mentions and sentiment for a ticker
        WSB can be a leading indicator for retail momentum
        """
        if not self.reddit:
            return {"error": "Reddit API not configured"}
        
        try:
            subreddit = self.reddit.subreddit("wallstreetbets")
            
            mentions = []
            total_score = 0
            bullish_count = 0
            bearish_count = 0
            
            # Search for ticker mentions
            for submission in subreddit.search(ticker, limit=50, time_filter="week"):
                title_lower = submission.title.lower()
                selftext_lower = (submission.selftext or "").lower()
                
                # Simple sentiment detection
                bullish_words = ["buy", "long", "calls", "moon", "rocket", "bull", "yolo", "diamond hands"]
                bearish_words = ["sell", "short", "puts", "crash", "bear", "dump", "paper hands"]
                
                is_bullish = any(word in title_lower or word in selftext_lower for word in bullish_words)
                is_bearish = any(word in title_lower or word in selftext_lower for word in bearish_words)
                
                if is_bullish:
                    bullish_count += 1
                if is_bearish:
                    bearish_count += 1
                
                total_score += submission.score
                
                mentions.append({
                    "title": submission.title[:100],
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created": datetime.fromtimestamp(submission.created_utc).isoformat(),
                    "sentiment": "bullish" if is_bullish and not is_bearish else "bearish" if is_bearish and not is_bullish else "neutral",
                })
            
            # Calculate overall sentiment
            total_mentions = len(mentions)
            if total_mentions == 0:
                sentiment = "no_mentions"
            elif bullish_count > bearish_count * 1.5:
                sentiment = "bullish"
            elif bearish_count > bullish_count * 1.5:
                sentiment = "bearish"
            else:
                sentiment = "mixed"
            
            return {
                "ticker": ticker,
                "subreddit": "wallstreetbets",
                "total_mentions": total_mentions,
                "bullish_mentions": bullish_count,
                "bearish_mentions": bearish_count,
                "total_upvotes": total_score,
                "overall_sentiment": sentiment,
                "top_posts": mentions[:10],
                "fetch_time": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_investing_subreddit_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Fetch sentiment from r/investing and r/stocks"""
        if not self.reddit:
            return {"error": "Reddit API not configured"}
        
        try:
            all_mentions = []
            
            for sub_name in ["investing", "stocks"]:
                subreddit = self.reddit.subreddit(sub_name)
                
                for submission in subreddit.search(ticker, limit=25, time_filter="week"):
                    all_mentions.append({
                        "subreddit": sub_name,
                        "title": submission.title[:100],
                        "score": submission.score,
                        "num_comments": submission.num_comments,
                        "created": datetime.fromtimestamp(submission.created_utc).isoformat(),
                    })
            
            # Sort by score
            all_mentions.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "ticker": ticker,
                "total_mentions": len(all_mentions),
                "top_posts": all_mentions[:10],
                "fetch_time": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_google_trends(self, keyword: str) -> Dict[str, Any]:
        """
        Get Google Trends data for a keyword
        Rising search interest can indicate upcoming momentum
        
        Note: This uses unofficial method - may be rate limited
        """
        try:
            from pytrends.request import TrendReq
            
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload([keyword], cat=0, timeframe='now 7-d')
            
            interest_over_time = pytrends.interest_over_time()
            
            if interest_over_time.empty:
                return {"error": "No trends data available"}
            
            values = interest_over_time[keyword].tolist()
            dates = [str(d) for d in interest_over_time.index]
            
            # Calculate trend direction
            if len(values) >= 2:
                recent_avg = sum(values[-3:]) / 3
                earlier_avg = sum(values[:3]) / 3
                trend = "rising" if recent_avg > earlier_avg * 1.1 else "falling" if recent_avg < earlier_avg * 0.9 else "stable"
            else:
                trend = "unknown"
            
            return {
                "keyword": keyword,
                "current_interest": values[-1] if values else 0,
                "trend_direction": trend,
                "values": values,
                "dates": dates,
                "fetch_time": datetime.now().isoformat(),
            }
        except ImportError:
            return {"error": "pytrends not installed. Run: pip install pytrends"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_twitter_x_notice(self) -> Dict[str, Any]:
        """
        Returns notice about Twitter/X API pricing
        Free tier is no longer available for useful access
        """
        return {
            "notice": "Twitter/X API requires paid subscription",
            "pricing": {
                "basic": "$200/month - 100 posts/month, 10K reads/month",
                "pro": "$5000/month - Higher limits",
            },
            "recommendation": "Consider using news APIs and Reddit as alternatives for social sentiment",
            "alternatives": [
                "StockData.org (includes sentiment)",
                "Reddit API (free)",
                "Google Trends (free)",
                "News API aggregators",
            ]
        }
    
    def get_truth_social_notice(self) -> Dict[str, Any]:
        """
        Returns notice about Truth Social API access
        No official API available
        """
        return {
            "notice": "Truth Social has no official API",
            "options": [
                "Third-party scrapers (paid, unreliable)",
                "RSS feeds from news sites covering Truth Social posts",
                "Manual monitoring",
            ],
            "recommendation": "Monitor news sources that cover Trump's Truth Social posts"
        }
    
    def get_social_summary(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive social sentiment summary
        """
        summary = {
            "ticker": ticker,
            "fetch_time": datetime.now().isoformat(),
        }
        
        # Reddit data
        if self.reddit:
            summary["wallstreetbets"] = self.get_wallstreetbets_sentiment(ticker)
            summary["investing_subreddits"] = self.get_investing_subreddit_sentiment(ticker)
        else:
            summary["reddit_status"] = "Reddit API not configured"
        
        # Google Trends
        summary["google_trends"] = self.get_google_trends(f"{ticker} stock")
        
        # API status notices
        summary["twitter_status"] = self.get_twitter_x_notice()
        summary["truth_social_status"] = self.get_truth_social_notice()
        
        return summary

