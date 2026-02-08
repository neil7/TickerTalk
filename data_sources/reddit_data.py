"""
Reddit Data Fetcher
Fetches sentiment and discussion data from investing-related subreddits
Free tier: 60 requests/minute for personal/script apps
"""
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import Counter
from config import api_config


class RedditDataFetcher:
    """
    Fetches data from Reddit investing communities
    
    Subreddits monitored:
    - r/wallstreetbets - Retail trading sentiment, meme stocks
    - r/stocks - General stock discussion
    - r/investing - Long-term investment discussion
    - r/options - Options trading discussion
    - r/stockmarket - Market news and analysis
    - r/dividends - Dividend investing
    - r/ValueInvesting - Value investing strategies
    """
    
    # Subreddits to monitor (in order of relevance for short-term trading)
    TRADING_SUBREDDITS = [
        "wallstreetbets",
        "stocks", 
        "options",
        "stockmarket",
    ]
    
    INVESTING_SUBREDDITS = [
        "investing",
        "dividends",
        "ValueInvesting",
    ]
    
    # Sentiment keywords
    BULLISH_KEYWORDS = [
        "buy", "long", "calls", "moon", "rocket", "bull", "yolo", 
        "diamond hands", "hold", "hodl", "breakout", "undervalued",
        "upside", "pump", "rip", "tendies", "gain", "green",
        "bullish", "mooning", "squeeze", "to the moon", "🚀", "💎", "🙌"
    ]
    
    BEARISH_KEYWORDS = [
        "sell", "short", "puts", "crash", "bear", "dump", "drop",
        "paper hands", "overvalued", "downside", "tank", "red",
        "bearish", "drilling", "loss", "bag holder", "bagholding",
        "dead", "rip", "falling knife", "📉", "🐻"
    ]
    
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
                # Test connection
                self.reddit.user.me()
            except Exception as e:
                # Read-only mode (no auth needed for public subreddits)
                try:
                    import praw
                    self.reddit = praw.Reddit(
                        client_id=self.reddit_client_id,
                        client_secret=self.reddit_client_secret,
                        user_agent=self.reddit_user_agent,
                    )
                except:
                    self.reddit = None
    
    def is_available(self) -> bool:
        """Check if Reddit API is available"""
        return self.reddit is not None
    
    def get_ticker_mentions(self, ticker: str, subreddits: List[str] = None, 
                           time_filter: str = "week", limit: int = 100) -> Dict[str, Any]:
        """
        Search for ticker mentions across subreddits
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            subreddits: List of subreddits to search (defaults to trading subs)
            time_filter: Time range (hour, day, week, month, year, all)
            limit: Maximum posts to fetch per subreddit
        """
        if not self.reddit:
            return {"error": "Reddit API not configured. Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env"}
        
        if subreddits is None:
            subreddits = self.TRADING_SUBREDDITS
        
        all_mentions = []
        subreddit_stats = {}
        
        ticker_upper = ticker.upper()
        ticker_pattern = rf'\b{ticker_upper}\b|\${ticker_upper}\b'
        
        for sub_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(sub_name)
                mentions = []
                
                # Search for ticker in subreddit
                for submission in subreddit.search(
                    ticker_upper, 
                    limit=limit, 
                    time_filter=time_filter,
                    sort="relevance"
                ):
                    # Verify ticker is actually mentioned (not just partial match)
                    title_match = re.search(ticker_pattern, submission.title.upper())
                    text_match = re.search(ticker_pattern, (submission.selftext or "").upper())
                    
                    if title_match or text_match:
                        mention = self._process_submission(submission, sub_name)
                        mentions.append(mention)
                
                all_mentions.extend(mentions)
                subreddit_stats[sub_name] = {
                    "mentions": len(mentions),
                    "total_score": sum(m["score"] for m in mentions),
                    "total_comments": sum(m["num_comments"] for m in mentions),
                }
                
            except Exception as e:
                subreddit_stats[sub_name] = {"error": str(e)}
        
        # Calculate overall statistics
        total_mentions = len(all_mentions)
        bullish_count = sum(1 for m in all_mentions if m["sentiment"] == "bullish")
        bearish_count = sum(1 for m in all_mentions if m["sentiment"] == "bearish")
        neutral_count = total_mentions - bullish_count - bearish_count
        
        # Determine overall sentiment
        if total_mentions == 0:
            overall_sentiment = "no_data"
            sentiment_score = 0
        else:
            sentiment_score = (bullish_count - bearish_count) / total_mentions
            if sentiment_score > 0.3:
                overall_sentiment = "bullish"
            elif sentiment_score < -0.3:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "mixed"
        
        return {
            "ticker": ticker_upper,
            "total_mentions": total_mentions,
            "bullish_mentions": bullish_count,
            "bearish_mentions": bearish_count,
            "neutral_mentions": neutral_count,
            "sentiment_score": round(sentiment_score, 3),
            "overall_sentiment": overall_sentiment,
            "subreddit_breakdown": subreddit_stats,
            "top_posts": sorted(all_mentions, key=lambda x: x["score"], reverse=True)[:10],
            "recent_posts": sorted(all_mentions, key=lambda x: x["created"], reverse=True)[:10],
            "time_filter": time_filter,
            "fetch_time": datetime.now().isoformat(),
        }
    
    def _process_submission(self, submission, subreddit: str) -> Dict[str, Any]:
        """Process a Reddit submission and extract relevant data"""
        title = submission.title
        selftext = submission.selftext or ""
        combined_text = f"{title} {selftext}".lower()
        
        # Determine sentiment
        bullish_hits = sum(1 for kw in self.BULLISH_KEYWORDS if kw.lower() in combined_text)
        bearish_hits = sum(1 for kw in self.BEARISH_KEYWORDS if kw.lower() in combined_text)
        
        if bullish_hits > bearish_hits:
            sentiment = "bullish"
        elif bearish_hits > bullish_hits:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        return {
            "subreddit": subreddit,
            "title": title[:200],
            "url": f"https://reddit.com{submission.permalink}",
            "score": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "created": datetime.fromtimestamp(submission.created_utc).isoformat(),
            "author": str(submission.author) if submission.author else "[deleted]",
            "sentiment": sentiment,
            "bullish_signals": bullish_hits,
            "bearish_signals": bearish_hits,
            "flair": submission.link_flair_text,
            "awards": submission.total_awards_received,
        }
    
    def get_wsb_daily_sentiment(self) -> Dict[str, Any]:
        """
        Get daily sentiment and trending tickers from WallStreetBets
        Useful for detecting retail momentum plays
        """
        if not self.reddit:
            return {"error": "Reddit API not configured"}
        
        try:
            wsb = self.reddit.subreddit("wallstreetbets")
            
            ticker_mentions = Counter()
            ticker_sentiment = {}
            posts_analyzed = 0
            
            # Common stock ticker pattern ($AAPL or just AAPL in caps)
            ticker_pattern = r'\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b'
            
            # Exclude common words that look like tickers
            excluded = {
                "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
                "CAN", "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "HAS",
                "HIS", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE",
                "WAY", "WHO", "BOY", "DID", "GET", "PUT", "SAY", "SHE",
                "TOO", "USE", "CEO", "CFO", "COO", "USA", "NYSE", "SEC",
                "ETF", "IPO", "GDP", "CPI", "FED", "IMO", "YOLO", "FOMO",
                "DD", "TL", "DR", "TLDR", "OP", "WSB", "ITM", "OTM", "ATM",
                "IV", "DTE", "EOD", "EOW", "EOM", "LMAO", "LMFAO", "LOL",
            }
            
            # Analyze hot posts
            for submission in wsb.hot(limit=50):
                posts_analyzed += 1
                combined = f"{submission.title} {submission.selftext or ''}"
                
                # Find ticker mentions
                matches = re.findall(ticker_pattern, combined)
                for match in matches:
                    ticker = match[0] or match[1]
                    if ticker and ticker not in excluded and len(ticker) >= 2:
                        ticker_mentions[ticker] += 1
                        
                        if ticker not in ticker_sentiment:
                            ticker_sentiment[ticker] = {"bullish": 0, "bearish": 0, "total_score": 0}
                        
                        # Simple sentiment from post
                        text_lower = combined.lower()
                        is_bullish = any(kw in text_lower for kw in self.BULLISH_KEYWORDS[:10])
                        is_bearish = any(kw in text_lower for kw in self.BEARISH_KEYWORDS[:10])
                        
                        if is_bullish:
                            ticker_sentiment[ticker]["bullish"] += 1
                        if is_bearish:
                            ticker_sentiment[ticker]["bearish"] += 1
                        ticker_sentiment[ticker]["total_score"] += submission.score
            
            # Get top mentioned tickers
            top_tickers = []
            for ticker, count in ticker_mentions.most_common(15):
                sentiment_data = ticker_sentiment.get(ticker, {})
                bullish = sentiment_data.get("bullish", 0)
                bearish = sentiment_data.get("bearish", 0)
                
                if bullish > bearish:
                    sentiment = "bullish"
                elif bearish > bullish:
                    sentiment = "bearish"
                else:
                    sentiment = "neutral"
                
                top_tickers.append({
                    "ticker": ticker,
                    "mentions": count,
                    "sentiment": sentiment,
                    "bullish_mentions": bullish,
                    "bearish_mentions": bearish,
                    "total_score": sentiment_data.get("total_score", 0),
                })
            
            return {
                "subreddit": "wallstreetbets",
                "posts_analyzed": posts_analyzed,
                "trending_tickers": top_tickers,
                "fetch_time": datetime.now().isoformat(),
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_subreddit_hot_posts(self, subreddit_name: str, limit: int = 25) -> List[Dict]:
        """Get hot posts from a specific subreddit"""
        if not self.reddit:
            return []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            for submission in subreddit.hot(limit=limit):
                posts.append({
                    "title": submission.title[:200],
                    "url": f"https://reddit.com{submission.permalink}",
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created": datetime.fromtimestamp(submission.created_utc).isoformat(),
                    "flair": submission.link_flair_text,
                })
            
            return posts
        except Exception as e:
            return []
    
    def get_dd_posts(self, ticker: str = None, limit: int = 10) -> List[Dict]:
        """
        Get Due Diligence posts (in-depth analysis posts)
        DD posts are often more insightful than regular posts
        """
        if not self.reddit:
            return []
        
        try:
            dd_posts = []
            
            for sub_name in ["wallstreetbets", "stocks", "investing"]:
                subreddit = self.reddit.subreddit(sub_name)
                
                # Search for DD flair or DD in title
                query = f"{ticker} flair:DD" if ticker else "flair:DD"
                
                for submission in subreddit.search(query, limit=limit, time_filter="month"):
                    if ticker:
                        # Verify ticker is mentioned
                        if ticker.upper() not in submission.title.upper():
                            continue
                    
                    dd_posts.append({
                        "subreddit": sub_name,
                        "title": submission.title[:200],
                        "url": f"https://reddit.com{submission.permalink}",
                        "score": submission.score,
                        "num_comments": submission.num_comments,
                        "created": datetime.fromtimestamp(submission.created_utc).isoformat(),
                        "author": str(submission.author) if submission.author else "[deleted]",
                        "preview": (submission.selftext or "")[:500],
                    })
            
            # Sort by score
            dd_posts.sort(key=lambda x: x["score"], reverse=True)
            return dd_posts[:limit]
            
        except Exception as e:
            return []
    
    def get_comprehensive_reddit_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive Reddit analysis for a ticker
        Combines data from multiple sources
        """
        result = {
            "ticker": ticker.upper(),
            "reddit_available": self.is_available(),
            "fetch_time": datetime.now().isoformat(),
        }
        
        if not self.is_available():
            result["error"] = "Reddit API not configured"
            result["setup_instructions"] = {
                "step1": "Go to https://www.reddit.com/prefs/apps",
                "step2": "Click 'Create App' or 'Create Another App'",
                "step3": "Select 'script' as the app type",
                "step4": "Add any name and redirect URI (http://localhost)",
                "step5": "Copy the client_id (under app name) and secret",
                "step6": "Add to .env: REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET",
            }
            return result
        
        # Get ticker mentions from trading subreddits
        result["trading_sentiment"] = self.get_ticker_mentions(
            ticker, 
            subreddits=self.TRADING_SUBREDDITS,
            time_filter="week",
            limit=50
        )
        
        # Get ticker mentions from investing subreddits
        result["investing_sentiment"] = self.get_ticker_mentions(
            ticker,
            subreddits=self.INVESTING_SUBREDDITS,
            time_filter="month",
            limit=30
        )
        
        # Get DD posts
        result["dd_posts"] = self.get_dd_posts(ticker, limit=5)
        
        # Get WSB trending (for context)
        result["wsb_trending"] = self.get_wsb_daily_sentiment()
        
        # Calculate combined sentiment
        trading_sent = result["trading_sentiment"]
        investing_sent = result["investing_sentiment"]
        
        if "error" not in trading_sent and "error" not in investing_sent:
            total_bullish = trading_sent.get("bullish_mentions", 0) + investing_sent.get("bullish_mentions", 0)
            total_bearish = trading_sent.get("bearish_mentions", 0) + investing_sent.get("bearish_mentions", 0)
            total_mentions = trading_sent.get("total_mentions", 0) + investing_sent.get("total_mentions", 0)
            
            if total_mentions > 0:
                combined_score = (total_bullish - total_bearish) / total_mentions
                
                if combined_score > 0.3:
                    combined_sentiment = "bullish"
                elif combined_score < -0.3:
                    combined_sentiment = "bearish"
                else:
                    combined_sentiment = "mixed"
            else:
                combined_sentiment = "no_data"
                combined_score = 0
            
            result["combined_analysis"] = {
                "total_mentions": total_mentions,
                "total_bullish": total_bullish,
                "total_bearish": total_bearish,
                "sentiment_score": round(combined_score, 3),
                "overall_sentiment": combined_sentiment,
                "retail_interest": "high" if total_mentions > 20 else "moderate" if total_mentions > 5 else "low",
            }
        
        return result

