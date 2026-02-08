"""
Reddit Sentiment Agent
Analyzes retail investor sentiment from Reddit communities
"""
from typing import Dict, Any, List
from .base_agent import BaseAgent
from data_sources.reddit_data import RedditDataFetcher


class RedditSentimentAgent(BaseAgent):
    """
    Agent specialized in Reddit sentiment analysis
    
    Monitors:
    - r/wallstreetbets - Retail momentum, meme stocks
    - r/stocks - General stock discussion
    - r/investing - Long-term sentiment
    - r/options - Options activity sentiment
    """
    
    def __init__(self):
        super().__init__("Reddit Sentiment Agent")
        self.reddit_fetcher = RedditDataFetcher()
    
    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform Reddit sentiment analysis
        
        Analyzes:
        - Mention frequency and trends
        - Bullish vs bearish sentiment
        - WSB activity and momentum
        - Due diligence post quality
        """
        ticker = state.get("ticker", "UNKNOWN")
        
        # Check if Reddit is available
        if not self.reddit_fetcher.is_available():
            state["reddit_analysis"] = {
                "available": False,
                "error": "Reddit API not configured",
                "recommendation": "Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env file",
            }
            state["messages"].append("Reddit analysis: API not configured")
            return state
        
        # Fetch comprehensive Reddit data
        reddit_data = self.reddit_fetcher.get_comprehensive_reddit_analysis(ticker)
        
        # Process and analyze the data
        analysis = self._analyze_reddit_data(ticker, reddit_data)
        
        # Get LLM interpretation
        interpretation = self._get_llm_interpretation(ticker, reddit_data, analysis)
        
        state["reddit_analysis"] = {
            "available": True,
            "raw_data": reddit_data,
            "analysis": analysis,
            "interpretation": interpretation,
        }
        state["messages"].append("Reddit sentiment analysis completed")
        
        return state
    
    def _analyze_reddit_data(self, ticker: str, reddit_data: Dict) -> Dict[str, Any]:
        """Analyze Reddit data and extract insights"""
        analysis = {
            "retail_sentiment": "unknown",
            "momentum_indicator": "neutral",
            "risk_signals": [],
            "opportunity_signals": [],
        }
        
        combined = reddit_data.get("combined_analysis", {})
        trading = reddit_data.get("trading_sentiment", {})
        wsb = reddit_data.get("wsb_trending", {})
        
        # Overall retail sentiment
        overall_sent = combined.get("overall_sentiment", "unknown")
        analysis["retail_sentiment"] = overall_sent
        analysis["sentiment_score"] = combined.get("sentiment_score", 0)
        analysis["total_mentions"] = combined.get("total_mentions", 0)
        analysis["retail_interest"] = combined.get("retail_interest", "low")
        
        # Momentum analysis
        if trading and "error" not in trading:
            top_posts = trading.get("top_posts", [])
            if top_posts:
                avg_score = sum(p["score"] for p in top_posts) / len(top_posts)
                avg_comments = sum(p["num_comments"] for p in top_posts) / len(top_posts)
                
                analysis["avg_post_score"] = round(avg_score, 1)
                analysis["avg_comments"] = round(avg_comments, 1)
                
                if avg_score > 500 and avg_comments > 100:
                    analysis["momentum_indicator"] = "high_momentum"
                elif avg_score > 100:
                    analysis["momentum_indicator"] = "building_interest"
                else:
                    analysis["momentum_indicator"] = "low_activity"
        
        # WSB specific analysis (meme stock potential)
        if wsb and "error" not in wsb:
            trending = wsb.get("trending_tickers", [])
            for stock in trending:
                if stock["ticker"] == ticker.upper():
                    analysis["wsb_trending"] = True
                    analysis["wsb_rank"] = trending.index(stock) + 1
                    analysis["wsb_mentions"] = stock["mentions"]
                    
                    # WSB momentum can be both opportunity and risk
                    if stock["mentions"] > 10:
                        analysis["opportunity_signals"].append("High WSB interest - potential momentum play")
                        analysis["risk_signals"].append("WSB attention - high volatility expected")
                    break
            else:
                analysis["wsb_trending"] = False
        
        # Risk signals from sentiment
        if overall_sent == "bullish" and analysis.get("total_mentions", 0) > 30:
            analysis["risk_signals"].append("Very high bullish sentiment - possible crowded trade")
        
        if analysis.get("momentum_indicator") == "high_momentum":
            analysis["risk_signals"].append("High retail momentum - watch for reversal")
        
        # Opportunity signals
        if overall_sent == "bullish" and analysis.get("retail_interest") == "moderate":
            analysis["opportunity_signals"].append("Building bullish sentiment - early momentum")
        
        if overall_sent == "bearish" and analysis.get("total_mentions", 0) < 10:
            analysis["opportunity_signals"].append("Low attention + bearish - potential contrarian play")
        
        return analysis
    
    def _get_llm_interpretation(self, ticker: str, reddit_data: Dict, 
                                analysis: Dict) -> str:
        """Get LLM interpretation of Reddit sentiment"""
        
        # Format top posts for context
        trading = reddit_data.get("trading_sentiment", {})
        top_posts = trading.get("top_posts", [])[:5] if "error" not in trading else []
        
        posts_text = "\n".join([
            f"- [{p['subreddit']}] {p['title'][:80]}... (Score: {p['score']}, Sentiment: {p['sentiment']})"
            for p in top_posts
        ]) if top_posts else "No recent posts found"
        
        # Format DD posts
        dd_posts = reddit_data.get("dd_posts", [])[:3]
        dd_text = "\n".join([
            f"- {p['title'][:80]}... (Score: {p['score']})"
            for p in dd_posts
        ]) if dd_posts else "No DD posts found"
        
        # Check WSB trending
        wsb = reddit_data.get("wsb_trending", {})
        wsb_trending = wsb.get("trending_tickers", [])[:5] if "error" not in wsb else []
        wsb_text = ", ".join([f"{t['ticker']} ({t['mentions']} mentions)" for t in wsb_trending]) if wsb_trending else "N/A"
        
        prompt = self._create_prompt("""Analyze Reddit sentiment for {ticker} and its implications for trading.

REDDIT SENTIMENT DATA:
- Total Mentions: {total_mentions}
- Sentiment: {sentiment} (Score: {sentiment_score})
- Retail Interest Level: {retail_interest}
- Momentum: {momentum}

TOP RECENT POSTS:
{posts}

DUE DILIGENCE POSTS:
{dd_posts}

WSB TRENDING TICKERS:
{wsb_trending}

RISK SIGNALS: {risk_signals}
OPPORTUNITY SIGNALS: {opportunity_signals}

Provide a brief analysis (under 150 words) covering:
1. Reddit sentiment summary (bullish/bearish/neutral)
2. Is there unusual retail activity or momentum?
3. Any meme stock characteristics?
4. How should traders factor Reddit sentiment into their decision?
5. Key risks from retail positioning""")
        
        try:
            interpretation = self._invoke_llm(
                prompt,
                ticker=ticker,
                total_mentions=analysis.get("total_mentions", 0),
                sentiment=analysis.get("retail_sentiment", "unknown"),
                sentiment_score=analysis.get("sentiment_score", 0),
                retail_interest=analysis.get("retail_interest", "low"),
                momentum=analysis.get("momentum_indicator", "neutral"),
                posts=posts_text,
                dd_posts=dd_text,
                wsb_trending=wsb_text,
                risk_signals=", ".join(analysis.get("risk_signals", [])) or "None",
                opportunity_signals=", ".join(analysis.get("opportunity_signals", [])) or "None",
            )
            return interpretation
        except Exception as e:
            return f"Reddit interpretation unavailable: {str(e)}"
    
    def get_wsb_momentum_scan(self) -> Dict[str, Any]:
        """
        Scan WSB for momentum plays
        Returns trending tickers with sentiment
        """
        if not self.reddit_fetcher.is_available():
            return {"error": "Reddit API not configured"}
        
        wsb_data = self.reddit_fetcher.get_wsb_daily_sentiment()
        
        if "error" in wsb_data:
            return wsb_data
        
        trending = wsb_data.get("trending_tickers", [])
        
        # Analyze each trending ticker
        momentum_plays = []
        for stock in trending[:10]:
            if stock["mentions"] >= 3:  # Minimum threshold
                momentum_plays.append({
                    "ticker": stock["ticker"],
                    "mentions": stock["mentions"],
                    "sentiment": stock["sentiment"],
                    "momentum_score": stock["total_score"] / max(stock["mentions"], 1),
                    "is_bullish": stock["sentiment"] == "bullish",
                })
        
        return {
            "trending_momentum_plays": momentum_plays,
            "posts_analyzed": wsb_data.get("posts_analyzed", 0),
            "fetch_time": wsb_data.get("fetch_time"),
        }

