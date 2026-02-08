"""
Sentiment Analysis Agent
Analyzes sentiment from news, social media, and political events
"""
from typing import Dict, Any, List
from .base_agent import BaseAgent


class SentimentAnalysisAgent(BaseAgent):
    """Agent specialized in sentiment analysis from multiple sources"""
    
    def __init__(self):
        super().__init__("Sentiment Analysis Agent")
    
    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis
        
        Analyzes:
        - Stock-specific news sentiment
        - White House/Political news impact
        - Social media sentiment (Reddit)
        - Google search trends
        """
        ticker = state.get("ticker", "UNKNOWN")
        sentiment_data = state.get("sentiment_data", {})
        
        if not sentiment_data:
            state["sentiment_analysis"] = {"error": "No sentiment data available"}
            state["messages"].append("Sentiment analysis: No data")
            return state
        
        # Process stock-specific news
        stock_news_summary = self._summarize_stock_news(
            sentiment_data.get("stock_news", [])
        )
        
        # Process political/White House news
        political_summary = self._summarize_political_news(
            sentiment_data.get("white_house_news", [])
        )
        
        # Get LLM sentiment analysis
        sentiment_analysis = self._get_llm_sentiment_analysis(
            ticker,
            stock_news_summary,
            political_summary
        )
        
        # Determine overall sentiment scores
        sentiment_scores = self._calculate_sentiment_scores(sentiment_data)
        
        state["sentiment_analysis"] = {
            "stock_news_summary": stock_news_summary,
            "political_summary": political_summary,
            "llm_analysis": sentiment_analysis,
            "scores": sentiment_scores,
        }
        state["messages"].append("Sentiment analysis completed")
        
        return state
    
    def _summarize_stock_news(self, news_items: List[Dict]) -> Dict[str, Any]:
        """Summarize stock-specific news"""
        if not news_items:
            return {"count": 0, "headlines": [], "has_news": False}
        
        headlines = []
        positive_count = 0
        negative_count = 0
        
        positive_words = ["surge", "gain", "up", "rise", "beat", "exceed", "growth", 
                         "bullish", "upgrade", "strong", "profit", "success", "record"]
        negative_words = ["fall", "drop", "down", "decline", "miss", "cut", "weak",
                         "bearish", "downgrade", "loss", "concern", "risk", "warning"]
        
        for item in news_items[:10]:
            title = item.get('title', '')
            if title:
                headlines.append(title)
                title_lower = title.lower()
                
                if any(word in title_lower for word in positive_words):
                    positive_count += 1
                if any(word in title_lower for word in negative_words):
                    negative_count += 1
        
        # Determine news sentiment
        if positive_count > negative_count * 1.5:
            news_sentiment = "positive"
        elif negative_count > positive_count * 1.5:
            news_sentiment = "negative"
        else:
            news_sentiment = "mixed"
        
        return {
            "count": len(news_items),
            "headlines": headlines,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "sentiment": news_sentiment,
            "has_news": len(headlines) > 0,
        }
    
    def _summarize_political_news(self, news_items: List[Dict]) -> Dict[str, Any]:
        """Summarize political news with market impact assessment"""
        if not news_items:
            return {"count": 0, "headlines": [], "potential_impact": "none"}
        
        headlines = []
        market_impact_keywords = ["tariff", "tax", "trade", "regulation", "fed", 
                                  "interest rate", "stimulus", "spending", "deficit",
                                  "inflation", "economy", "jobs", "employment",
                                  "oil", "energy", "tech", "antitrust", "crypto"]
        
        impact_count = 0
        
        for item in news_items[:10]:
            title = item.get('title', '')
            if title:
                headlines.append(title)
                title_lower = title.lower()
                
                if any(keyword in title_lower for keyword in market_impact_keywords):
                    impact_count += 1
        
        # Assess potential market impact
        if impact_count >= 3:
            potential_impact = "high"
        elif impact_count >= 1:
            potential_impact = "moderate"
        else:
            potential_impact = "low"
        
        return {
            "count": len(news_items),
            "headlines": headlines,
            "market_relevant_count": impact_count,
            "potential_impact": potential_impact,
        }
    
    def _calculate_sentiment_scores(self, sentiment_data: Dict) -> Dict[str, Any]:
        """Calculate numerical sentiment scores from various sources"""
        scores = {}
        
        # Stock news score
        stock_news = sentiment_data.get("stock_news", [])
        if stock_news:
            # Check if any items have pre-calculated sentiment
            sentiments = [item.get("sentiment") for item in stock_news if item.get("sentiment")]
            if sentiments:
                scores["stock_news_api_sentiment"] = sentiments[:5]
        
        # Social media scores (if available)
        wsb = sentiment_data.get("wallstreetbets", {})
        if wsb and "error" not in wsb:
            bullish = wsb.get("bullish_mentions", 0)
            bearish = wsb.get("bearish_mentions", 0)
            total = bullish + bearish
            
            if total > 0:
                scores["wsb_sentiment"] = {
                    "bullish_pct": round(bullish / total * 100, 1),
                    "bearish_pct": round(bearish / total * 100, 1),
                    "overall": wsb.get("overall_sentiment", "unknown"),
                    "mentions": wsb.get("total_mentions", 0),
                }
        
        # Google Trends (if available)
        trends = sentiment_data.get("google_trends", {})
        if trends and "error" not in trends:
            scores["google_trends"] = {
                "interest": trends.get("current_interest", 0),
                "trend": trends.get("trend_direction", "unknown"),
            }
        
        return scores
    
    def _get_llm_sentiment_analysis(self, ticker: str, 
                                    stock_news: Dict, 
                                    political_news: Dict) -> str:
        """Get LLM interpretation of sentiment data"""
        
        # Format news headlines for prompt
        stock_headlines = "\n".join([f"- {h}" for h in stock_news.get("headlines", [])[:5]])
        political_headlines = "\n".join([f"- {h}" for h in political_news.get("headlines", [])[:3]])
        
        if not stock_headlines:
            stock_headlines = "No recent stock-specific news available"
        if not political_headlines:
            political_headlines = "No recent political news with market impact"
        
        prompt = self._create_prompt("""Analyze the sentiment landscape for {ticker} based on recent news and political developments.

STOCK-SPECIFIC NEWS ({stock_count} articles, {stock_sentiment} sentiment):
{stock_headlines}

POLITICAL/WHITE HOUSE NEWS (Market Impact: {political_impact}):
{political_headlines}

Based on this information, provide:
1. Overall market sentiment (Bullish/Bearish/Neutral) with confidence level
2. Key themes or catalysts driving sentiment
3. Potential short-term price impact
4. Any risks or concerns from the news

Keep your analysis concise and actionable (under 150 words).""")
        
        try:
            analysis = self._invoke_llm(
                prompt,
                ticker=ticker,
                stock_count=stock_news.get("count", 0),
                stock_sentiment=stock_news.get("sentiment", "unknown"),
                stock_headlines=stock_headlines,
                political_impact=political_news.get("potential_impact", "unknown"),
                political_headlines=political_headlines,
            )
            return analysis
        except Exception as e:
            return f"Sentiment analysis unavailable: {str(e)}"
    
    def analyze_specific_event(self, event_description: str, ticker: str) -> str:
        """
        Analyze the potential impact of a specific event on a stock
        Useful for ad-hoc analysis of breaking news
        """
        prompt = self._create_prompt("""Analyze how the following event might impact {ticker} stock:

EVENT:
{event_description}

Provide:
1. Expected market reaction (positive/negative/neutral)
2. Magnitude of expected impact (low/moderate/high)
3. Time horizon of impact (immediate/short-term/long-term)
4. Trading recommendation based on this event

Keep response under 100 words.""")
        
        try:
            analysis = self._invoke_llm(
                prompt,
                ticker=ticker,
                event_description=event_description,
            )
            return analysis
        except Exception as e:
            return f"Event analysis unavailable: {str(e)}"

