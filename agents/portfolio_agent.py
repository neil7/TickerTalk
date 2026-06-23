"""
Portfolio Manager Agent
Synthesizes all analyses and generates final trading recommendations
Includes price targets for short-term trading
"""
import numpy as np
from typing import Dict, Any, List
from .base_agent import BaseAgent


class PortfolioManagerAgent(BaseAgent):
    """Agent that synthesizes all analyses and makes final recommendations"""
    
    def __init__(self):
        super().__init__("Portfolio Manager Agent")
    
    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize all agent analyses and generate final recommendation
        
        Considers:
        - Technical analysis signals
        - Sentiment analysis
        - Fundamental analysis
        - Risk assessment
        """
        ticker = state.get("ticker", "UNKNOWN")
        
        # Gather all analyses
        technical = state.get("technical_analysis", {})
        sentiment = state.get("sentiment_analysis", {})
        fundamental = state.get("fundamental_analysis", {})
        risk = state.get("risk_assessment", {})
        market_data = state.get("market_data", {})
        
        # Generate scores from each analysis
        scores = self._calculate_composite_score(
            technical, sentiment, fundamental, risk
        )
        
        # Determine trading action
        action_data = self._determine_action(scores, risk)
        action_type = action_data.get("recommendation", "HOLD")
        
        # Calculate price targets for short-term trading
        price_targets = self._calculate_price_targets(market_data, technical, risk, action_type)
        
        # Generate final recommendation with LLM
        recommendation = self._generate_recommendation(
            ticker,
            technical,
            sentiment,
            fundamental,
            risk,
            market_data,
            scores,
            action_data,
            price_targets
        )
        
        state["final_recommendation"] = {
            "ticker": ticker,
            "action": action_data,
            "scores": scores,
            "price_targets": price_targets,
            "recommendation": recommendation,
        }
        state["messages"].append("Final recommendation generated")
        
        return state
    
    def _calculate_composite_score(self, technical: Dict, sentiment: Dict,
                                   fundamental: Dict, risk: Dict) -> Dict[str, Any]:
        """Calculate composite scores from all analyses"""
        scores = {
            "technical_score": 0,
            "sentiment_score": 0,
            "fundamental_score": 0,
            "risk_score": 0,
            "composite_score": 0,
        }
        
        # Technical score (-2 to +2)
        signals = technical.get("signals", {})
        tech_score = 0
        
        overall = signals.get("overall", "neutral")
        if overall == "bullish":
            tech_score += 1
        elif overall == "bearish":
            tech_score -= 1
        
        trend = signals.get("trend", "neutral")
        if trend == "strong_bullish":
            tech_score += 1
        elif trend == "strong_bearish":
            tech_score -= 1
        
        rsi = signals.get("rsi", "neutral")
        if rsi == "oversold":
            tech_score += 0.5
        elif rsi == "overbought":
            tech_score -= 0.5
        
        scores["technical_score"] = round(tech_score, 1)
        
        # Sentiment score (-2 to +2)
        sent_score = 0
        
        llm_analysis = sentiment.get("llm_analysis", "").lower()
        if "bullish" in llm_analysis:
            sent_score += 1
        elif "bearish" in llm_analysis:
            sent_score -= 1
        
        news_summary = sentiment.get("stock_news_summary", {})
        news_sentiment = news_summary.get("sentiment", "mixed")
        if news_sentiment == "positive":
            sent_score += 0.5
        elif news_sentiment == "negative":
            sent_score -= 0.5
        
        scores["sentiment_score"] = round(sent_score, 1)
        
        # Fundamental score (-2 to +2)
        fund_score = 0
        
        valuation = fundamental.get("valuation_assessment", {})
        overall_val = valuation.get("overall_valuation", "fair")
        if overall_val == "attractive":
            fund_score += 1
        elif overall_val == "expensive":
            fund_score -= 1
        
        position_52w = valuation.get("52w_position", "mid_range")
        if position_52w in ["near_low", "lower_range"]:
            fund_score += 0.5
        elif position_52w == "near_high":
            fund_score -= 0.5
        
        fund_interpretation = fundamental.get("interpretation", "").lower()
        if "bullish" in fund_interpretation or "undervalued" in fund_interpretation:
            fund_score += 0.5
        elif "bearish" in fund_interpretation or "overvalued" in fund_interpretation:
            fund_score -= 0.5
        
        scores["fundamental_score"] = round(fund_score, 1)
        
        # Risk adjustment score (-2 to 0, higher is better/less risky)
        risk_level = risk.get("risk_level", {}).get("level", "moderate")
        if risk_level == "low":
            risk_adjustment = 0
        elif risk_level == "moderate":
            risk_adjustment = -0.5
        elif risk_level == "high":
            risk_adjustment = -1
        else:  # very_high
            risk_adjustment = -1.5
        
        scores["risk_adjustment"] = risk_adjustment
        
        # Composite score (weighted average)
        weights = {
            "technical": 0.30,
            "sentiment": 0.20,
            "fundamental": 0.25,
            "risk": 0.25,
        }
        
        composite = (
            scores["technical_score"] * weights["technical"] +
            scores["sentiment_score"] * weights["sentiment"] +
            scores["fundamental_score"] * weights["fundamental"] +
            risk_adjustment * weights["risk"]
        )
        
        scores["composite_score"] = round(composite, 2)
        
        # Conviction level (1-10)
        # Based on alignment of different scores
        score_list = [scores["technical_score"], scores["sentiment_score"], scores["fundamental_score"]]
        all_positive = all(s >= 0.5 for s in score_list)
        all_negative = all(s <= -0.5 for s in score_list)
        
        if all_positive or all_negative:
            conviction = min(10, 7 + abs(composite))
        else:
            conviction = max(1, 5 + composite)
        
        scores["conviction"] = round(conviction, 1)
        
        return scores
    
    def _determine_action(self, scores: Dict, risk: Dict) -> Dict[str, Any]:
        """Determine trading action based on scores"""
        composite = scores.get("composite_score", 0)
        conviction = scores.get("conviction", 5)
        risk_level = risk.get("risk_level", {}).get("level", "moderate")
        
        # Determine action
        if composite >= 1:
            action = "STRONG_BUY"
        elif composite >= 0.3:
            action = "BUY"
        elif composite <= -1:
            action = "STRONG_SELL"
        elif composite <= -0.3:
            action = "SELL"
        else:
            action = "HOLD"
        
        # Adjust for high risk
        if risk_level in ["high", "very_high"]:
            if action == "STRONG_BUY":
                action = "BUY"
            elif action == "BUY":
                action = "HOLD"
            conviction = max(1, conviction - 2)
        
        # Time horizon
        if abs(composite) > 1:
            time_horizon = "short_term"  # 1-4 weeks
        elif abs(composite) > 0.5:
            time_horizon = "medium_term"  # 1-3 months
        else:
            time_horizon = "swing_trade"  # Days to weeks
        
        return {
            "recommendation": action,
            "conviction": round(conviction, 1),
            "time_horizon": time_horizon,
        }
    
    def _calculate_price_targets(self, market_data: Dict, technical: Dict, 
                                 risk: Dict, action_type: str = "HOLD") -> Dict[str, Any]:
        """
        Calculate buy/sell price targets for short-term trading
        Handles both LONG (Buy Low -> Sell High) and SHORT (Sell High -> Buy Low)
        """
        current_price = market_data.get("current_price", 0)
        if not current_price:
            return {"error": "No price data"}
        
        prices = market_data.get("historical_prices", [])
        indicators = technical.get("indicators", {})
        
        # Get ATR from technical analysis or calculate
        atr = indicators.get("atr_14", 0)
        if not atr and len(prices) >= 14:
            prices_arr = np.array(prices[-14:])
            atr = np.mean(np.abs(np.diff(prices_arr)))
        elif not atr:
            atr = current_price * 0.02  # Default 2% ATR
        
        # Get support/resistance
        support = indicators.get("support_level", current_price * 0.95)
        resistance = indicators.get("resistance_level", current_price * 1.05)
        
        # Determine trade direction
        trade_type = "LONG"  # always long-only: buy low, sell high
        
        if trade_type == "LONG":
            # LONG STRATEGY: Buy Low -> Sell High
            entry_ideal = max(support, current_price - atr * 0.5)
            entry_aggressive = current_price
            
            stop_loss = max(support * 0.98, current_price - atr * 2)
            
            target_1 = current_price + atr * 1.5
            target_2 = current_price + atr * 2.5
            target_3 = current_price + atr * 4
            
            # Cap at resistance
            if resistance > current_price:
                target_3 = min(resistance * 1.02, target_3)
                
            # Profit/Loss percentages
            target_1_pct = ((target_1 - current_price) / current_price) * 100
            target_2_pct = ((target_2 - current_price) / current_price) * 100
            target_3_pct = ((target_3 - current_price) / current_price) * 100
            stop_loss_pct = ((stop_loss - current_price) / current_price) * 100
            
        else:
            # SHORT STRATEGY: Sell High -> Buy Low
            # Entry is at current price or on a bounce to resistance
            entry_ideal = min(resistance, current_price + atr * 0.5)
            entry_aggressive = current_price
            
            # Stop loss is ABOVE entry
            stop_loss = min(resistance * 1.02, current_price + atr * 2)
            
            # Targets are BELOW current price
            target_1 = current_price - atr * 1.5
            target_2 = current_price - atr * 2.5
            target_3 = current_price - atr * 4
            
            # Floor at support
            if support < current_price:
                target_3 = max(support * 0.98, target_3)
                
            # Profit/Loss percentages (Positive for short if price drops)
            # Profit = (Entry - Target) / Entry
            target_1_pct = ((current_price - target_1) / current_price) * 100
            target_2_pct = ((current_price - target_2) / current_price) * 100
            target_3_pct = ((current_price - target_3) / current_price) * 100
            # Loss is negative if stop is hit (Stop > Entry)
            stop_loss_pct = -((stop_loss - current_price) / current_price) * 100

        # Calculate Fibonacci levels (for reference)
        if len(prices) >= 20:
            recent_high = max(prices[-20:])
            recent_low = min(prices[-20:])
            fib_range = recent_high - recent_low
            
            fib_236 = recent_high - (fib_range * 0.236)
            fib_382 = recent_high - (fib_range * 0.382)
            fib_500 = recent_high - (fib_range * 0.500)
            fib_618 = recent_high - (fib_range * 0.618)
        else:
            fib_236 = current_price * 1.02
            fib_382 = current_price * 0.98
            fib_500 = current_price * 0.97
            fib_618 = current_price * 0.95
        
        # Calculate risk/reward
        risk_amount = abs(entry_aggressive - stop_loss)
        reward_1 = abs(target_1 - entry_aggressive)
        rr_ratio = reward_1 / risk_amount if risk_amount > 0 else 0
        
        # Position sizing based on 1-2% portfolio risk
        risk_pct_val = (risk_amount / current_price) * 100
        suggested_position = min(10, 2 / risk_pct_val * 100) if risk_pct_val > 0 else 5
        
        return {
            "trade_type": trade_type,
            "current_price": round(current_price, 2),
            
            # Entry points
            "entry_ideal": round(entry_ideal, 2),
            "entry_aggressive": round(entry_aggressive, 2),
            "entry_zone": f"${min(entry_ideal, entry_aggressive):.2f} - ${max(entry_ideal, entry_aggressive):.2f}",
            
            # Stop loss
            "stop_loss": round(stop_loss, 2),
            "stop_loss_pct": round(stop_loss_pct, 1),
            
            # Profit targets
            "target_1": round(target_1, 2),
            "target_1_pct": round(target_1_pct, 1),
            "target_2": round(target_2, 2),
            "target_2_pct": round(target_2_pct, 1),
            "target_3": round(target_3, 2),
            "target_3_pct": round(target_3_pct, 1),
            
            # Risk metrics
            "risk_reward_ratio": round(rr_ratio, 2),
            "atr": round(atr, 2),
            "suggested_position_pct": round(suggested_position, 1),
            
            # Key levels
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            
            # Fibonacci levels
            "fib_levels": {
                "23.6%": round(fib_236, 2),
                "38.2%": round(fib_382, 2),
                "50.0%": round(fib_500, 2),
                "61.8%": round(fib_618, 2),
            }
        }

    
    def _generate_recommendation(self, ticker: str, technical: Dict,
                                 sentiment: Dict, fundamental: Dict,
                                 risk: Dict, market_data: Dict,
                                 scores: Dict, action: Dict,
                                 price_targets: Dict = None) -> str:
        """Generate comprehensive final recommendation using LLM"""
        
        # Extract key points from each analysis
        tech_interpretation = technical.get("interpretation", "Technical analysis not available")[:200]
        sent_interpretation = sentiment.get("llm_analysis", "Sentiment analysis not available")[:200]
        fund_interpretation = fundamental.get("interpretation", "Fundamental analysis not available")[:200]
        risk_interpretation = risk.get("interpretation", "Risk assessment not available")[:200]
        
        # Get stop loss and position sizing
        risk_assessment = risk.get("risk_assessment", risk)
        stop_loss = risk.get("stop_loss", {})
        position_sizing = risk.get("position_sizing", {})
        
        # Price targets for short-term trading
        if price_targets is None:
            price_targets = {}
        
        prompt = self._create_prompt("""As a portfolio manager specializing in SHORT-TERM TRADING, provide a final trading recommendation for {ticker}.

CURRENT PRICE: ${current_price}

=== TECHNICAL ANALYSIS (Score: {tech_score}) ===
{tech_interpretation}

=== SENTIMENT ANALYSIS (Score: {sent_score}) ===
{sent_interpretation}

=== FUNDAMENTAL ANALYSIS (Score: {fund_score}) ===
{fund_interpretation}

=== RISK ASSESSMENT (Level: {risk_level}) ===
{risk_interpretation}

=== COMPOSITE ANALYSIS ===
- Composite Score: {composite_score} (range: -2 to +2)
- Conviction Level: {conviction}/10
- Time Horizon: {time_horizon}

=== 📊 SHORT-TERM TRADING LEVELS ===
ENTRY ZONE: {entry_zone}
- Ideal Entry (on pullback): ${entry_ideal}
- Aggressive Entry (market): ${entry_aggressive}

PROFIT TARGETS:
- Target 1: ${target_1} (+{target_1_pct}%) - Take 50% profit
- Target 2: ${target_2} (+{target_2_pct}%) - Take 30% profit  
- Target 3: ${target_3} (+{target_3_pct}%) - Let rest ride

RISK/REWARD: {rr_ratio}:1
SUGGESTED POSITION: {position_pct}% of portfolio

KEY LEVELS:
- Support: ${support}
- Resistance: ${resistance}

Provide your FINAL TRADING RECOMMENDATION:

1. **ACTION**: {action} with conviction {conviction}/10

2. **EXACT ENTRY**: Best price to enter and conditions to wait for

3. **PROFIT TAKING PLAN**: When and at what price to take profits at each target

4. **TIME EXPECTATION**: How long to hold for targets

5. **RISK WARNING**: Top risks that could invalidate this trade

Be specific with prices. This is for SHORT-TERM trading (days to weeks).""")
        
        try:
            recommendation = self._invoke_llm(
                prompt,
                ticker=ticker,
                current_price=market_data.get("current_price", "N/A"),
                tech_score=scores.get("technical_score", 0),
                tech_interpretation=tech_interpretation,
                sent_score=scores.get("sentiment_score", 0),
                sent_interpretation=sent_interpretation,
                fund_score=scores.get("fundamental_score", 0),
                fund_interpretation=fund_interpretation,
                risk_level=risk.get("risk_level", {}).get("level", "N/A"),
                risk_interpretation=risk_interpretation,
                composite_score=scores.get("composite_score", 0),
                conviction=scores.get("conviction", 5),
                action=action.get("recommendation", "HOLD"),
                trade_type=price_targets.get("trade_type", "N/A"),
                time_horizon=action.get("time_horizon", "N/A"),
                entry_zone=price_targets.get("entry_zone", "N/A"),
                entry_ideal=price_targets.get("entry_ideal", "N/A"),
                entry_aggressive=price_targets.get("entry_aggressive", "N/A"),
                target_1=price_targets.get("target_1", "N/A"),
                target_1_pct=price_targets.get("target_1_pct", "N/A"),
                target_2=price_targets.get("target_2", "N/A"),
                target_2_pct=price_targets.get("target_2_pct", "N/A"),
                target_3=price_targets.get("target_3", "N/A"),
                target_3_pct=price_targets.get("target_3_pct", "N/A"),
                rr_ratio=price_targets.get("risk_reward_ratio", "N/A"),
                position_pct=price_targets.get("suggested_position_pct", "N/A"),
                support=price_targets.get("support", "N/A"),
                resistance=price_targets.get("resistance", "N/A"),
            )
            return recommendation
        except Exception as e:
            return f"Final recommendation generation failed: {str(e)}"
    
    def quick_assessment(self, ticker: str, market_data: Dict) -> str:
        """
        Generate a quick assessment without full agent pipeline
        Useful for rapid screening of multiple stocks
        """
        prompt = self._create_prompt("""Provide a quick 2-3 sentence assessment of {ticker} based on:

Current Price: ${price}
Change Today: {change}%
P/E Ratio: {pe}
52-Week Position: {position}
Volume vs Average: {volume_ratio}x

Is this stock worth deeper analysis? Why or why not?""")
        
        try:
            current = market_data.get("current_price", 0)
            high_52 = market_data.get("52_week_high", current)
            low_52 = market_data.get("52_week_low", current)
            
            position = "N/A"
            if high_52 and low_52 and high_52 != low_52:
                pos_pct = (current - low_52) / (high_52 - low_52) * 100
                position = f"{pos_pct:.0f}% of range"
            
            assessment = self._invoke_llm(
                prompt,
                ticker=ticker,
                price=current,
                change=market_data.get("change_percent", 0),
                pe=market_data.get("pe_ratio", "N/A"),
                position=position,
                volume_ratio=market_data.get("volume", 0) / market_data.get("avg_volume", 1) if market_data.get("avg_volume") else 1,
            )
            return assessment
        except Exception as e:
            return f"Quick assessment failed: {str(e)}"

