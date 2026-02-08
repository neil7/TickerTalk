"""
Risk Management Agent
Assesses risk and provides position sizing recommendations
"""
import numpy as np
from typing import Dict, Any
from .base_agent import BaseAgent
from config import analysis_config


class RiskManagementAgent(BaseAgent):
    """Agent specialized in risk assessment and management"""
    
    def __init__(self):
        super().__init__("Risk Management Agent")
    
    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment
        
        Analyzes:
        - Volatility metrics
        - Drawdown risk
        - Position sizing recommendations
        - Stop-loss levels
        """
        ticker = state.get("ticker", "UNKNOWN")
        market_data = state.get("market_data", {})
        technical_analysis = state.get("technical_analysis", {})
        
        prices = market_data.get("historical_prices", [])
        
        if len(prices) < 20:
            state["risk_assessment"] = {
                "error": "Insufficient data for risk analysis",
                "data_points": len(prices),
            }
            state["messages"].append("Risk assessment: Insufficient data")
            return state
        
        prices_array = np.array(prices, dtype=float)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(prices_array)
        
        # Calculate position sizing
        position_sizing = self._calculate_position_sizing(
            risk_metrics,
            technical_analysis,
            prices_array[-1]
        )
        
        # Calculate stop-loss levels
        stop_loss = self._calculate_stop_loss(prices_array, risk_metrics)
        
        # Get risk level assessment
        risk_level = self._assess_risk_level(risk_metrics, technical_analysis)
        
        # Get LLM risk interpretation
        interpretation = self._get_llm_interpretation(
            ticker,
            risk_metrics,
            risk_level,
            position_sizing,
            stop_loss,
            technical_analysis
        )
        
        state["risk_assessment"] = {
            "risk_metrics": risk_metrics,
            "risk_level": risk_level,
            "position_sizing": position_sizing,
            "stop_loss": stop_loss,
            "interpretation": interpretation,
        }
        state["messages"].append("Risk assessment completed")
        
        return state
    
    def _calculate_risk_metrics(self, prices: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        metrics = {}
        
        # Daily returns
        returns = np.diff(prices) / prices[:-1]
        
        # Volatility (annualized)
        daily_vol = np.std(returns)
        metrics["daily_volatility"] = round(daily_vol * 100, 2)
        metrics["annualized_volatility"] = round(daily_vol * np.sqrt(252) * 100, 2)
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics["max_drawdown"] = round(np.min(drawdown) * 100, 2)
        metrics["current_drawdown"] = round(drawdown[-1] * 100, 2)
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)
        metrics["var_95_daily"] = round(var_95 * 100, 2)
        metrics["var_95_weekly"] = round(var_95 * np.sqrt(5) * 100, 2)
        
        # Sharpe-like metric (simplified - assuming 0 risk-free rate)
        avg_return = np.mean(returns)
        if daily_vol > 0:
            sharpe = (avg_return / daily_vol) * np.sqrt(252)
            metrics["sharpe_ratio"] = round(sharpe, 2)
        
        # Downside deviation (Sortino-like)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_dev = np.std(negative_returns) * np.sqrt(252)
            metrics["downside_deviation"] = round(downside_dev * 100, 2)
        
        # Win rate (percentage of positive days)
        metrics["win_rate"] = round(np.sum(returns > 0) / len(returns) * 100, 1)
        
        # Average gain vs average loss
        positive_returns = returns[returns > 0]
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            avg_gain = np.mean(positive_returns)
            avg_loss = np.abs(np.mean(negative_returns))
            metrics["gain_loss_ratio"] = round(avg_gain / avg_loss, 2)
        
        # Recent volatility vs historical
        recent_vol = np.std(returns[-20:])
        historical_vol = np.std(returns[:-20]) if len(returns) > 40 else daily_vol
        
        if historical_vol > 0:
            vol_ratio = recent_vol / historical_vol
            metrics["volatility_regime"] = "elevated" if vol_ratio > 1.3 else "normal" if vol_ratio > 0.7 else "subdued"
            metrics["vol_ratio"] = round(vol_ratio, 2)
        
        # Beta estimate (vs simple market proxy - using price momentum as rough proxy)
        if len(returns) > 20:
            market_proxy = np.mean(returns[-20:])
            stock_move = returns[-20:]
            correlation = np.corrcoef(stock_move, np.full(20, market_proxy))[0, 1] if market_proxy != 0 else 0
            metrics["estimated_beta"] = round(correlation * (daily_vol / 0.01), 2)  # Rough estimate
        
        return metrics
    
    def _calculate_position_sizing(self, risk_metrics: Dict, 
                                   technical_analysis: Dict,
                                   current_price: float) -> Dict[str, Any]:
        """Calculate recommended position size based on risk"""
        sizing = {}
        
        # Base position size
        max_position = analysis_config.max_position_size * 100  # Convert to percentage
        
        # Adjust based on volatility
        vol = risk_metrics.get("annualized_volatility", 20)
        vol_adjustment = 1.0
        
        if vol > 40:  # Very high volatility
            vol_adjustment = 0.5
        elif vol > 30:  # High volatility
            vol_adjustment = 0.7
        elif vol > 20:  # Moderate volatility
            vol_adjustment = 0.85
        elif vol < 15:  # Low volatility
            vol_adjustment = 1.1
        
        sizing["volatility_adjustment"] = vol_adjustment
        
        # Adjust based on technical signals
        tech_adjustment = 1.0
        signals = technical_analysis.get("signals", {})
        
        overall_signal = signals.get("overall", "neutral")
        if overall_signal == "bullish":
            tech_adjustment = 1.1
        elif overall_signal == "bearish":
            tech_adjustment = 0.7
        
        # RSI extremes reduce position size
        rsi_signal = signals.get("rsi", "neutral")
        if rsi_signal in ["overbought", "oversold"]:
            tech_adjustment *= 0.8
        
        sizing["technical_adjustment"] = tech_adjustment
        
        # Calculate final position size
        adjusted_position = max_position * vol_adjustment * tech_adjustment
        sizing["recommended_pct"] = round(min(adjusted_position, max_position), 1)
        
        # Calculate in terms of shares (assuming $10,000 portfolio)
        portfolio_value = 10000
        position_value = portfolio_value * (sizing["recommended_pct"] / 100)
        sizing["shares_for_10k_portfolio"] = int(position_value / current_price) if current_price > 0 else 0
        sizing["position_value"] = round(position_value, 2)
        
        return sizing
    
    def _calculate_stop_loss(self, prices: np.ndarray, risk_metrics: Dict) -> Dict[str, Any]:
        """Calculate stop-loss levels"""
        current_price = prices[-1]
        
        stop_loss = {}
        
        # ATR-based stop loss (2x ATR)
        atr = np.mean(np.abs(np.diff(prices[-14:]))) if len(prices) >= 14 else np.std(prices[-5:])
        stop_loss["atr_based"] = round(current_price - (2 * atr), 2)
        stop_loss["atr_based_pct"] = round((2 * atr / current_price) * 100, 1)
        
        # Volatility-based stop loss
        daily_vol = risk_metrics.get("daily_volatility", 1.5) / 100
        vol_stop = current_price * (1 - 2 * daily_vol)
        stop_loss["volatility_based"] = round(vol_stop, 2)
        stop_loss["volatility_based_pct"] = round((1 - vol_stop/current_price) * 100, 1)
        
        # Support-based stop loss (recent low)
        recent_low = np.min(prices[-20:])
        stop_loss["support_based"] = round(recent_low * 0.98, 2)  # 2% below recent low
        stop_loss["support_based_pct"] = round((1 - stop_loss["support_based"]/current_price) * 100, 1)
        
        # Fixed percentage stops
        stop_loss["fixed_5pct"] = round(current_price * 0.95, 2)
        stop_loss["fixed_10pct"] = round(current_price * 0.90, 2)
        
        # Recommended stop (choose most conservative)
        recommended = max(stop_loss["atr_based"], stop_loss["volatility_based"], stop_loss["support_based"])
        stop_loss["recommended"] = round(recommended, 2)
        stop_loss["recommended_pct"] = round((1 - recommended/current_price) * 100, 1)
        
        # Take profit levels
        stop_loss["take_profit_1"] = round(current_price * 1.10, 2)  # 10% profit
        stop_loss["take_profit_2"] = round(current_price * 1.20, 2)  # 20% profit
        stop_loss["take_profit_3"] = round(current_price * 1.30, 2)  # 30% profit
        
        # Risk/Reward ratio (using recommended stop and first take profit)
        risk = current_price - stop_loss["recommended"]
        reward = stop_loss["take_profit_1"] - current_price
        stop_loss["risk_reward_ratio"] = round(reward / risk, 2) if risk > 0 else 0
        
        return stop_loss
    
    def _assess_risk_level(self, risk_metrics: Dict, technical_analysis: Dict) -> Dict[str, Any]:
        """Assess overall risk level"""
        risk_score = 0
        risk_factors = []
        
        # Volatility risk
        vol = risk_metrics.get("annualized_volatility", 20)
        if vol > 40:
            risk_score += 3
            risk_factors.append("Very high volatility")
        elif vol > 30:
            risk_score += 2
            risk_factors.append("High volatility")
        elif vol > 20:
            risk_score += 1
        
        # Drawdown risk
        max_dd = abs(risk_metrics.get("max_drawdown", 0))
        if max_dd > 20:
            risk_score += 2
            risk_factors.append("Significant historical drawdown")
        elif max_dd > 10:
            risk_score += 1
        
        # VaR risk
        var = abs(risk_metrics.get("var_95_daily", 0))
        if var > 5:
            risk_score += 2
            risk_factors.append("High daily VaR")
        elif var > 3:
            risk_score += 1
        
        # Technical risk
        signals = technical_analysis.get("signals", {})
        if signals.get("rsi") == "overbought":
            risk_score += 1
            risk_factors.append("RSI indicates overbought")
        if signals.get("sr_position") == "near_resistance":
            risk_score += 1
            risk_factors.append("Price near resistance")
        
        # Volatility regime
        if risk_metrics.get("volatility_regime") == "elevated":
            risk_score += 1
            risk_factors.append("Elevated volatility regime")
        
        # Determine risk level
        if risk_score >= 6:
            risk_level = "very_high"
        elif risk_score >= 4:
            risk_level = "high"
        elif risk_score >= 2:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        return {
            "level": risk_level,
            "score": risk_score,
            "max_score": 10,
            "factors": risk_factors,
        }
    
    def _get_llm_interpretation(self, ticker: str, risk_metrics: Dict,
                                risk_level: Dict, position_sizing: Dict,
                                stop_loss: Dict, technical_analysis: Dict) -> str:
        """Get LLM interpretation of risk assessment"""
        
        # Format risk factors
        risk_factors_text = "\n".join([f"- {f}" for f in risk_level.get("factors", [])]) or "No major risk factors"
        
        prompt = self._create_prompt("""Provide risk management recommendations for {ticker}.

RISK METRICS:
- Annualized Volatility: {volatility}%
- Maximum Drawdown: {max_dd}%
- Daily VaR (95%): {var}%
- Volatility Regime: {vol_regime}
- Risk Level: {risk_level} (Score: {risk_score}/10)

RISK FACTORS IDENTIFIED:
{risk_factors}

POSITION SIZING:
- Recommended Position: {position_pct}% of portfolio
- Volatility Adjustment: {vol_adj}

STOP LOSS LEVELS:
- Recommended Stop: ${stop_price} ({stop_pct}% below current)
- Risk/Reward Ratio: {rr_ratio}

Based on this analysis, provide:
1. Risk verdict (Low/Medium/High/Very High risk)
2. Specific position sizing advice
3. Stop-loss and take-profit strategy
4. Key risk warnings

Keep response concise and actionable (under 150 words).""")
        
        try:
            interpretation = self._invoke_llm(
                prompt,
                ticker=ticker,
                volatility=risk_metrics.get("annualized_volatility", "N/A"),
                max_dd=abs(risk_metrics.get("max_drawdown", 0)),
                var=abs(risk_metrics.get("var_95_daily", 0)),
                vol_regime=risk_metrics.get("volatility_regime", "N/A"),
                risk_level=risk_level.get("level", "N/A"),
                risk_score=risk_level.get("score", "N/A"),
                risk_factors=risk_factors_text,
                position_pct=position_sizing.get("recommended_pct", "N/A"),
                vol_adj=position_sizing.get("volatility_adjustment", "N/A"),
                stop_price=stop_loss.get("recommended", "N/A"),
                stop_pct=stop_loss.get("recommended_pct", "N/A"),
                rr_ratio=stop_loss.get("risk_reward_ratio", "N/A"),
            )
            return interpretation
        except Exception as e:
            return f"Risk interpretation unavailable: {str(e)}"

