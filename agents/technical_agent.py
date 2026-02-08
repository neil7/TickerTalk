"""
Technical Analysis Agent
Performs comprehensive technical analysis including indicators and pattern recognition
"""
import numpy as np
from typing import Dict, Any, List
from .base_agent import BaseAgent
from config import analysis_config


class TechnicalAnalysisAgent(BaseAgent):
    """Agent specialized in technical analysis"""
    
    def __init__(self):
        super().__init__("Technical Analysis Agent")
        
    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis
        
        Analyzes:
        - Moving averages (SMA, EMA)
        - RSI, MACD, Bollinger Bands
        - Support/Resistance levels
        - Volume analysis
        - Pattern recognition hints
        """
        market_data = state.get("market_data", {})
        prices = market_data.get("historical_prices", [])
        volumes = market_data.get("historical_volumes", [])
        
        if len(prices) < analysis_config.sma_long:
            state["technical_analysis"] = {
                "error": "Insufficient historical data for technical analysis",
                "data_points": len(prices),
                "required": analysis_config.sma_long,
            }
            state["messages"].append("Technical analysis: Insufficient data")
            return state
        
        prices_array = np.array(prices, dtype=float)
        volumes_array = np.array(volumes, dtype=float) if volumes else None
        
        # Calculate all indicators
        indicators = self._calculate_indicators(prices_array, volumes_array)
        
        # Determine signals
        signals = self._determine_signals(indicators, prices_array[-1])
        
        # Get LLM interpretation
        interpretation = self._get_llm_interpretation(
            state.get("ticker", "UNKNOWN"),
            indicators,
            signals
        )
        
        state["technical_analysis"] = {
            "indicators": indicators,
            "signals": signals,
            "interpretation": interpretation,
            "current_price": float(prices_array[-1]),
        }
        state["messages"].append("Technical analysis completed")
        
        return state
    
    def _calculate_indicators(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        indicators = {}
        
        # Simple Moving Averages
        indicators["sma_20"] = float(np.mean(prices[-20:]))
        indicators["sma_50"] = float(np.mean(prices[-50:])) if len(prices) >= 50 else None
        
        # Exponential Moving Averages
        indicators["ema_12"] = float(self._calculate_ema(prices, 12))
        indicators["ema_26"] = float(self._calculate_ema(prices, 26))
        
        # MACD
        macd_line = indicators["ema_12"] - indicators["ema_26"]
        indicators["macd"] = macd_line
        indicators["macd_signal"] = float(self._calculate_ema(prices[-9:], 9)) if len(prices) >= 9 else 0
        indicators["macd_histogram"] = macd_line - indicators["macd_signal"]
        
        # RSI
        indicators["rsi_14"] = float(self._calculate_rsi(prices, 14))
        
        # Bollinger Bands
        bb = self._calculate_bollinger_bands(prices, analysis_config.bollinger_period, analysis_config.bollinger_std)
        indicators["bollinger_upper"] = bb["upper"]
        indicators["bollinger_middle"] = bb["middle"]
        indicators["bollinger_lower"] = bb["lower"]
        indicators["bollinger_width"] = bb["width"]
        
        # Average True Range (ATR) - simplified
        indicators["atr_14"] = float(self._calculate_atr(prices, 14))
        
        # Price momentum
        indicators["momentum_10"] = float((prices[-1] - prices[-10]) / prices[-10] * 100) if len(prices) >= 10 else None
        indicators["momentum_20"] = float((prices[-1] - prices[-20]) / prices[-20] * 100) if len(prices) >= 20 else None
        
        # Rate of Change
        indicators["roc_10"] = indicators["momentum_10"]
        
        # Stochastic Oscillator
        stoch = self._calculate_stochastic(prices, 14)
        indicators["stochastic_k"] = stoch["k"]
        indicators["stochastic_d"] = stoch["d"]
        
        # Volume indicators (if available)
        if volumes is not None and len(volumes) > 0:
            indicators["volume_sma_20"] = float(np.mean(volumes[-20:]))
            indicators["volume_ratio"] = float(volumes[-1] / np.mean(volumes[-20:])) if np.mean(volumes[-20:]) > 0 else 1
            indicators["obv_trend"] = self._calculate_obv_trend(prices, volumes)
        
        # Support and Resistance (simplified)
        sr_levels = self._find_support_resistance(prices)
        indicators["support_level"] = sr_levels["support"]
        indicators["resistance_level"] = sr_levels["resistance"]
        
        # Price position
        current = prices[-1]
        indicators["pct_from_20sma"] = float((current - indicators["sma_20"]) / indicators["sma_20"] * 100)
        
        if indicators["sma_50"]:
            indicators["pct_from_50sma"] = float((current - indicators["sma_50"]) / indicators["sma_50"] * 100)
        
        return indicators
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return float(np.mean(prices))
        
        multiplier = 2 / (period + 1)
        ema = prices[-period]
        
        for price in prices[-period + 1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int, num_std: float) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        return {
            "upper": float(sma + (num_std * std)),
            "middle": float(sma),
            "lower": float(sma - (num_std * std)),
            "width": float((2 * num_std * std) / sma * 100),  # Width as percentage
        }
    
    def _calculate_atr(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range (simplified - using close prices only)"""
        if len(prices) < period + 1:
            return 0.0
        
        true_ranges = np.abs(np.diff(prices[-period - 1:]))
        return float(np.mean(true_ranges))
    
    def _calculate_stochastic(self, prices: np.ndarray, period: int = 14) -> Dict[str, float]:
        """Calculate Stochastic Oscillator"""
        if len(prices) < period:
            return {"k": 50.0, "d": 50.0}
        
        recent = prices[-period:]
        high = np.max(recent)
        low = np.min(recent)
        current = prices[-1]
        
        if high == low:
            k = 50.0
        else:
            k = ((current - low) / (high - low)) * 100
        
        # %D is 3-period SMA of %K (simplified)
        d = k  # Would need multiple K values for proper calculation
        
        return {"k": float(k), "d": float(d)}
    
    def _calculate_obv_trend(self, prices: np.ndarray, volumes: np.ndarray) -> str:
        """Determine On-Balance Volume trend"""
        if len(prices) < 10 or len(volumes) < 10:
            return "unknown"
        
        obv = 0
        obv_values = []
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]
            obv_values.append(obv)
        
        # Compare recent OBV to earlier OBV
        recent_obv = np.mean(obv_values[-5:]) if len(obv_values) >= 5 else obv_values[-1]
        earlier_obv = np.mean(obv_values[-15:-5]) if len(obv_values) >= 15 else np.mean(obv_values[:5])
        
        if recent_obv > earlier_obv * 1.1:
            return "accumulation"
        elif recent_obv < earlier_obv * 0.9:
            return "distribution"
        else:
            return "neutral"
    
    def _find_support_resistance(self, prices: np.ndarray) -> Dict[str, float]:
        """Find basic support and resistance levels"""
        if len(prices) < 20:
            return {"support": float(np.min(prices)), "resistance": float(np.max(prices))}
        
        recent = prices[-30:] if len(prices) >= 30 else prices
        current = prices[-1]
        
        # Simple approach: use recent lows and highs
        lows = []
        highs = []
        
        for i in range(2, len(recent) - 2):
            if recent[i] < recent[i-1] and recent[i] < recent[i+1] and recent[i] < recent[i-2] and recent[i] < recent[i+2]:
                lows.append(recent[i])
            if recent[i] > recent[i-1] and recent[i] > recent[i+1] and recent[i] > recent[i-2] and recent[i] > recent[i+2]:
                highs.append(recent[i])
        
        support = float(np.mean(lows)) if lows else float(np.min(recent))
        resistance = float(np.mean(highs)) if highs else float(np.max(recent))
        
        return {"support": support, "resistance": resistance}
    
    def _determine_signals(self, indicators: Dict[str, Any], current_price: float) -> Dict[str, str]:
        """Determine trading signals from indicators"""
        signals = {}
        
        # Trend signal (based on SMAs)
        if indicators.get("sma_20") and indicators.get("sma_50"):
            if current_price > indicators["sma_20"] > indicators["sma_50"]:
                signals["trend"] = "strong_bullish"
            elif current_price > indicators["sma_20"]:
                signals["trend"] = "bullish"
            elif current_price < indicators["sma_20"] < indicators["sma_50"]:
                signals["trend"] = "strong_bearish"
            elif current_price < indicators["sma_20"]:
                signals["trend"] = "bearish"
            else:
                signals["trend"] = "neutral"
        
        # RSI signal
        rsi = indicators.get("rsi_14", 50)
        if rsi > 70:
            signals["rsi"] = "overbought"
        elif rsi < 30:
            signals["rsi"] = "oversold"
        elif rsi > 60:
            signals["rsi"] = "bullish"
        elif rsi < 40:
            signals["rsi"] = "bearish"
        else:
            signals["rsi"] = "neutral"
        
        # MACD signal
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        if macd > macd_signal:
            signals["macd"] = "bullish"
        else:
            signals["macd"] = "bearish"
        
        # Bollinger Bands signal
        bb_upper = indicators.get("bollinger_upper", current_price * 1.1)
        bb_lower = indicators.get("bollinger_lower", current_price * 0.9)
        
        if current_price > bb_upper:
            signals["bollinger"] = "overbought"
        elif current_price < bb_lower:
            signals["bollinger"] = "oversold"
        else:
            bb_middle = indicators.get("bollinger_middle", current_price)
            if current_price > bb_middle:
                signals["bollinger"] = "above_mean"
            else:
                signals["bollinger"] = "below_mean"
        
        # Stochastic signal
        stoch_k = indicators.get("stochastic_k", 50)
        if stoch_k > 80:
            signals["stochastic"] = "overbought"
        elif stoch_k < 20:
            signals["stochastic"] = "oversold"
        else:
            signals["stochastic"] = "neutral"
        
        # Volume signal
        volume_ratio = indicators.get("volume_ratio", 1)
        if volume_ratio > 1.5:
            signals["volume"] = "high"
        elif volume_ratio < 0.7:
            signals["volume"] = "low"
        else:
            signals["volume"] = "normal"
        
        # OBV trend
        signals["obv"] = indicators.get("obv_trend", "unknown")
        
        # Support/Resistance proximity
        support = indicators.get("support_level", current_price * 0.95)
        resistance = indicators.get("resistance_level", current_price * 1.05)
        
        if current_price <= support * 1.02:
            signals["sr_position"] = "near_support"
        elif current_price >= resistance * 0.98:
            signals["sr_position"] = "near_resistance"
        else:
            signals["sr_position"] = "mid_range"
        
        # Overall signal (simple aggregation)
        bullish_count = sum(1 for v in signals.values() if v in ["bullish", "strong_bullish", "oversold", "accumulation", "near_support"])
        bearish_count = sum(1 for v in signals.values() if v in ["bearish", "strong_bearish", "overbought", "distribution", "near_resistance"])
        
        if bullish_count > bearish_count + 2:
            signals["overall"] = "bullish"
        elif bearish_count > bullish_count + 2:
            signals["overall"] = "bearish"
        else:
            signals["overall"] = "neutral"
        
        return signals
    
    def _get_llm_interpretation(self, ticker: str, indicators: Dict, signals: Dict) -> str:
        """Get LLM interpretation of technical analysis"""
        prompt = self._create_prompt("""Analyze the following technical indicators for {ticker} and provide a brief technical outlook.

Key Indicators:
- Current Price: ${current_price:.2f}
- 20-day SMA: ${sma_20:.2f} (Price {pct_from_20sma:+.1f}% from SMA)
- RSI(14): {rsi:.1f}
- MACD: {macd:.2f}
- Bollinger Width: {bb_width:.1f}%
- Stochastic %K: {stoch_k:.1f}

Signals Detected:
- Trend: {trend_signal}
- RSI Signal: {rsi_signal}
- MACD Signal: {macd_signal}
- Volume: {volume_signal}
- S/R Position: {sr_signal}
- Overall: {overall_signal}

Provide a concise 3-4 sentence technical analysis summary with actionable insights.""")
        
        try:
            interpretation = self._invoke_llm(
                prompt,
                ticker=ticker,
                current_price=indicators.get("sma_20", 0) if "sma_20" in indicators else 0,
                sma_20=indicators.get("sma_20", 0),
                pct_from_20sma=indicators.get("pct_from_20sma", 0),
                rsi=indicators.get("rsi_14", 50),
                macd=indicators.get("macd", 0),
                bb_width=indicators.get("bollinger_width", 0),
                stoch_k=indicators.get("stochastic_k", 50),
                trend_signal=signals.get("trend", "neutral"),
                rsi_signal=signals.get("rsi", "neutral"),
                macd_signal=signals.get("macd", "neutral"),
                volume_signal=signals.get("volume", "normal"),
                sr_signal=signals.get("sr_position", "mid_range"),
                overall_signal=signals.get("overall", "neutral"),
            )
            return interpretation
        except Exception as e:
            return f"LLM interpretation unavailable: {str(e)}"

