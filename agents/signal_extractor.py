"""
SignalExtractor — converts AgentState into a 44-dim float32 feature vector.
No LLM calls; pure deterministic maths.
"""
import numpy as np
from typing import Dict, Any, List

FEATURE_VERSION = "1.0"
N_FEATURES = 44

FEATURE_NAMES: List[str] = [
    # Group 1 (0-9): Technical indicators
    "rsi_norm", "stoch_k_norm", "bb_position", "bb_width_norm",
    "macd_hist_norm", "pct_from_20sma", "pct_from_50sma",
    "momentum_10_norm", "volume_ratio_norm", "atr_pct_norm",
    # Group 2 (10-14): Technical signals (ordinal)
    "trend_signal", "rsi_signal", "macd_signal", "obv_signal", "sr_position_signal",
    # Group 3 (15-22): Risk metrics
    "annualized_vol_norm", "max_drawdown_norm", "var_95_norm",
    "sharpe_norm", "win_rate_norm", "gain_loss_ratio_norm",
    "risk_level_encoded", "vol_regime_encoded",
    # Group 4 (23-27): Sentiment
    "news_sentiment_encoded", "news_count_norm", "political_impact_encoded",
    "llm_sentiment_bias", "news_positive_ratio",
    # Group 5 (28-31): Reddit
    "reddit_available", "retail_sentiment_encoded", "reddit_score_norm", "wsb_trending_flag",
    # Group 6 (32-37): Fundamental
    "pe_norm", "peg_norm", "w52_position_norm", "valuation_encoded",
    "beta_norm", "pe_assessment_encoded",
    # Group 7 (38-40): Portfolio scores
    "technical_score_norm", "sentiment_score_norm", "composite_score_norm",
    # Group 8 (41-43): Market context
    "log_price_norm", "market_cap_tier", "momentum_divergence",
]

assert len(FEATURE_NAMES) == N_FEATURES


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(np.clip(v, lo, hi))


class SignalExtractor:
    """Converts a fully-populated AgentState dict into a numeric feature vector."""

    def extract(self, state: Dict[str, Any]) -> Dict[str, Any]:
        features: Dict[str, float] = {}
        missing: List[str] = []

        self._extract_technical(state, features, missing)
        self._extract_risk(state, features, missing)
        self._extract_sentiment(state, features, missing)
        self._extract_reddit(state, features, missing)
        self._extract_fundamental(state, features, missing)
        self._extract_portfolio_scores(state, features, missing)
        self._extract_market_context(state, features, missing)

        vector = np.array([features.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float32)
        data_quality = 1.0 - len(set(missing)) / 8.0

        return {
            "vector": vector,
            "features": features,
            "data_quality": round(data_quality, 3),
            "missing": list(set(missing)),
            "version": FEATURE_VERSION,
        }

    def _extract_technical(self, state: Dict, f: Dict, missing: List) -> None:
        tech = state.get("technical_analysis", {})
        if not tech or "error" in tech:
            missing.append("technical")
            f.update({k: 0.0 for k in FEATURE_NAMES[0:15]})
            return

        ind = tech.get("indicators", {})
        sig = tech.get("signals", {})
        price = tech.get("current_price") or state.get("market_data", {}).get("current_price", 1.0)
        if not price:
            price = 1.0

        rsi = ind.get("rsi_14", 50.0) or 50.0
        f["rsi_norm"] = _clip(rsi / 100.0, 0.0, 1.0)

        stoch_k = ind.get("stochastic_k", 50.0) or 50.0
        f["stoch_k_norm"] = _clip(stoch_k / 100.0, 0.0, 1.0)

        bb_upper = ind.get("bollinger_upper", price * 1.02) or price * 1.02
        bb_lower = ind.get("bollinger_lower", price * 0.98) or price * 0.98
        bb_range = bb_upper - bb_lower
        f["bb_position"] = _clip((price - bb_lower) / bb_range, 0.0, 1.0) if bb_range > 0 else 0.5
        bb_width = ind.get("bollinger_width", bb_range) or bb_range
        f["bb_width_norm"] = _clip(bb_width / price, 0.0, 1.0)

        macd_hist = ind.get("macd_histogram", 0.0) or 0.0
        f["macd_hist_norm"] = _clip(macd_hist / (price * 0.02))

        sma20 = ind.get("sma_20", price) or price
        f["pct_from_20sma"] = _clip((price - sma20) / sma20 * 10) if sma20 else 0.0
        sma50 = ind.get("sma_50")
        f["pct_from_50sma"] = _clip((price - sma50) / sma50 * 10) if sma50 else 0.0

        mom10 = ind.get("momentum_10", 0.0) or 0.0
        f["momentum_10_norm"] = _clip(mom10 / 20.0)

        vol_ratio = ind.get("volume_ratio", 1.0) or 1.0
        f["volume_ratio_norm"] = _clip((vol_ratio - 1.0) / 2.0)

        atr = ind.get("atr_14", price * 0.02) or price * 0.02
        f["atr_pct_norm"] = _clip(atr / price, 0.0, 1.0)

        trend_map = {
            "strong_bullish": 1.0, "bullish": 0.5, "neutral": 0.0,
            "bearish": -0.5, "strong_bearish": -1.0,
        }
        f["trend_signal"] = trend_map.get(sig.get("trend", "neutral"), 0.0)

        rsi_map = {
            "oversold": 1.0, "bullish": 0.5, "neutral": 0.0,
            "bearish": -0.5, "overbought": -1.0,
        }
        f["rsi_signal"] = rsi_map.get(sig.get("rsi", "neutral"), 0.0)

        macd_map = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
        f["macd_signal"] = macd_map.get(sig.get("macd", "neutral"), 0.0)

        obv_map = {"accumulation": 1.0, "neutral": 0.0, "distribution": -1.0}
        f["obv_signal"] = obv_map.get(sig.get("obv", "neutral"), 0.0)

        sr_map = {"near_support": 1.0, "neutral": 0.0, "near_resistance": -1.0}
        f["sr_position_signal"] = sr_map.get(sig.get("sr_position", "neutral"), 0.0)

    def _extract_risk(self, state: Dict, f: Dict, missing: List) -> None:
        risk = state.get("risk_assessment", {})
        if not risk or "error" in risk:
            missing.append("risk")
            f.update({k: 0.0 for k in FEATURE_NAMES[15:23]})
            return

        metrics = risk.get("risk_metrics", {})
        rl = risk.get("risk_level", {})

        ann_vol = metrics.get("annualized_volatility", 20.0) or 20.0
        f["annualized_vol_norm"] = _clip(ann_vol / 100.0, 0.0, 1.0)

        max_dd = abs(metrics.get("max_drawdown", 0.0) or 0.0)
        f["max_drawdown_norm"] = _clip(max_dd / 50.0, 0.0, 1.0)

        var95 = abs(metrics.get("var_95_daily", 0.0) or 0.0)
        f["var_95_norm"] = _clip(var95 / 5.0, 0.0, 1.0)

        sharpe = metrics.get("sharpe_ratio", 0.0) or 0.0
        f["sharpe_norm"] = _clip(sharpe / 3.0)

        win_rate = metrics.get("win_rate", 50.0) or 50.0
        f["win_rate_norm"] = _clip(win_rate / 100.0, 0.0, 1.0)

        gl_ratio = metrics.get("gain_loss_ratio", 1.0) or 1.0
        f["gain_loss_ratio_norm"] = _clip(gl_ratio / 3.0, 0.0, 1.0)

        risk_level_map = {"low": 0.0, "moderate": 0.33, "high": 0.67, "very_high": 1.0}
        f["risk_level_encoded"] = risk_level_map.get(rl.get("level", "moderate"), 0.33)

        vol_regime_map = {"subdued": 0.0, "normal": 0.5, "elevated": 1.0}
        f["vol_regime_encoded"] = vol_regime_map.get(
            metrics.get("volatility_regime", "normal"), 0.5
        )

    def _extract_sentiment(self, state: Dict, f: Dict, missing: List) -> None:
        sent = state.get("sentiment_analysis", {})
        if not sent or "error" in sent:
            missing.append("sentiment")
            f.update({k: 0.0 for k in FEATURE_NAMES[23:28]})
            return

        news = sent.get("stock_news_summary", {})
        pol = sent.get("political_summary", {})
        llm_text = sent.get("llm_analysis", "").lower()

        news_sent_map = {"positive": 1.0, "mixed": 0.0, "negative": -1.0}
        f["news_sentiment_encoded"] = news_sent_map.get(news.get("sentiment", "mixed"), 0.0)

        count = news.get("count", 0) or 0
        f["news_count_norm"] = _clip(count / 20.0, 0.0, 1.0)

        pol_map = {"high": 1.0, "medium": 0.5, "low": 0.25, "none": 0.0}
        f["political_impact_encoded"] = pol_map.get(pol.get("potential_impact", "none"), 0.0)

        if "strong bullish" in llm_text or "very bullish" in llm_text:
            bias = 1.0
        elif "bullish" in llm_text:
            bias = 0.5
        elif "strong bearish" in llm_text or "very bearish" in llm_text:
            bias = -1.0
        elif "bearish" in llm_text:
            bias = -0.5
        else:
            bias = 0.0
        f["llm_sentiment_bias"] = bias

        pos = news.get("positive_count", 0) or 0
        neg = news.get("negative_count", 0) or 0
        total = pos + neg
        f["news_positive_ratio"] = _clip(pos / total, 0.0, 1.0) if total > 0 else 0.5

    def _extract_reddit(self, state: Dict, f: Dict, missing: List) -> None:
        reddit = state.get("reddit_analysis", {})
        available = reddit.get("available", False)
        f["reddit_available"] = 1.0 if available else 0.0

        if not available:
            f["retail_sentiment_encoded"] = 0.0
            f["reddit_score_norm"] = 0.0
            f["wsb_trending_flag"] = 0.0
            return

        analysis = reddit.get("analysis", {})
        retail_map = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0, "unknown": 0.0}
        f["retail_sentiment_encoded"] = retail_map.get(
            analysis.get("retail_sentiment", "unknown"), 0.0
        )

        score = analysis.get("sentiment_score", 0.0) or 0.0
        f["reddit_score_norm"] = _clip(score)

        raw = reddit.get("raw_data", {})
        wsb = raw.get("wsb_trending", {})
        f["wsb_trending_flag"] = 1.0 if wsb.get("is_trending", False) else 0.0

    def _extract_fundamental(self, state: Dict, f: Dict, missing: List) -> None:
        fund = state.get("fundamental_analysis", {})
        market_data = state.get("market_data", {})

        if not fund or "error" in fund:
            missing.append("fundamental")
            f.update({k: 0.0 for k in FEATURE_NAMES[32:38]})
            return

        valuation = fund.get("valuation_assessment", {})
        fundamentals = fund.get("fundamentals", {}) or market_data

        pe = fundamentals.get("pe_ratio") or market_data.get("pe_ratio")
        if pe and pe > 0:
            f["pe_norm"] = _clip(1.0 - pe / 60.0, -1.0, 1.0)
        else:
            f["pe_norm"] = 0.0

        peg = fundamentals.get("peg_ratio") or market_data.get("peg_ratio")
        if peg and peg > 0:
            f["peg_norm"] = _clip(1.0 - peg / 3.0, -1.0, 1.0)
        else:
            f["peg_norm"] = 0.0

        price = market_data.get("current_price", 0) or 0
        high52 = market_data.get("52_week_high", 0) or 0
        low52 = market_data.get("52_week_low", 0) or 0
        if high52 and low52 and high52 != low52 and price:
            f["w52_position_norm"] = _clip((price - low52) / (high52 - low52), 0.0, 1.0)
        else:
            f["w52_position_norm"] = 0.5

        val_map = {"attractive": 1.0, "fair": 0.0, "expensive": -1.0, "N/A": 0.0}
        f["valuation_encoded"] = val_map.get(valuation.get("overall_valuation", "N/A"), 0.0)

        beta = market_data.get("beta")
        if beta is not None:
            f["beta_norm"] = _clip(beta / 2.0)
        else:
            est_beta = state.get("risk_assessment", {}).get("risk_metrics", {}).get("estimated_beta")
            f["beta_norm"] = _clip(est_beta / 2.0) if est_beta is not None else 0.0

        pe_assess = valuation.get("pe_assessment", "N/A")
        pe_assess_map = {
            "undervalued": 1.0, "fairly_valued": 0.0, "expensive": -1.0, "N/A": 0.0,
        }
        f["pe_assessment_encoded"] = pe_assess_map.get(pe_assess, 0.0)

    def _extract_portfolio_scores(self, state: Dict, f: Dict, missing: List) -> None:
        fr = state.get("final_recommendation", {})
        scores = fr.get("scores", {})

        if not scores:
            missing.append("portfolio_scores")
            f["technical_score_norm"] = 0.0
            f["sentiment_score_norm"] = 0.0
            f["composite_score_norm"] = 0.0
            return

        f["technical_score_norm"] = _clip(scores.get("technical_score", 0.0) / 2.0)
        f["sentiment_score_norm"] = _clip(scores.get("sentiment_score", 0.0) / 2.0)
        f["composite_score_norm"] = _clip(scores.get("composite_score", 0.0) / 2.0)

    def _extract_market_context(self, state: Dict, f: Dict, missing: List) -> None:
        market = state.get("market_data", {})
        price = market.get("current_price", 0) or 0

        if price > 0:
            log_p = float(np.log(price))
            f["log_price_norm"] = _clip(log_p / 9.21, 0.0, 1.0)
        else:
            f["log_price_norm"] = 0.5

        mkt_cap = market.get("market_cap", 0) or 0
        if mkt_cap >= 200e9:
            tier = 1.0
        elif mkt_cap >= 10e9:
            tier = 0.75
        elif mkt_cap >= 2e9:
            tier = 0.5
        elif mkt_cap > 0:
            tier = 0.25
        else:
            tier = 0.0
        f["market_cap_tier"] = tier

        tech_score = f.get("technical_score_norm", 0.0)
        sent_score = f.get("sentiment_score_norm", 0.0)
        f["momentum_divergence"] = _clip(tech_score - sent_score)
