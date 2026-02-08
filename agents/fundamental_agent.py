"""
Fundamental Analysis Agent
Analyzes company fundamentals and economic indicators
"""
from typing import Dict, Any
from .base_agent import BaseAgent


class FundamentalAnalysisAgent(BaseAgent):
    """Agent specialized in fundamental analysis"""
    
    def __init__(self):
        super().__init__("Fundamental Analysis Agent")
    
    def analyze(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis
        
        Analyzes:
        - Company valuation metrics (P/E, P/B, PEG)
        - Growth metrics
        - Financial health indicators
        - Economic environment impact
        """
        ticker = state.get("ticker", "UNKNOWN")
        market_data = state.get("market_data", {})
        economic_data = state.get("economic_indicators", {})
        
        # Extract fundamental metrics
        fundamentals = self._extract_fundamentals(market_data)
        
        # Analyze valuation
        valuation_analysis = self._analyze_valuation(fundamentals)
        
        # Assess economic environment impact
        economic_impact = self._assess_economic_impact(economic_data, market_data)
        
        # Get LLM interpretation
        interpretation = self._get_llm_interpretation(
            ticker, 
            fundamentals, 
            valuation_analysis,
            economic_impact,
            economic_data
        )
        
        state["fundamental_analysis"] = {
            "metrics": fundamentals,
            "valuation_assessment": valuation_analysis,
            "economic_impact": economic_impact,
            "interpretation": interpretation,
        }
        state["messages"].append("Fundamental analysis completed")
        
        return state
    
    def _extract_fundamentals(self, market_data: Dict) -> Dict[str, Any]:
        """Extract key fundamental metrics from market data"""
        return {
            "pe_ratio": market_data.get("pe_ratio"),
            "forward_pe": market_data.get("forward_pe"),
            "peg_ratio": market_data.get("peg_ratio"),
            "price_to_book": market_data.get("price_to_book"),
            "eps": market_data.get("eps"),
            "dividend_yield": market_data.get("dividend_yield"),
            "market_cap": market_data.get("market_cap"),
            "beta": market_data.get("beta"),
            "52_week_high": market_data.get("52_week_high"),
            "52_week_low": market_data.get("52_week_low"),
            "50_day_avg": market_data.get("50_day_avg"),
            "200_day_avg": market_data.get("200_day_avg"),
            "shares_short": market_data.get("shares_short"),
            "short_ratio": market_data.get("short_ratio"),
            "current_price": market_data.get("current_price"),
        }
    
    def _analyze_valuation(self, fundamentals: Dict) -> Dict[str, Any]:
        """Analyze valuation metrics"""
        assessment = {
            "pe_assessment": "N/A",
            "peg_assessment": "N/A",
            "52w_position": "N/A",
            "overall_valuation": "N/A",
        }
        
        # P/E Ratio assessment
        pe = fundamentals.get("pe_ratio")
        if pe is not None:
            if pe < 0:
                assessment["pe_assessment"] = "negative_earnings"
            elif pe < 15:
                assessment["pe_assessment"] = "undervalued"
            elif pe < 25:
                assessment["pe_assessment"] = "fairly_valued"
            elif pe < 40:
                assessment["pe_assessment"] = "growth_premium"
            else:
                assessment["pe_assessment"] = "highly_valued"
        
        # PEG Ratio assessment
        peg = fundamentals.get("peg_ratio")
        if peg is not None:
            if peg < 1:
                assessment["peg_assessment"] = "undervalued_vs_growth"
            elif peg < 2:
                assessment["peg_assessment"] = "fairly_valued"
            else:
                assessment["peg_assessment"] = "expensive_vs_growth"
        
        # 52-week position
        current = fundamentals.get("current_price")
        high_52w = fundamentals.get("52_week_high")
        low_52w = fundamentals.get("52_week_low")
        
        if current and high_52w and low_52w and high_52w != low_52w:
            position = (current - low_52w) / (high_52w - low_52w) * 100
            assessment["52w_position_pct"] = round(position, 1)
            
            if position > 90:
                assessment["52w_position"] = "near_high"
            elif position > 70:
                assessment["52w_position"] = "upper_range"
            elif position > 30:
                assessment["52w_position"] = "mid_range"
            elif position > 10:
                assessment["52w_position"] = "lower_range"
            else:
                assessment["52w_position"] = "near_low"
        
        # Short interest analysis
        short_ratio = fundamentals.get("short_ratio")
        if short_ratio is not None:
            if short_ratio > 10:
                assessment["short_interest"] = "very_high"
            elif short_ratio > 5:
                assessment["short_interest"] = "high"
            elif short_ratio > 2:
                assessment["short_interest"] = "moderate"
            else:
                assessment["short_interest"] = "low"
        
        # Overall valuation score
        valuation_signals = []
        
        if assessment["pe_assessment"] in ["undervalued"]:
            valuation_signals.append(1)
        elif assessment["pe_assessment"] in ["highly_valued"]:
            valuation_signals.append(-1)
        
        if assessment["peg_assessment"] in ["undervalued_vs_growth"]:
            valuation_signals.append(1)
        elif assessment["peg_assessment"] in ["expensive_vs_growth"]:
            valuation_signals.append(-1)
        
        if assessment["52w_position"] in ["near_low", "lower_range"]:
            valuation_signals.append(1)
        elif assessment["52w_position"] in ["near_high"]:
            valuation_signals.append(-1)
        
        if valuation_signals:
            avg_signal = sum(valuation_signals) / len(valuation_signals)
            if avg_signal > 0.3:
                assessment["overall_valuation"] = "attractive"
            elif avg_signal < -0.3:
                assessment["overall_valuation"] = "expensive"
            else:
                assessment["overall_valuation"] = "fair"
        
        return assessment
    
    def _assess_economic_impact(self, economic_data: Dict, market_data: Dict) -> Dict[str, Any]:
        """Assess how current economic conditions affect the stock"""
        impact = {
            "interest_rate_sensitivity": "N/A",
            "inflation_impact": "N/A",
            "economic_cycle_position": "N/A",
        }
        
        if "error" in economic_data:
            return {"status": "Economic data unavailable"}
        
        # Interest rate sensitivity (based on beta and sector)
        beta = market_data.get("beta", 1.0)
        sector = market_data.get("sector", "")
        
        rate_sensitive_sectors = ["Real Estate", "Utilities", "Financial Services", "Banks"]
        
        if sector in rate_sensitive_sectors:
            impact["interest_rate_sensitivity"] = "high"
        elif beta and beta > 1.2:
            impact["interest_rate_sensitivity"] = "moderate_high"
        elif beta and beta < 0.8:
            impact["interest_rate_sensitivity"] = "low"
        else:
            impact["interest_rate_sensitivity"] = "moderate"
        
        # Inflation impact assessment
        consumer_sectors = ["Consumer Discretionary", "Retail", "Consumer Cyclical"]
        defensive_sectors = ["Consumer Defensive", "Healthcare", "Utilities"]
        
        if sector in consumer_sectors:
            impact["inflation_impact"] = "negative"
        elif sector in defensive_sectors:
            impact["inflation_impact"] = "neutral"
        else:
            impact["inflation_impact"] = "mixed"
        
        # Get specific economic indicators
        try:
            fed_rate = None
            unemployment = None
            
            for key, data in economic_data.items():
                if isinstance(data, dict):
                    if "Fed" in data.get("description", ""):
                        fed_rate = data.get("latest_value")
                    elif "Unemployment" in data.get("description", ""):
                        unemployment = data.get("latest_value")
            
            impact["current_fed_rate"] = fed_rate
            impact["current_unemployment"] = unemployment
            
            # Economic cycle assessment
            if unemployment is not None:
                if unemployment < 4:
                    impact["labor_market"] = "tight"
                elif unemployment < 5.5:
                    impact["labor_market"] = "healthy"
                elif unemployment < 7:
                    impact["labor_market"] = "weakening"
                else:
                    impact["labor_market"] = "weak"
        except:
            pass
        
        return impact
    
    def _get_llm_interpretation(self, ticker: str, fundamentals: Dict,
                                valuation: Dict, economic_impact: Dict,
                                economic_data: Dict) -> str:
        """Get LLM interpretation of fundamental analysis"""
        
        # Format economic data summary
        economic_summary = []
        if not isinstance(economic_data, dict) or "error" in economic_data:
            economic_summary.append("Economic data: Unavailable")
        else:
            for key, data in list(economic_data.items())[:6]:
                if isinstance(data, dict) and "latest_value" in data:
                    desc = data.get("description", key)
                    value = data.get("latest_value", "N/A")
                    economic_summary.append(f"- {desc}: {value}")
        
        economic_summary_text = "\n".join(economic_summary) if economic_summary else "No economic data"
        
        prompt = self._create_prompt("""Analyze the fundamental strength of {ticker} given current market conditions.

VALUATION METRICS:
- P/E Ratio: {pe_ratio} ({pe_assessment})
- PEG Ratio: {peg_ratio}
- Price to Book: {price_to_book}
- 52-Week Position: {position_52w}

MARKET DATA:
- Market Cap: ${market_cap}
- Beta: {beta}
- Short Ratio: {short_ratio}

ECONOMIC ENVIRONMENT:
{economic_summary}

IMPACT ASSESSMENT:
- Interest Rate Sensitivity: {rate_sensitivity}
- Inflation Impact: {inflation_impact}

Provide a concise fundamental assessment including:
1. Valuation verdict (undervalued/fairly valued/overvalued)
2. Key fundamental strengths and weaknesses
3. How current economic conditions affect this stock
4. Fundamental outlook (bullish/neutral/bearish)

Keep response under 150 words.""")
        
        try:
            interpretation = self._invoke_llm(
                prompt,
                ticker=ticker,
                pe_ratio=fundamentals.get("pe_ratio", "N/A"),
                pe_assessment=valuation.get("pe_assessment", "N/A"),
                peg_ratio=fundamentals.get("peg_ratio", "N/A"),
                price_to_book=fundamentals.get("price_to_book", "N/A"),
                position_52w=valuation.get("52w_position", "N/A"),
                market_cap=fundamentals.get("market_cap", "N/A"),
                beta=fundamentals.get("beta", "N/A"),
                short_ratio=fundamentals.get("short_ratio", "N/A"),
                economic_summary=economic_summary_text,
                rate_sensitivity=economic_impact.get("interest_rate_sensitivity", "N/A"),
                inflation_impact=economic_impact.get("inflation_impact", "N/A"),
            )
            return interpretation
        except Exception as e:
            return f"Fundamental interpretation unavailable: {str(e)}"

