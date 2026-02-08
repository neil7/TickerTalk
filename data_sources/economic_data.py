"""
Economic Data Fetcher
Handles Federal Reserve (FRED) data and other economic indicators
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from config import api_config, FRED_INDICATORS


class EconomicDataFetcher:
    """Fetches economic indicators from FRED and other sources"""
    
    def __init__(self):
        self.fred_api_key = api_config.fred_api_key
        self.fred = None
        self._init_fred()
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
        
    def _init_fred(self):
        """Initialize FRED API client if key is available"""
        if self.fred_api_key:
            try:
                from fredapi import Fred
                self.fred = Fred(api_key=self.fred_api_key)
            except ImportError:
                self.fred = None
    
    def get_all_indicators(self) -> Dict[str, Any]:
        """
        Fetch all key economic indicators
        These are crucial for understanding market environment
        """
        if not self.fred:
            return {"error": "FRED API not configured. Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html"}
        
        # Check cache (indicators update infrequently)
        if self._cache_time and (datetime.now() - self._cache_time).seconds < 3600:
            return self._cache
        
        indicators = {}
        
        for series_id, description in FRED_INDICATORS.items():
            try:
                data = self.fred.get_series_latest_release(series_id)
                if data is not None and len(data) > 0:
                    latest_value = float(data.iloc[-1])
                    previous_value = float(data.iloc[-2]) if len(data) > 1 else None
                    
                    change = None
                    if previous_value is not None:
                        change = latest_value - previous_value
                    
                    indicators[series_id] = {
                        "description": description,
                        "latest_value": latest_value,
                        "previous_value": previous_value,
                        "change": change,
                        "date": str(data.index[-1].date()) if hasattr(data.index[-1], 'date') else str(data.index[-1]),
                    }
            except Exception as e:
                indicators[series_id] = {
                    "description": description,
                    "error": str(e)
                }
        
        indicators["fetch_time"] = datetime.now().isoformat()
        
        # Update cache
        self._cache = indicators
        self._cache_time = datetime.now()
        
        return indicators
    
    def get_inflation_data(self) -> Dict[str, Any]:
        """Get detailed inflation metrics"""
        if not self.fred:
            return {"error": "FRED API not configured"}
        
        try:
            inflation_series = {
                "CPIAUCSL": "Consumer Price Index (All Urban)",
                "CPILFESL": "Core CPI (Less Food & Energy)",
                "PCEPI": "PCE Price Index",
                "PCEPILFE": "Core PCE",
                "PPIACO": "Producer Price Index",
            }
            
            results = {}
            for series_id, description in inflation_series.items():
                try:
                    data = self.fred.get_series_latest_release(series_id)
                    if data is not None and len(data) >= 13:
                        latest = float(data.iloc[-1])
                        year_ago = float(data.iloc[-13])  # ~12 months ago
                        yoy_change = ((latest - year_ago) / year_ago) * 100
                        
                        results[series_id] = {
                            "description": description,
                            "latest": latest,
                            "yoy_change_pct": round(yoy_change, 2),
                            "date": str(data.index[-1].date()) if hasattr(data.index[-1], 'date') else str(data.index[-1]),
                        }
                except Exception as e:
                    results[series_id] = {"error": str(e)}
            
            return results
        except Exception as e:
            return {"error": str(e)}
    
    def get_labor_data(self) -> Dict[str, Any]:
        """Get employment and labor market data"""
        if not self.fred:
            return {"error": "FRED API not configured"}
        
        try:
            labor_series = {
                "UNRATE": "Unemployment Rate",
                "PAYEMS": "Total Nonfarm Payrolls",
                "ICSA": "Initial Jobless Claims",
                "JTSJOL": "Job Openings (JOLTS)",
                "AWHMAN": "Avg Weekly Hours (Manufacturing)",
            }
            
            results = {}
            for series_id, description in labor_series.items():
                try:
                    data = self.fred.get_series_latest_release(series_id)
                    if data is not None and len(data) > 0:
                        results[series_id] = {
                            "description": description,
                            "latest": float(data.iloc[-1]),
                            "previous": float(data.iloc[-2]) if len(data) > 1 else None,
                            "date": str(data.index[-1].date()) if hasattr(data.index[-1], 'date') else str(data.index[-1]),
                        }
                except Exception as e:
                    results[series_id] = {"error": str(e)}
            
            return results
        except Exception as e:
            return {"error": str(e)}
    
    def get_interest_rates(self) -> Dict[str, Any]:
        """Get interest rate data"""
        if not self.fred:
            return {"error": "FRED API not configured"}
        
        try:
            rate_series = {
                "FEDFUNDS": "Federal Funds Rate",
                "DFF": "Effective Fed Funds Rate",
                "DGS2": "2-Year Treasury",
                "DGS10": "10-Year Treasury",
                "DGS30": "30-Year Treasury",
                "T10Y2Y": "10Y-2Y Spread (Yield Curve)",
                "MORTGAGE30US": "30-Year Mortgage Rate",
            }
            
            results = {}
            for series_id, description in rate_series.items():
                try:
                    data = self.fred.get_series_latest_release(series_id)
                    if data is not None and len(data) > 0:
                        results[series_id] = {
                            "description": description,
                            "latest": float(data.iloc[-1]),
                            "date": str(data.index[-1].date()) if hasattr(data.index[-1], 'date') else str(data.index[-1]),
                        }
                except Exception as e:
                    results[series_id] = {"error": str(e)}
            
            return results
        except Exception as e:
            return {"error": str(e)}
    
    def get_market_indicators(self) -> Dict[str, Any]:
        """Get market-related indicators from FRED"""
        if not self.fred:
            return {"error": "FRED API not configured"}
        
        try:
            market_series = {
                "VIXCLS": "VIX Volatility Index",
                "DTWEXBGS": "Trade Weighted Dollar Index",
                "DCOILWTICO": "WTI Crude Oil Price",
                "GOLDAMGBD228NLBM": "Gold Price (London)",
                "SP500": "S&P 500 Index",
            }
            
            results = {}
            for series_id, description in market_series.items():
                try:
                    data = self.fred.get_series_latest_release(series_id)
                    if data is not None and len(data) > 0:
                        results[series_id] = {
                            "description": description,
                            "latest": float(data.iloc[-1]),
                            "date": str(data.index[-1].date()) if hasattr(data.index[-1], 'date') else str(data.index[-1]),
                        }
                except Exception as e:
                    results[series_id] = {"error": str(e)}
            
            return results
        except Exception as e:
            return {"error": str(e)}
    
    def get_comprehensive_economic_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all economic data
        Suitable for feeding into the AI agent
        """
        return {
            "all_indicators": self.get_all_indicators(),
            "inflation": self.get_inflation_data(),
            "labor": self.get_labor_data(),
            "interest_rates": self.get_interest_rates(),
            "market_indicators": self.get_market_indicators(),
            "fetch_time": datetime.now().isoformat(),
        }
    
    def get_summary_for_prompt(self) -> str:
        """
        Get a formatted text summary suitable for LLM prompts
        """
        indicators = self.get_all_indicators()
        
        if "error" in indicators:
            return f"Economic data unavailable: {indicators['error']}"
        
        summary_parts = ["Current Economic Indicators:"]
        
        for series_id, data in indicators.items():
            if series_id == "fetch_time":
                continue
            if isinstance(data, dict) and "error" not in data:
                desc = data.get("description", series_id)
                value = data.get("latest_value", "N/A")
                summary_parts.append(f"- {desc}: {value}")
        
        return "\n".join(summary_parts)

