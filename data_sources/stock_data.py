"""
Stock Market Data Fetcher
Handles real-time and historical stock data from multiple sources
"""
import yfinance as yf
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from config import api_config, analysis_config


class StockDataFetcher:
    """Fetches stock market data from various APIs"""
    
    def __init__(self):
        self.alpha_vantage_key = api_config.alpha_vantage_key
        self.stockdata_key = api_config.stockdata_key
        self._cache: Dict[str, Dict] = {}
        
    def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch comprehensive stock data for a given ticker
        
        Returns:
            Dict containing current price, volume, fundamentals, and historical data
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{analysis_config.historical_days}d")
            info = stock.info
            
            # Calculate additional metrics
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
            
            change_pct = None
            if current_price and prev_close:
                change_pct = ((current_price - prev_close) / prev_close) * 100
            
            return {
                "ticker": ticker.upper(),
                "company_name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "current_price": current_price,
                "previous_close": prev_close,
                "change_percent": change_pct,
                "volume": info.get("volume") or info.get("regularMarketVolume"),
                "avg_volume": info.get("averageVolume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "eps": info.get("trailingEps"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "50_day_avg": info.get("fiftyDayAverage"),
                "200_day_avg": info.get("twoHundredDayAverage"),
                "beta": info.get("beta"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "shares_short": info.get("sharesShort"),
                "short_ratio": info.get("shortRatio"),
                "historical_prices": hist['Close'].tolist() if len(hist) > 0 else [],
                "historical_volumes": hist['Volume'].tolist() if len(hist) > 0 else [],
                "historical_highs": hist['High'].tolist() if len(hist) > 0 else [],
                "historical_lows": hist['Low'].tolist() if len(hist) > 0 else [],
                "historical_dates": [d.strftime('%Y-%m-%d') for d in hist.index] if len(hist) > 0 else [],
                "fetch_time": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "ticker": ticker.upper(),
                "error": str(e),
                "fetch_time": datetime.now().isoformat(),
            }
    
    def get_intraday_data(self, ticker: str, interval: str = "5min") -> Dict[str, Any]:
        """
        Fetch intraday price data using Alpha Vantage
        
        Args:
            ticker: Stock symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
        """
        if self.alpha_vantage_key == "demo":
            return {"error": "Alpha Vantage API key not configured"}
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": ticker,
                "interval": interval,
                "apikey": self.alpha_vantage_key,
                "outputsize": "compact"
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if "Error Message" in data:
                return {"error": data["Error Message"]}
            
            time_series_key = f"Time Series ({interval})"
            if time_series_key not in data:
                return {"error": "No intraday data available"}
            
            time_series = data[time_series_key]
            prices = []
            times = []
            
            for timestamp, values in list(time_series.items())[:50]:  # Last 50 intervals
                times.append(timestamp)
                prices.append(float(values["4. close"]))
            
            return {
                "ticker": ticker,
                "interval": interval,
                "prices": prices,
                "times": times,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_options_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch options chain data for unusual activity detection"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get available expiration dates
            expirations = stock.options
            if not expirations:
                return {"error": "No options data available"}
            
            # Get nearest expiration
            nearest_exp = expirations[0]
            opt_chain = stock.option_chain(nearest_exp)
            
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Find unusual volume
            calls_high_vol = calls[calls['volume'] > calls['volume'].mean() * 2] if len(calls) > 0 else pd.DataFrame()
            puts_high_vol = puts[puts['volume'] > puts['volume'].mean() * 2] if len(puts) > 0 else pd.DataFrame()
            
            return {
                "ticker": ticker,
                "expiration": nearest_exp,
                "total_call_volume": int(calls['volume'].sum()) if len(calls) > 0 else 0,
                "total_put_volume": int(puts['volume'].sum()) if len(puts) > 0 else 0,
                "put_call_ratio": (puts['volume'].sum() / calls['volume'].sum()) if len(calls) > 0 and calls['volume'].sum() > 0 else None,
                "unusual_calls": len(calls_high_vol),
                "unusual_puts": len(puts_high_vol),
                "implied_volatility_avg": float(calls['impliedVolatility'].mean()) if len(calls) > 0 else None,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_insider_trades(self, ticker: str) -> List[Dict]:
        """Fetch recent insider trading activity"""
        try:
            stock = yf.Ticker(ticker)
            insider_trades = stock.insider_transactions
            
            if insider_trades is None or len(insider_trades) == 0:
                return []
            
            trades = []
            for _, row in insider_trades.head(10).iterrows():
                trades.append({
                    "insider": row.get('Insider Trading', 'Unknown'),
                    "relation": row.get('Relationship', 'Unknown'),
                    "transaction": row.get('Transaction', 'Unknown'),
                    "shares": row.get('Shares', 0),
                    "value": row.get('Value', 0),
                })
            
            return trades
        except Exception as e:
            return []
    
    def get_institutional_holders(self, ticker: str) -> List[Dict]:
        """Fetch institutional ownership data"""
        try:
            stock = yf.Ticker(ticker)
            holders = stock.institutional_holders
            
            if holders is None or len(holders) == 0:
                return []
            
            result = []
            for _, row in holders.head(10).iterrows():
                result.append({
                    "holder": row.get('Holder', 'Unknown'),
                    "shares": row.get('Shares', 0),
                    "date_reported": str(row.get('Date Reported', '')),
                    "pct_out": row.get('% Out', 0),
                    "value": row.get('Value', 0),
                })
            
            return result
        except Exception as e:
            return []
    
    def get_earnings_calendar(self, ticker: str) -> Dict[str, Any]:
        """Get upcoming earnings dates and estimates"""
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is None:
                return {"error": "No earnings calendar available"}
            
            # Handle both dict and DataFrame formats
            if isinstance(calendar, pd.DataFrame):
                return {
                    "earnings_date": str(calendar.get('Earnings Date', ['Unknown'])[0]) if 'Earnings Date' in calendar else None,
                    "eps_estimate": calendar.get('EPS Estimate', [None])[0] if 'EPS Estimate' in calendar else None,
                    "revenue_estimate": calendar.get('Revenue Estimate', [None])[0] if 'Revenue Estimate' in calendar else None,
                }
            else:
                return dict(calendar)
        except Exception as e:
            return {"error": str(e)}

