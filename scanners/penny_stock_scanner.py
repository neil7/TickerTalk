"""
Penny Stock Scanner
Identifies high-potential penny stocks that haven't spiked yet but have 100%+ upside potential.
Focuses on:
1. Low Price (<$10)
2. Oversold/Consolidation phases ("has not increased yet")
3. High Volatility (potential to move fast)
4. Accumulation signals
"""
import yfinance as yf
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

@dataclass
class PennyCandidate:
    """A potential penny stock candidate"""
    ticker: str
    company_name: str
    current_price: float
    target_price: float  # 100% gain target
    potential_pct: float
    
    # Technicals
    rsi: float
    volume_ratio: float
    distance_from_low: float  # % above 52w low
    volatility: float  # ATR%
    
    # Analysis
    reason: str
    risk_level: str  # High/Extreme
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "price": self.current_price,
            "target": self.target_price,
            "potential": self.potential_pct,
            "rsi": self.rsi,
            "volume_ratio": self.volume_ratio,
            "from_low": self.distance_from_low,
            "volatility": self.volatility,
            "reason": self.reason,
            "risk": self.risk_level
        }

class PennyStockScanner:
    """Scans for explosive penny stock opportunities"""
    
    # Base Watchlist - Fallback if dynamic scanning misses specific sectors
    # Includes Biotech, EV, Crypto, Tech, Energy
    PENNY_WATCHLIST = [
        # Biotech/Pharma (Explosive potential)
        "BNGO", "SENS", "TNXP", "JAGX", "ZOM", "OCGN", "IBIO", "ATOS", "CTXR", "ASRT",
        "TRVN", "XBI", "LABU", "NVAX", "INO", "VXRT", "SRNE", "AGEN", "ZIOP", "CRSP",
        
        # EV / Clean Energy
        "MULN", "NKLA", "GOEV", "FSR", "WKHS", "RIDE", "HYLN", "XL", "SOLO", "IDEX",
        "PLUG", "FCEL", "BE", "BLDP", "CLNE", "GEVO", "AMRS", "SUNW", "SPI", "CBAT",
        
        # Tech / AI / Growth
        "SOUN", "BBAI", "MARK", "WISH", "CLOV", "PLTR", "SOFI", "OPEN", "SDC", "TTCF",
        "BB", "NOK", "ERIC", "KOSS", "EXPR", "GPRO", "DBI", "REAL", "POSH", "JMIA",
        
        # Crypto / Blockchain
        "MARA", "RIOT", "HUT", "HIVE", "BITF", "BTBT", "MSTR", "COIN", "SI", "BKKT",
        "SOS", "EBON", "CAN", "MIGI", "WULF", "IREN", "CORZ", "SDIG", "GREE", "ANY",
        
        # Cannabis
        "SNDL", "TLRY", "ACB", "CGC", "CRON", "OGI", "HEXO", "VFF", "GRWG", "IIPR",
        
        # Others / Penny Favorites
        "AMC", "GME", "BBBY", "CEI", "PED", "IMPP", "INDO", "HUSA", "ENSV", "NINE"
    ]

    def __init__(self):
        self.candidates: List[PennyCandidate] = []

    def _fetch_dynamic_tickers(self) -> List[str]:
        """Fetch potential penny stocks from Yahoo Finance market movers"""
        tickers = set()
        urls = [
            "https://finance.yahoo.com/most-active",
            "https://finance.yahoo.com/losers", # Good for finding oversold
            "https://finance.yahoo.com/gainers" # Good for finding momentum
        ]
        
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        print("  [Fetching dynamic market data (Most Active, Losers, Gainers)...]")
        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    # Find ticker symbols in the page
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        if '/quote/' in href:
                            parts = href.split('/quote/')[-1].split('?')[0].split('/')
                            if parts:
                                ticker = parts[0]
                                if ticker.isalpha() and len(ticker) <= 5:
                                    tickers.add(ticker.upper())
            except Exception:
                pass
                
        return list(tickers)

    def scan(self, max_price: float = 10.0, min_potential: float = 100.0) -> List[PennyCandidate]:
        """
        Scan for stocks with 100%+ potential
        Criteria:
        - Price < max_price (User requested < $10)
        - RSI < 50 (Not overbought yet)
        - Near 52-week low (Bottoming)
        - High Volatility (Capable of moving fast)
        """
        # 1. Get Dynamic Tickers
        dynamic_tickers = self._fetch_dynamic_tickers()
        print(f"  [Found {len(dynamic_tickers)} active stocks from market scan]")
        
        # 2. Combine with Watchlist (prioritize dynamic ones first in processing order isn't strict)
        # We use a set to remove duplicates
        all_tickers = list(set(self.PENNY_WATCHLIST + dynamic_tickers))
        
        print(f"  [Scanning {len(all_tickers)} total stocks for opportunities < ${max_price} ...]")
        
        candidates = []
        with ThreadPoolExecutor(max_workers=20) as executor: # Increased workers for more stocks
            futures = {
                executor.submit(self._analyze_stock, ticker, max_price): ticker 
                for ticker in all_tickers
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        candidates.append(result)
                except Exception:
                    pass
        
        # Sort by "Explosiveness" (Volatility + Low RSI combo)
        # We want low RSI (room to run) and High Volatility (speed)
        candidates.sort(key=lambda x: (x.volatility / x.rsi if x.rsi > 0 else 0), reverse=True)
        
        self.candidates = candidates
        return candidates

    def _analyze_stock(self, ticker: str, max_price: float) -> Optional[PennyCandidate]:
        try:
            stock = yf.Ticker(ticker)
            
            # Fast check price first (optimization)
            # Use 'fast_info' if available or try to fetch minimal history first
            try:
                # fast_info is faster than history
                price = stock.fast_info.last_price
                if price > max_price or price == 0:
                    return None
            except:
                pass # Fallback to history
            
            # Need history for volatility
            hist = stock.history(period="3mo")
            if len(hist) < 20:
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            # 1. Filter: Must be under max_price ($10)
            if current_price > max_price:
                return None
                
            # 2. Check "Has not increased yet" -> RSI & Price vs High
            rsi = self._calculate_rsi(hist['Close'].values)
            if rsi > 60: # Already moving up? Skip if we want "haven't increased yet"
                return None
                
            # 3. Check Volatility (Can it double?)
            # Calculate ATR percentage
            atr = self._calculate_atr(hist)
            volatility_pct = (atr / current_price) * 100
            
            # If low volatility (< 3%), unlikely to double in a month
            if volatility_pct < 3.0: 
                return None
            
            info = stock.info
            
            # 4. Check Position in Range
            year_low = info.get('fiftyTwoWeekLow', hist['Low'].min())
            year_high = info.get('fiftyTwoWeekHigh', hist['High'].max())
            
            # Avoid division by zero
            if year_low == 0 or np.isnan(year_low): year_low = hist['Low'].min()
            if year_high == 0 or np.isnan(year_high): year_high = hist['High'].max()
            
            from_low_pct = ((current_price - year_low) / year_low) * 100
            
            # Must be closer to bottom than top (Upside room)
            if from_low_pct > 60: # Allow a bit more room since we scan up to $10
                 if rsi > 55: return None
            
            # Volume Check - Accumulation?
            avg_vol = hist['Volume'].mean()
            curr_vol = hist['Volume'].iloc[-1]
            vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 0
            
            # Reason generation
            reasons = []
            if rsi < 35: reasons.append("Deeply Oversold")
            elif rsi < 45: reasons.append("Oversold/Bottom")
            
            if volatility_pct > 8: reasons.append("Extreme Volatility")
            elif volatility_pct > 5: reasons.append("High Volatility")
            
            if vol_ratio > 1.5: reasons.append("Volume Spiking (Accumulation)")
            
            if from_low_pct < 15: reasons.append("Near 52-Week Low")
            
            potential = 100.0 # Goal is double
            target = current_price * 2.0
            
            return PennyCandidate(
                ticker=ticker,
                company_name=info.get('shortName', ticker),
                current_price=round(current_price, 2),
                target_price=round(target, 2),
                potential_pct=potential,
                rsi=round(rsi, 1),
                volume_ratio=round(vol_ratio, 1),
                distance_from_low=round(from_low_pct, 1),
                volatility=round(volatility_pct, 1),
                reason=", ".join(reasons),
                risk_level="EXTREME" if volatility_pct > 10 else "HIGH"
            )
            
        except Exception:
            return None

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1: return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, hist: pd.DataFrame, period: int = 14) -> float:
        high = hist['High'].values
        low = hist['Low'].values
        close = hist['Close'].values
        if len(close) < period + 1: return close[-1] * 0.05
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-period:])
