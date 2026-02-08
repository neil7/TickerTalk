"""
Day Trading Scanner
Identifies best stocks for short-term and day trading opportunities
Dynamically fetches current market movers and analyzes them
"""
import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import requests


@dataclass
class TradingCandidate:
    """A potential trading candidate with analysis"""
    ticker: str
    company_name: str
    current_price: float
    change_pct: float
    volume_ratio: float
    
    # Trading signals
    signal_strength: float  # 0-100
    trade_type: str  # "long", "short", "neutral"
    
    # Price targets
    entry_price: float
    stop_loss: float
    target_1: float  # First profit target
    target_2: float  # Second profit target
    target_3: float  # Extended target
    
    # Risk metrics
    risk_reward_ratio: float
    position_size_pct: float  # Suggested position as % of portfolio
    
    # Technical indicators
    rsi: float
    above_vwap: bool
    near_support: bool
    near_resistance: bool
    
    # Catalysts
    catalysts: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "current_price": self.current_price,
            "change_pct": self.change_pct,
            "volume_ratio": self.volume_ratio,
            "signal_strength": self.signal_strength,
            "trade_type": self.trade_type,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "target_2": self.target_2,
            "target_3": self.target_3,
            "risk_reward_ratio": self.risk_reward_ratio,
            "position_size_pct": self.position_size_pct,
            "rsi": self.rsi,
            "above_vwap": self.above_vwap,
            "near_support": self.near_support,
            "near_resistance": self.near_resistance,
            "catalysts": self.catalysts,
        }


class DayTradingScanner:
    """
    Scans market for day trading and short-term opportunities
    Dynamically fetches current market movers for fresh data
    """
    
    # Base watchlist - always scan these
    BASE_WATCHLIST = [
        # Mega caps (most liquid)
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        # High beta tech
        "AMD", "INTC", "MU", "QCOM", "AVGO", "CRM", "ORCL", "SNOW", "PLTR",
        # Financials
        "JPM", "BAC", "GS", "MS", "C", "WFC", "SCHW",
        # Consumer
        "DIS", "NFLX", "SBUX", "NKE", "MCD", "COST", "WMT", "TGT",
        # Energy (volatile)
        "XOM", "CVX", "OXY", "SLB", "HAL", "DVN",
        # EV/Clean Energy
        "RIVN", "LCID", "F", "GM", "PLUG", "FCEL",
        # Biotech (high volatility)
        "MRNA", "BNTX", "PFE", "JNJ", "ABBV", "LLY",
        # Aerospace/Defense
        "BA", "LMT", "RTX", "NOC", "GD",
        # Retail
        "HD", "LOW", "AMZN", "EBAY",
        # Popular trading stocks
        "SOFI", "COIN", "HOOD", "RKLB", "SPCE", "AMC", "GME", "BBBY",
        # Semiconductors
        "SMCI", "ARM", "MRVL", "ON", "AMAT",
        # AI/Tech
        "AI", "PATH", "DDOG", "NET", "ZS",
        # China ADRs (volatile)
        "BABA", "JD", "PDD", "NIO", "XPEV", "LI",
    ]
    
    def __init__(self):
        self.candidates: List[TradingCandidate] = []
        self.dynamic_tickers: List[str] = []
    
    def _fetch_market_movers(self) -> List[str]:
        """Fetch current market movers dynamically from Yahoo Finance"""
        movers = set()
        
        try:
            # Get gainers
            gainers_url = "https://finance.yahoo.com/gainers"
            losers_url = "https://finance.yahoo.com/losers"
            active_url = "https://finance.yahoo.com/most-active"
            
            # Use yfinance screener for top movers
            # Fetch top gainers
            try:
                from bs4 import BeautifulSoup
                headers = {'User-Agent': 'Mozilla/5.0'}
                
                # Try to get gainers
                resp = requests.get(gainers_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    # Find ticker symbols in the page
                    for link in soup.find_all('a', href=True):
                        href = link.get('href', '')
                        if '/quote/' in href:
                            ticker = href.split('/quote/')[-1].split('?')[0].split('/')[0]
                            if ticker.isalpha() and len(ticker) <= 5:
                                movers.add(ticker.upper())
            except:
                pass
            
            # Alternative: Use yfinance to get trending tickers
            try:
                # Get S&P 500 top movers
                sp500 = yf.Ticker("^GSPC")
                # This doesn't directly give movers, but we can check volume
            except:
                pass
                
        except Exception as e:
            pass
        
        # If we couldn't fetch dynamic movers, return empty
        # The base watchlist will still be used
        return list(movers)[:30]  # Limit to top 30 dynamic tickers
    
    def _get_todays_movers_yf(self) -> List[str]:
        """Get today's biggest movers using yfinance"""
        movers = []
        
        # Check a broad list and find those with biggest moves
        check_list = self.BASE_WATCHLIST.copy()
        
        moves = []
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(self._quick_check, t): t for t in check_list}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    moves.append(result)
        
        # Sort by absolute change and volume
        moves.sort(key=lambda x: (abs(x[1]) * x[2]), reverse=True)
        
        # Return top movers
        return [m[0] for m in moves[:40]]
    
    def _quick_check(self, ticker: str) -> Optional[tuple]:
        """Quick check a ticker for movement"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if len(hist) < 2:
                return None
            
            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            change_pct = ((current - prev) / prev) * 100
            
            avg_vol = hist['Volume'].mean()
            curr_vol = hist['Volume'].iloc[-1]
            vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1
            
            # Only return if there's some movement or volume
            if abs(change_pct) > 0.3 or vol_ratio > 1.2:
                return (ticker, change_pct, vol_ratio)
            return None
        except:
            return None
    
    def scan(self, 
             min_volume_ratio: float = 0.8,
             min_change_pct: float = 0.0,
             max_stocks: int = 25) -> List[TradingCandidate]:
        """
        Scan for day trading opportunities
        Uses dynamic market data to find current opportunities
        """
        # Get dynamic movers first
        print("  [Fetching current market movers...]")
        dynamic_movers = self._get_todays_movers_yf()
        
        # Combine with base watchlist, prioritizing dynamic movers
        scan_list = list(dict.fromkeys(dynamic_movers + self.BASE_WATCHLIST))[:60]
        
        print(f"  [Analyzing {len(scan_list)} stocks...]")
        
        candidates = []
        
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {
                executor.submit(self._analyze_stock, ticker): ticker 
                for ticker in scan_list
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        # More lenient filtering - include more candidates
                        if result.volume_ratio >= min_volume_ratio:
                            candidates.append(result)
                except Exception as e:
                    pass
        
        # Sort by signal strength
        candidates.sort(key=lambda x: x.signal_strength, reverse=True)
        
        self.candidates = candidates[:max_stocks]
        return self.candidates
    
    def _analyze_stock(self, ticker: str) -> Optional[TradingCandidate]:
        """Analyze a single stock for day trading potential"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist = stock.history(period="1mo", interval="1d")
            info = stock.info
            
            if len(hist) < 5:
                return None
            
            # Current price data
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            change_pct = ((current_price - prev_close) / prev_close) * 100
            
            # Volume analysis
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].rolling(10).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Technical indicators
            rsi = self._calculate_rsi(hist['Close'].values)
            atr = self._calculate_atr(hist)
            atr_pct = (atr / current_price) * 100
            
            # VWAP approximation
            typical_price = (hist['High'] + hist['Low'] + hist['Close']) / 3
            vwap = (typical_price * hist['Volume']).sum() / hist['Volume'].sum()
            above_vwap = current_price > vwap
            
            # Support/Resistance
            support, resistance = self._find_support_resistance(hist['Close'].values)
            near_support = current_price <= support * 1.03
            near_resistance = current_price >= resistance * 0.97
            
            # Calculate trading signals
            signal_strength, trade_type, catalysts = self._calculate_signal(
                current_price, prev_close, change_pct, volume_ratio,
                rsi, above_vwap, near_support, near_resistance, atr_pct
            )
            
            # Calculate price targets
            entry_price, stop_loss, targets = self._calculate_price_targets(
                current_price, atr, trade_type, support, resistance
            )
            
            # Risk/reward calculation
            risk = abs(entry_price - stop_loss)
            reward = abs(targets[0] - entry_price)
            rr_ratio = reward / risk if risk > 0 else 1
            
            # Position sizing (based on 2% risk rule)
            risk_pct = (risk / current_price) * 100
            position_size = min(10.0, 2.0 / risk_pct * 100) if risk_pct > 0.5 else 5.0
            
            return TradingCandidate(
                ticker=ticker,
                company_name=info.get('longName', info.get('shortName', ticker)),
                current_price=round(current_price, 2),
                change_pct=round(change_pct, 2),
                volume_ratio=round(volume_ratio, 2),
                signal_strength=round(signal_strength, 1),
                trade_type=trade_type,
                entry_price=round(entry_price, 2),
                stop_loss=round(stop_loss, 2),
                target_1=round(targets[0], 2),
                target_2=round(targets[1], 2),
                target_3=round(targets[2], 2),
                risk_reward_ratio=round(rr_ratio, 2),
                position_size_pct=round(position_size, 1),
                rsi=round(rsi, 1),
                above_vwap=above_vwap,
                near_support=near_support,
                near_resistance=near_resistance,
                catalysts=catalysts,
            )
            
        except Exception as e:
            return None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, hist: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = hist['High'].values
        low = hist['Low'].values
        close = hist['Close'].values
        
        if len(close) < period + 1:
            return abs(close[-1] - close[-2]) if len(close) >= 2 else close[-1] * 0.02
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        return np.mean(tr[-period:])
    
    def _find_support_resistance(self, prices: np.ndarray) -> tuple:
        """Find support and resistance levels"""
        if len(prices) < 10:
            return prices[-1] * 0.97, prices[-1] * 1.03
        
        recent = prices[-20:] if len(prices) >= 20 else prices
        
        # Simple approach: recent lows for support, highs for resistance
        support = np.percentile(recent, 10)  # 10th percentile
        resistance = np.percentile(recent, 90)  # 90th percentile
        
        return support, resistance
    
    def _calculate_signal(self, current_price: float, prev_close: float,
                         change_pct: float, volume_ratio: float,
                         rsi: float, above_vwap: bool,
                         near_support: bool, near_resistance: bool,
                         atr_pct: float) -> tuple:
        """Calculate trading signal strength and type"""
        signal = 40  # Base signal
        catalysts = []
        long_score = 0
        short_score = 0
        
        # Volume signal (strong volume = strong signal)
        if volume_ratio > 2.0:
            signal += 15
            catalysts.append(f"High volume ({volume_ratio:.1f}x)")
        elif volume_ratio > 1.3:
            signal += 8
            catalysts.append(f"Above avg volume ({volume_ratio:.1f}x)")
        
        # Price momentum
        if change_pct > 2:
            signal += 10
            long_score += 2
            catalysts.append(f"Strong up move (+{change_pct:.1f}%)")
        elif change_pct > 0.5:
            signal += 5
            long_score += 1
            catalysts.append(f"Positive momentum (+{change_pct:.1f}%)")
        elif change_pct < -2:
            signal += 10
            short_score += 2
            catalysts.append(f"Strong down move ({change_pct:.1f}%)")
        elif change_pct < -0.5:
            signal += 5
            short_score += 1
            catalysts.append(f"Negative momentum ({change_pct:.1f}%)")
        
        # RSI signals
        if rsi < 30:
            signal += 12
            long_score += 3
            catalysts.append(f"RSI oversold ({rsi:.0f})")
        elif rsi < 40:
            signal += 5
            long_score += 1
            catalysts.append(f"RSI low ({rsi:.0f})")
        elif rsi > 70:
            signal += 12
            short_score += 3
            catalysts.append(f"RSI overbought ({rsi:.0f})")
        elif rsi > 60:
            signal += 5
            short_score += 1
            catalysts.append(f"RSI elevated ({rsi:.0f})")
        
        # VWAP signal
        if above_vwap:
            long_score += 1
            if change_pct > 0:
                catalysts.append("Above VWAP (bullish)")
        else:
            short_score += 1
            if change_pct < 0:
                catalysts.append("Below VWAP (bearish)")
        
        # Support/Resistance
        if near_support:
            signal += 8
            long_score += 2
            catalysts.append("Near support (bounce zone)")
        if near_resistance:
            signal += 8
            short_score += 2
            catalysts.append("Near resistance (rejection zone)")
        
        # Volatility (good for day trading)
        if atr_pct > 3:
            signal += 8
            catalysts.append(f"High volatility ({atr_pct:.1f}%)")
        elif atr_pct > 2:
            signal += 4
        
        # Determine trade type based on scores
        if long_score > short_score + 1:
            trade_type = "long"
        elif short_score > long_score + 1:
            trade_type = "short"
        elif long_score > short_score:
            trade_type = "long"
        elif short_score > long_score:
            trade_type = "short"
        else:
            # Tie-breaker: use momentum
            if change_pct > 0:
                trade_type = "long"
            elif change_pct < 0:
                trade_type = "short"
            else:
                trade_type = "neutral"
        
        signal = min(100, max(20, signal))
        
        return signal, trade_type, catalysts
    
    def _calculate_price_targets(self, current_price: float, atr: float,
                                trade_type: str, support: float, 
                                resistance: float) -> tuple:
        """Calculate entry, stop-loss, and profit targets"""
        
        # Ensure ATR is reasonable
        if atr < current_price * 0.005:
            atr = current_price * 0.02  # Minimum 2% ATR
        
        if trade_type == "long":
            # Long trade
            entry = current_price
            stop_loss = max(support * 0.99, current_price - (atr * 1.5))
            
            # Targets based on ATR
            target_1 = current_price + (atr * 1.5)
            target_2 = current_price + (atr * 2.5)
            target_3 = min(resistance * 1.02, current_price + (atr * 4))
            
        elif trade_type == "short":
            # Short trade
            entry = current_price
            stop_loss = min(resistance * 1.01, current_price + (atr * 1.5))
            
            target_1 = current_price - (atr * 1.5)
            target_2 = current_price - (atr * 2.5)
            target_3 = max(support * 0.98, current_price - (atr * 4))
            
        else:
            # Neutral - provide both scenarios, lean long
            entry = current_price
            stop_loss = current_price - (atr * 1.5)
            target_1 = current_price + (atr * 1.5)
            target_2 = current_price + (atr * 2.5)
            target_3 = current_price + (atr * 4)
        
        return entry, stop_loss, [target_1, target_2, target_3]
    
    def get_top_longs(self, n: int = 10) -> List[TradingCandidate]:
        """Get top long candidates"""
        longs = [c for c in self.candidates if c.trade_type == "long"]
        return sorted(longs, key=lambda x: x.signal_strength, reverse=True)[:n]
    
    def get_top_shorts(self, n: int = 10) -> List[TradingCandidate]:
        """Get top short candidates"""
        shorts = [c for c in self.candidates if c.trade_type == "short"]
        return sorted(shorts, key=lambda x: x.signal_strength, reverse=True)[:n]
    
    def get_best_risk_reward(self, n: int = 10) -> List[TradingCandidate]:
        """Get candidates with best risk/reward ratio"""
        return sorted(self.candidates, key=lambda x: x.risk_reward_ratio, reverse=True)[:n]
    
    def get_high_volume(self, n: int = 10) -> List[TradingCandidate]:
        """Get candidates with highest relative volume"""
        return sorted(self.candidates, key=lambda x: x.volume_ratio, reverse=True)[:n]
    
    def get_oversold_bounces(self, n: int = 10) -> List[TradingCandidate]:
        """Get oversold stocks that might bounce"""
        oversold = [c for c in self.candidates if c.rsi < 35 and c.trade_type == "long"]
        return sorted(oversold, key=lambda x: x.signal_strength, reverse=True)[:n]
    
    def get_momentum_plays(self, n: int = 10) -> List[TradingCandidate]:
        """Get stocks with strong momentum"""
        momentum = [c for c in self.candidates if abs(c.change_pct) > 1.5]
        return sorted(momentum, key=lambda x: abs(x.change_pct), reverse=True)[:n]
