"""
Market Movers Fetcher
Identifies top gaining and losing stocks for potential opportunities
"""
import yfinance as yf
import pandas as pd
import requests
from typing import Dict, List, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import BLUE_CHIP_TICKERS, SECTOR_ETFS, analysis_config


class MarketMoversFetcher:
    """Identifies and analyzes top market movers"""
    
    def __init__(self):
        self.blue_chips = BLUE_CHIP_TICKERS
        self.sector_etfs = SECTOR_ETFS
    
    def get_sp500_tickers(self) -> List[str]:
        """Fetch current S&P 500 component tickers"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            tickers = tables[0]['Symbol'].tolist()
            # Clean tickers (replace . with -)
            tickers = [t.replace('.', '-') for t in tickers]
            return tickers
        except Exception as e:
            # Fallback to predefined blue chips
            return self.blue_chips
    
    def _get_stock_change(self, ticker: str) -> Dict[str, Any]:
        """Get daily change for a single stock"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            
            if len(hist) < 2:
                return None
            
            current = hist['Close'].iloc[-1]
            previous = hist['Close'].iloc[-2]
            change_pct = ((current - previous) / previous) * 100
            volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            
            info = stock.info
            
            return {
                "ticker": ticker,
                "company_name": info.get('longName', info.get('shortName', ticker)),
                "current_price": current,
                "previous_close": previous,
                "change_pct": round(change_pct, 2),
                "volume": int(volume),
                "avg_volume": int(avg_volume),
                "volume_ratio": round(volume_ratio, 2),
                "market_cap": info.get('marketCap'),
                "sector": info.get('sector', 'Unknown'),
            }
        except Exception as e:
            return None
    
    def get_daily_movers(self, num_stocks: int = 50) -> Dict[str, Any]:
        """
        Get top daily gainers and losers
        Uses parallel processing for speed
        """
        # Get tickers to scan
        tickers = self.get_sp500_tickers()[:num_stocks]
        
        movers = []
        
        # Parallel fetch
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self._get_stock_change, ticker): ticker for ticker in tickers}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    movers.append(result)
        
        # Sort by change percentage
        gainers = sorted([m for m in movers if m['change_pct'] > 0], 
                        key=lambda x: x['change_pct'], reverse=True)[:10]
        losers = sorted([m for m in movers if m['change_pct'] < 0], 
                       key=lambda x: x['change_pct'])[:10]
        
        # Volume leaders (stocks with unusual volume)
        volume_leaders = sorted([m for m in movers if m['volume_ratio'] > 1.5], 
                               key=lambda x: x['volume_ratio'], reverse=True)[:10]
        
        return {
            "top_gainers": gainers,
            "top_losers": losers,
            "volume_leaders": volume_leaders,
            "total_scanned": len(movers),
            "fetch_time": datetime.now().isoformat(),
        }
    
    def get_sector_performance(self) -> Dict[str, Any]:
        """Get performance of major sector ETFs"""
        sectors = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self._get_stock_change, etf): name 
                      for etf, name in self.sector_etfs.items()}
            
            for future in as_completed(futures):
                sector_name = futures[future]
                result = future.result()
                if result:
                    sectors[sector_name] = {
                        "etf": result['ticker'],
                        "change_pct": result['change_pct'],
                        "volume_ratio": result['volume_ratio'],
                    }
        
        # Sort sectors by performance
        sorted_sectors = dict(sorted(sectors.items(), 
                                    key=lambda x: x[1]['change_pct'], 
                                    reverse=True))
        
        return {
            "sectors": sorted_sectors,
            "top_sector": list(sorted_sectors.keys())[0] if sorted_sectors else None,
            "worst_sector": list(sorted_sectors.keys())[-1] if sorted_sectors else None,
            "fetch_time": datetime.now().isoformat(),
        }
    
    def get_premarket_movers(self) -> Dict[str, Any]:
        """
        Get pre-market movers (when market is closed)
        Limited data available through free APIs
        """
        try:
            movers = []
            
            # Check a sample of blue chips for pre-market activity
            for ticker in self.blue_chips[:20]:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    pre_price = info.get('preMarketPrice')
                    prev_close = info.get('previousClose')
                    
                    if pre_price and prev_close:
                        change_pct = ((pre_price - prev_close) / prev_close) * 100
                        movers.append({
                            "ticker": ticker,
                            "pre_market_price": pre_price,
                            "previous_close": prev_close,
                            "change_pct": round(change_pct, 2),
                        })
                except:
                    continue
            
            gainers = sorted([m for m in movers if m['change_pct'] > 0], 
                            key=lambda x: x['change_pct'], reverse=True)[:5]
            losers = sorted([m for m in movers if m['change_pct'] < 0], 
                           key=lambda x: x['change_pct'])[:5]
            
            return {
                "pre_market_gainers": gainers,
                "pre_market_losers": losers,
                "fetch_time": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_52_week_signals(self) -> Dict[str, Any]:
        """
        Find stocks near 52-week highs/lows
        Potential breakout or breakdown candidates
        """
        signals = {
            "near_52w_high": [],
            "near_52w_low": [],
        }
        
        for ticker in self.blue_chips:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                current = info.get('currentPrice') or info.get('regularMarketPrice')
                high_52w = info.get('fiftyTwoWeekHigh')
                low_52w = info.get('fiftyTwoWeekLow')
                
                if current and high_52w and low_52w:
                    pct_from_high = ((high_52w - current) / high_52w) * 100
                    pct_from_low = ((current - low_52w) / low_52w) * 100
                    
                    stock_data = {
                        "ticker": ticker,
                        "current": current,
                        "52w_high": high_52w,
                        "52w_low": low_52w,
                    }
                    
                    # Within 5% of 52-week high
                    if pct_from_high < 5:
                        stock_data["pct_from_high"] = round(pct_from_high, 2)
                        signals["near_52w_high"].append(stock_data)
                    
                    # Within 10% of 52-week low
                    if pct_from_low < 10:
                        stock_data["pct_from_low"] = round(pct_from_low, 2)
                        signals["near_52w_low"].append(stock_data)
            except:
                continue
        
        signals["fetch_time"] = datetime.now().isoformat()
        return signals
    
    def identify_momentum_candidates(self) -> Dict[str, Any]:
        """
        Identify stocks with strong momentum characteristics
        Combines price action, volume, and technical signals
        """
        candidates = []
        
        for ticker in self.blue_chips:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="3mo")
                info = stock.info
                
                if len(hist) < 50:
                    continue
                
                current = hist['Close'].iloc[-1]
                sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
                sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
                avg_volume = hist['Volume'].mean()
                recent_volume = hist['Volume'].iloc[-5:].mean()
                
                # Momentum criteria
                above_20_sma = current > sma_20
                above_50_sma = current > sma_50
                sma_20_above_50 = sma_20 > sma_50  # Golden cross condition
                volume_increasing = recent_volume > avg_volume * 1.2
                
                # Calculate RSI
                delta = hist['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs))).iloc[-1]
                
                momentum_score = sum([
                    above_20_sma,
                    above_50_sma,
                    sma_20_above_50,
                    volume_increasing,
                    30 < rsi < 70,  # Not overbought/oversold
                ])
                
                if momentum_score >= 4:
                    candidates.append({
                        "ticker": ticker,
                        "company_name": info.get('longName', ticker),
                        "current_price": current,
                        "momentum_score": momentum_score,
                        "above_20_sma": above_20_sma,
                        "above_50_sma": above_50_sma,
                        "golden_cross": sma_20_above_50,
                        "volume_increasing": volume_increasing,
                        "rsi": round(rsi, 2),
                    })
            except:
                continue
        
        # Sort by momentum score
        candidates.sort(key=lambda x: x['momentum_score'], reverse=True)
        
        return {
            "momentum_candidates": candidates[:10],
            "total_found": len(candidates),
            "fetch_time": datetime.now().isoformat(),
        }
    
    def get_comprehensive_market_scan(self) -> Dict[str, Any]:
        """
        Comprehensive market scan combining all signals
        Best candidates for AI analysis
        """
        return {
            "daily_movers": self.get_daily_movers(),
            "sector_performance": self.get_sector_performance(),
            "momentum_candidates": self.identify_momentum_candidates(),
            "52_week_signals": self.get_52_week_signals(),
            "fetch_time": datetime.now().isoformat(),
        }

