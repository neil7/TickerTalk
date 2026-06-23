#!/usr/bin/env python3
"""
TickTalker - Agentic AI Stock Analysis System
Powered by Ollama + LangGraph

A multi-agent system for analyzing stocks and identifying potential top movers.
"""
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from workflow import create_workflow
from data_sources import MarketMoversFetcher, RedditDataFetcher
from agents import RedditSentimentAgent
from scanners import DayTradingScanner, PennyStockScanner
from config import BLUE_CHIP_TICKERS
from llm_config import print_model_recommendations

console = Console()


def print_banner():
    """Print the application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║  🤖 TICKTALKER - AGENTIC AI STOCK ANALYSIS                   ║
    ║     Powered by Ollama/Gemini + LangGraph                     ║
    ║                                                              ║
    ║  Multi-agent system for identifying top movers               ║
    ║  Data Sources: Market Data, News, FRED, Social Media         ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold cyan"))


def display_analysis_result(result: dict):
    """Display the analysis result in a formatted way"""
    ticker = result.get("ticker", "UNKNOWN")
    
    # Market Data Summary
    market_data = result.get("market_data", {})
    if market_data and "error" not in market_data:
        table = Table(title=f"📊 Market Data: {ticker}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Company", market_data.get("company_name", "N/A"))
        table.add_row("Current Price", f"${market_data.get('current_price', 'N/A')}")
        table.add_row("Change", f"{market_data.get('change_percent', 0):.2f}%")
        table.add_row("Volume", f"{market_data.get('volume', 0):,}")
        table.add_row("Market Cap", f"${market_data.get('market_cap', 0):,.0f}" if market_data.get('market_cap') else "N/A")
        table.add_row("P/E Ratio", str(market_data.get('pe_ratio', 'N/A')))
        table.add_row("52W High", f"${market_data.get('52_week_high', 'N/A')}")
        table.add_row("52W Low", f"${market_data.get('52_week_low', 'N/A')}")
        
        console.print(table)
        console.print()
    
    # Technical Analysis
    tech = result.get("technical_analysis", {})
    if tech and "error" not in tech:
        signals = tech.get("signals", {})
        indicators = tech.get("indicators", {})
        
        table = Table(title="📈 Technical Analysis")
        table.add_column("Indicator", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Signal", style="yellow")
        
        table.add_row("RSI (14)", f"{indicators.get('rsi_14', 0):.1f}", signals.get("rsi", "N/A"))
        table.add_row("MACD", f"{indicators.get('macd', 0):.2f}", signals.get("macd", "N/A"))
        table.add_row("Trend", "-", signals.get("trend", "N/A"))
        table.add_row("Bollinger", "-", signals.get("bollinger", "N/A"))
        table.add_row("Volume", f"{indicators.get('volume_ratio', 1):.2f}x avg", signals.get("volume", "N/A"))
        table.add_row("Overall", "-", f"[bold]{signals.get('overall', 'N/A')}[/bold]")
        
        console.print(table)
        
        if tech.get("interpretation"):
            console.print(Panel(tech["interpretation"], title="Technical Interpretation", border_style="blue"))
        console.print()
    
    # Reddit Analysis
    reddit = result.get("reddit_analysis", {})
    if reddit and reddit.get("available"):
        analysis = reddit.get("analysis", {})
        
        table = Table(title="🔴 Reddit Sentiment")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        sentiment = analysis.get("retail_sentiment", "N/A")
        sent_color = "green" if sentiment == "bullish" else "red" if sentiment == "bearish" else "yellow"
        
        table.add_row("Retail Sentiment", f"[{sent_color}]{sentiment.upper()}[/{sent_color}]")
        table.add_row("Total Mentions", str(analysis.get("total_mentions", 0)))
        table.add_row("Sentiment Score", f"{analysis.get('sentiment_score', 0):+.2f}")
        table.add_row("Retail Interest", analysis.get("retail_interest", "N/A"))
        table.add_row("Momentum", analysis.get("momentum_indicator", "N/A"))
        
        if analysis.get("wsb_trending"):
            table.add_row("WSB Trending", f"[bold yellow]Yes (Rank #{analysis.get('wsb_rank', 'N/A')})[/bold yellow]")
        else:
            table.add_row("WSB Trending", "No")
        
        console.print(table)
        
        # Show interpretation
        if reddit.get("interpretation"):
            console.print(Panel(reddit["interpretation"], title="Reddit Analysis", border_style="red"))
        console.print()
    elif reddit and not reddit.get("available"):
        console.print(Panel(
            "[yellow]Reddit API not configured. Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env[/yellow]",
            title="🔴 Reddit Sentiment",
            border_style="yellow"
        ))
        console.print()
    
    # Risk Assessment
    risk = result.get("risk_assessment", {})
    if risk and "error" not in risk:
        risk_level = risk.get("risk_level", {})
        risk_metrics = risk.get("risk_metrics", {})
        position = risk.get("position_sizing", {})
        stop_loss = risk.get("stop_loss", {})
        
        table = Table(title="⚠️  Risk Assessment")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        level = risk_level.get("level", "N/A")
        level_color = "red" if level in ["high", "very_high"] else "yellow" if level == "moderate" else "green"
        table.add_row("Risk Level", f"[{level_color}]{level.upper()}[/{level_color}]")
        table.add_row("Volatility (Annual)", f"{risk_metrics.get('annualized_volatility', 0):.1f}%")
        table.add_row("Max Drawdown", f"{abs(risk_metrics.get('max_drawdown', 0)):.1f}%")
        table.add_row("Daily VaR (95%)", f"{abs(risk_metrics.get('var_95_daily', 0)):.1f}%")
        table.add_row("Recommended Position", f"{position.get('recommended_pct', 'N/A')}% of portfolio")
        
        console.print(table)
        console.print()
    
    # Final Recommendation
    final = result.get("final_recommendation", {})
    if final:
        action = final.get("action", {})
        scores = final.get("scores", {})
        price_targets = final.get("price_targets", {})
        recommendation = final.get("recommendation", "")
        
        # Action summary
        action_rec = action.get("recommendation", "HOLD")
        conviction = action.get("conviction", 5)
        
        action_color = "green" if "BUY" in action_rec else "red" if "SELL" in action_rec else "yellow"
        
        console.print(Panel.fit(
            f"[bold {action_color}]{action_rec}[/bold {action_color}] | Conviction: {conviction}/10 | Horizon: {action.get('time_horizon', 'N/A')}",
            title="🎯 RECOMMENDATION",
            border_style=action_color
        ))
        
        # Price Targets Table
        if price_targets and "error" not in price_targets:
            current  = price_targets.get('current_price', 0)
            entry    = price_targets.get('entry_ideal', 0)
            target_1 = price_targets.get('target_1', 0)
            target_2 = price_targets.get('target_2', 0)
            target_3 = price_targets.get('target_3', 0)
            stop     = price_targets.get('stop_loss', 0)
            trade_type = price_targets.get('trade_type', "LONG")
            t1_pct = price_targets.get('target_1_pct', 0)
            t2_pct = price_targets.get('target_2_pct', 0)
            t3_pct = price_targets.get('target_3_pct', 0)
            rr     = price_targets.get('risk_reward_ratio', 0)

            if trade_type == "LONG":
                buy_label  = "BUY AT"
                sell_label = "SELL AT"
                buy_color  = "green"
                sell_color = "cyan"
                summary_text = (
                    f"[bold green]BUY AT:  ${entry:.2f}[/bold green]   "
                    f"(current ${current:.2f})\n\n"
                    f"[bold cyan]SELL AT: ${target_1:.2f}[/bold cyan]   "
                    f"(+{t1_pct:.1f}% quick profit)\n"
                    f"[cyan]         ${target_2:.2f}[/cyan]   "
                    f"(+{t2_pct:.1f}% mid target)\n"
                    f"[cyan]         ${target_3:.2f}[/cyan]   "
                    f"(+{t3_pct:.1f}% full target)"
                )
                border = "green"
                title  = "📈 TRADE LEVELS — BUY LOW, SELL HIGH"

            console.print(Panel(summary_text, title=title, border_style=border))
            console.print()

            # Compact details table
            table = Table(title="📊 All Price Levels")
            table.add_column("Level",  style="cyan",  width=22)
            table.add_column("Price",  style="white", width=12)
            table.add_column("Notes",  style="dim",   width=28)

            table.add_row("[bold]Current Price[/bold]", f"[bold]${current:.2f}[/bold]", "Live market")
            table.add_row("", "", "")
            table.add_row("[green]🛒 Buy — ideal[/green]",  f"[green]${entry:.2f}[/green]", "Wait for dip to this")
            table.add_row("[green]🛒 Buy — now[/green]",    f"[green]${current:.2f}[/green]", "Market order, enter now")
            table.add_row("", "", "")
            tgt_verb = "Sell"
            table.add_row(f"[cyan]💰 {tgt_verb} — target 1[/cyan]", f"[cyan]${target_1:.2f}[/cyan]", f"+{t1_pct:.1f}% — take 50% profit")
            table.add_row(f"[cyan]💰 {tgt_verb} — target 2[/cyan]", f"[cyan]${target_2:.2f}[/cyan]", f"+{t2_pct:.1f}% — take 30% profit")
            table.add_row(f"[cyan]💰 {tgt_verb} — target 3[/cyan]", f"[cyan]${target_3:.2f}[/cyan]", f"+{t3_pct:.1f}% — hold remainder")
            table.add_row("", "", "")
            table.add_row("[yellow]Support[/yellow]",     f"${price_targets.get('support', 0):.2f}", "Strong floor")
            table.add_row("[yellow]Resistance[/yellow]",  f"${price_targets.get('resistance', 0):.2f}", "May stall here")
            table.add_row("[bold]Position size[/bold]",   f"[bold]{price_targets.get('suggested_position_pct', 0):.1f}%[/bold]", "of total portfolio")

            console.print(table)
            console.print()
        
        # Scores breakdown
        table = Table(title="📊 Analysis Scores")
        table.add_column("Analysis", style="cyan")
        table.add_column("Score", style="white")
        
        table.add_row("Technical", f"{scores.get('technical_score', 0):+.1f}")
        table.add_row("Sentiment", f"{scores.get('sentiment_score', 0):+.1f}")
        table.add_row("Fundamental", f"{scores.get('fundamental_score', 0):+.1f}")
        table.add_row("Risk Adjustment", f"{scores.get('risk_adjustment', 0):+.1f}")
        table.add_row("[bold]Composite[/bold]", f"[bold]{scores.get('composite_score', 0):+.2f}[/bold]")
        
        console.print(table)
        console.print()
        
        # Full recommendation text
        if recommendation:
            console.print(Panel(
                Markdown(recommendation),
                title="📋 DETAILED RECOMMENDATION",
                border_style="cyan"
            ))


def display_market_movers(movers: dict):
    """Display market movers in a formatted table"""
    daily = movers.get("daily_movers", {})
    
    # Top Gainers
    gainers = daily.get("top_gainers", [])
    if gainers:
        table = Table(title="📈 TOP GAINERS")
        table.add_column("Ticker", style="cyan")
        table.add_column("Change", style="green")
        table.add_column("Volume", style="white")
        table.add_column("Sector", style="white")
        
        for stock in gainers[:5]:
            table.add_row(
                stock.get("ticker", "N/A"),
                f"+{stock.get('change_pct', 0):.2f}%",
                f"{stock.get('volume_ratio', 1):.1f}x avg",
                stock.get("sector", "N/A")[:15]
            )
        
        console.print(table)
        console.print()
    
    # Top Losers
    losers = daily.get("top_losers", [])
    if losers:
        table = Table(title="📉 TOP LOSERS")
        table.add_column("Ticker", style="cyan")
        table.add_column("Change", style="red")
        table.add_column("Volume", style="white")
        table.add_column("Sector", style="white")
        
        for stock in losers[:5]:
            table.add_row(
                stock.get("ticker", "N/A"),
                f"{stock.get('change_pct', 0):.2f}%",
                f"{stock.get('volume_ratio', 1):.1f}x avg",
                stock.get("sector", "N/A")[:15]
            )
        
        console.print(table)
        console.print()
    
    # Momentum Candidates
    momentum = movers.get("momentum_candidates", {})
    candidates = momentum.get("momentum_candidates", [])
    if candidates:
        table = Table(title="🚀 MOMENTUM CANDIDATES")
        table.add_column("Ticker", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("RSI", style="white")
        table.add_column("Above SMAs", style="white")
        
        for stock in candidates[:5]:
            table.add_row(
                stock.get("ticker", "N/A"),
                f"{stock.get('momentum_score', 0)}/5",
                f"{stock.get('rsi', 50):.0f}",
                "✓" if stock.get("above_50_sma") else "✗"
            )
        
        console.print(table)


def analyze_stock(ticker: str):
    """Run full analysis on a stock"""
    workflow = create_workflow()
    result = workflow.analyze(ticker)
    console.print()
    display_analysis_result(result)


def scan_market():
    """Scan market for opportunities"""
    workflow = create_workflow()
    movers = workflow.scan_market_movers()
    console.print()
    display_market_movers(movers)
    
    # Ask if user wants to analyze top gainer
    console.print()
    gainers = movers.get("daily_movers", {}).get("top_gainers", [])
    if gainers:
        top_ticker = gainers[0].get("ticker")
        console.print(f"[yellow]Top gainer: {top_ticker}[/yellow]")
        response = console.input("[cyan]Analyze this stock? (y/n): [/cyan]")
        if response.lower() == 'y':
            analyze_stock(top_ticker)


def quick_scan(tickers: list):
    """Quick scan multiple stocks"""
    workflow = create_workflow()
    results = workflow.quick_scan(tickers)
    
    table = Table(title="🔍 QUICK SCAN RESULTS")
    table.add_column("Ticker", style="cyan")
    table.add_column("Price", style="white")
    table.add_column("Change", style="white")
    table.add_column("Assessment", style="white", width=50)
    
    for result in results:
        if "error" not in result:
            change = result.get("change_pct", 0)
            change_str = f"[green]+{change:.2f}%[/green]" if change >= 0 else f"[red]{change:.2f}%[/red]"
            
            table.add_row(
                result.get("ticker", "N/A"),
                f"${result.get('price', 'N/A')}",
                change_str,
                result.get("assessment", "N/A")[:50]
            )
    
    console.print(table)


def analyze_event(event: str, tickers: list):
    """Analyze event impact on stocks"""
    workflow = create_workflow()
    results = workflow.analyze_event_impact(event, tickers)
    
    console.print(Panel(f"[bold]Event:[/bold] {event}", title="📰 Event Impact Analysis", border_style="cyan"))
    
    for ticker, impact in results.items():
        console.print(Panel(impact, title=f"Impact on {ticker}", border_style="yellow"))


def scan_day_trading():
    """Scan for day trading opportunities with buy/sell levels"""
    console.print(Panel.fit(
        "[bold green]📈 DAY TRADING SCANNER[/bold green]\n"
        "Finding best setups for short-term trading...\n"
        "[dim]Fetching live market data...[/dim]",
        border_style="green"
    ))
    
    scanner = DayTradingScanner()
    
    candidates = scanner.scan(min_volume_ratio=0.8, min_change_pct=0.0, max_stocks=30)
    
    if not candidates:
        console.print("[yellow]No setups found. Market may be closed or quiet.[/yellow]")
        return
    
    console.print(f"[green]✓ Found {len(candidates)} potential setups[/green]\n")
    
    # Display LONG setups (BUY opportunities)
    longs = scanner.get_top_longs(15)  # Show more since we're only showing longs
    if longs:
        table = Table(title="💰 TOP BUYING OPPORTUNITIES")
        table.add_column("Ticker", style="cyan", width=8)
        table.add_column("Price", style="white", width=10)
        table.add_column("Change", style="white", width=8)
        table.add_column("Signal", style="green", width=8)
        table.add_column("BUY", style="green", width=10)
        table.add_column("SELL Target", style="cyan", width=12)
        table.add_column("Stop Loss", style="red", width=10)
        table.add_column("Profit %", style="yellow", width=10)
        table.add_column("Catalysts", style="white", width=30)
        
        for c in longs:
            change_color = "green" if c.change_pct >= 0 else "red"
            profit_pct = ((c.target_1 - c.entry_price) / c.entry_price) * 100
            table.add_row(
                c.ticker,
                f"${c.current_price:.2f}",
                f"[{change_color}]{c.change_pct:+.1f}%[/{change_color}]",
                f"{c.signal_strength:.0f}%",
                f"${c.entry_price:.2f}",
                f"${c.target_1:.2f}",
                f"${c.stop_loss:.2f}",
                f"+{profit_pct:.1f}%",
                ", ".join(c.catalysts[:2])[:30]
            )
        
        console.print(table)
        console.print()
    else:
        console.print("[yellow]No strong setups found right now.[/yellow]\n")
    
    
    # Oversold Bounces (potential reversals - also LONG trades)
    bounces = scanner.get_oversold_bounces(5)
    if bounces:
        table = Table(title="📉➡️📈 OVERSOLD BOUNCE CANDIDATES (RSI < 35)")
        table.add_column("Ticker", style="cyan")
        table.add_column("Price", style="white")
        table.add_column("RSI", style="yellow")
        table.add_column("Entry", style="green")
        table.add_column("Target", style="cyan")
        table.add_column("Catalyst", style="white", width=40)
        
        for c in bounces:
            table.add_row(
                c.ticker,
                f"${c.current_price:.2f}",
                f"{c.rsi:.0f}",
                f"${c.entry_price:.2f}",
                f"${c.target_1:.2f}",
                ", ".join(c.catalysts[:2])[:40]
            )
        
        console.print(table)
        console.print()
    
    # High Volume Movers
    high_vol = scanner.get_high_volume(5)
    if high_vol:
        table = Table(title="📊 HIGH VOLUME MOVERS (Unusual Activity - BUY Opportunities)")
        table.add_column("Ticker", style="cyan")
        table.add_column("Price", style="white")
        table.add_column("Change", style="white")
        table.add_column("Volume", style="yellow")
        table.add_column("Signal", style="white")
        
        for c in high_vol:
            # Only show LONG trades
            if c.trade_type.upper() != "LONG":
                continue
            change_color = "green" if c.change_pct >= 0 else "red"
            table.add_row(
                c.ticker,
                f"${c.current_price:.2f}",
                f"[{change_color}]{c.change_pct:+.1f}%[/{change_color}]",
                f"[bold]{c.volume_ratio:.1f}x[/bold] avg",
                f"{c.signal_strength:.0f}%"
            )
        
        console.print(table)
        console.print()
    
    # Best Risk/Reward Summary
    best_rr = scanner.get_best_risk_reward(5)
    if best_rr:
        table = Table(title="⭐ BEST RISK/REWARD SETUPS (Top 5 - BUY Opportunities)")
        table.add_column("Ticker", style="cyan")
        table.add_column("BUY", style="green")
        table.add_column("Stop Loss", style="red")
        table.add_column("SELL T1", style="cyan")
        table.add_column("SELL T2", style="cyan")
        table.add_column("SELL T3", style="cyan")
        table.add_column("R:R", style="yellow")
        
        for c in best_rr:
            # Only show LONG trades
            if c.trade_type.upper() != "LONG":
                continue
            table.add_row(
                c.ticker,
                f"${c.entry_price:.2f}",
                f"${c.stop_loss:.2f}",
                f"${c.target_1:.2f}",
                f"${c.target_2:.2f}",
                f"${c.target_3:.2f}",
                f"[bold yellow]{c.risk_reward_ratio:.1f}:1[/bold yellow]"
            )
        
        console.print(table)
        console.print()
    
    # Ask for detailed analysis
    if candidates:
        # Show top 3 picks summary
        console.print("[bold cyan]🎯 TOP 3 PICKS (BUY These):[/bold cyan]")
        long_picks = [c for c in candidates if c.trade_type.upper() == "LONG"][:3]
        for i, c in enumerate(long_picks, 1):
            profit_pct = ((c.target_1 - c.entry_price) / c.entry_price) * 100
            console.print(f"  {i}. [green]{c.ticker}[/green] - BUY @ ${c.entry_price:.2f} → SELL @ ${c.target_1:.2f} (+{profit_pct:.1f}%) [Signal: {c.signal_strength:.0f}%]")
        
        console.print()
        response = console.input("[cyan]Enter ticker for full analysis (or press Enter to skip): [/cyan]").strip().upper()
        if response and response in [c.ticker for c in candidates]:
            analyze_stock(response)
        elif response:
            analyze_stock(response)


def scan_wsb_trending():
    """Scan WallStreetBets for trending tickers"""
    console.print(Panel.fit(
        "[bold red]🔴 WALLSTREETBETS TRENDING SCAN[/bold red]",
        border_style="red"
    ))
    
    reddit_agent = RedditSentimentAgent()
    
    if not reddit_agent.reddit_fetcher.is_available():
        console.print("[yellow]Reddit API not configured.[/yellow]")
        console.print("\nTo enable Reddit scanning:")
        console.print("1. Go to https://www.reddit.com/prefs/apps")
        console.print("2. Create a 'script' app")
        console.print("3. Add to .env: REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
        return
    
    console.print("[cyan]Scanning WSB for trending stocks...[/cyan]")
    
    result = reddit_agent.get_wsb_momentum_scan()
    
    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return
    
    trending = result.get("trending_momentum_plays", [])
    
    if not trending:
        console.print("[yellow]No trending stocks found[/yellow]")
        return
    
    table = Table(title=f"🚀 WSB TRENDING ({result.get('posts_analyzed', 0)} posts analyzed)")
    table.add_column("Ticker", style="cyan")
    table.add_column("Mentions", style="white")
    table.add_column("Sentiment", style="white")
    table.add_column("Momentum Score", style="white")
    
    for stock in trending:
        sentiment = stock.get("sentiment", "neutral")
        sent_color = "green" if sentiment == "bullish" else "red" if sentiment == "bearish" else "yellow"
        
        table.add_row(
            stock.get("ticker", "N/A"),
            str(stock.get("mentions", 0)),
            f"[{sent_color}]{sentiment}[/{sent_color}]",
            f"{stock.get('momentum_score', 0):.0f}"
        )
    
    console.print(table)
    console.print()
    
    # Ask if user wants to analyze top ticker
    if trending:
        top_ticker = trending[0].get("ticker")
        console.print(f"[yellow]Top WSB ticker: {top_ticker}[/yellow]")
        response = console.input("[cyan]Analyze this stock? (y/n): [/cyan]")
        if response.lower() == 'y':
            analyze_stock(top_ticker)


def scan_penny_stocks():
    """Scan for high-potential penny stocks"""
    console.print(Panel.fit(
        "[bold magenta]💎 HIGH POTENTIAL SCANNER (< $10 | 100%+ Gains)[/bold magenta]\n"
        "Scanning dynamic market movers & watchlist for cheap stocks (<$10) ready to pop...",
        border_style="magenta"
    ))
    
    scanner = PennyStockScanner()
    candidates = scanner.scan(max_price=10.0)
    
    if not candidates:
        console.print("[yellow]No suitable candidates found.[/yellow]")
        return
        
    # Display results
    table = Table(title="💎 SLEEPER STOCKS (<$10 | 2x Potential)", border_style="magenta")
    table.add_column("Ticker", style="cyan", justify="center")
    table.add_column("Price", style="white", justify="right")
    table.add_column("Target (2x)", style="green", justify="right")
    table.add_column("RSI", style="white", justify="right")
    table.add_column("Risk", style="red", justify="center")
    table.add_column("Why it could double?", style="yellow")
    
    for c in candidates:
        rsi_color = "green" if c.rsi < 40 else "yellow"
        table.add_row(
            f"[bold]{c.ticker}[/bold]",
            f"${c.current_price:.2f}",
            f"${c.target_price:.2f}",
            f"[{rsi_color}]{c.rsi:.1f}[/{rsi_color}]",
            f"[bold red]{c.risk_level}[/bold red]",
            c.reason
        )
        
    console.print(table)
    console.print("\n[bold]⚠️  WARNING: Penny stocks are extremely risky. These are speculative plays.[/bold]")
    
    # Quick pick top 3
    console.print("\n[bold magenta]🔥 TOP 3 POTENTIAL DOUBLERS:[/bold magenta]")
    for i, c in enumerate(candidates[:3], 1):
         console.print(f"  {i}. [bold]{c.ticker}[/bold] (${c.current_price:.2f}) - Reason: {c.reason}")
         
    console.print()
    response = console.input("[cyan]Enter ticker for full analysis (or press Enter to skip): [/cyan]").strip().upper()
    if response:
        analyze_stock(response)


def interactive_mode():
    """Run in interactive mode"""
    print_banner()
    
    while True:
        console.print("\n[bold cyan]Options:[/bold cyan]")
        console.print("1. Analyze a specific stock")
        console.print("2. 📈 [green]DAY TRADING SCANNER[/green] (Best setups with buy/sell prices)")
        console.print("3. Scan market for movers")
        console.print("4. 💎 [magenta]PENNY STOCK SCANNER[/magenta] (Find 100% gainers)")
        console.print("5. Quick scan blue chips")
        console.print("6. Analyze event impact")
        console.print("7. 🔴 Scan WSB trending")
        console.print("8. 🤖 Show recommended LLM models")
        console.print("9. Exit")
        
        choice = console.input("\n[cyan]Enter choice (1-9): [/cyan]").strip()
        
        if choice == "1":
            ticker = console.input("[cyan]Enter stock ticker: [/cyan]").strip().upper()
            if ticker:
                analyze_stock(ticker)
        
        elif choice == "2":
            scan_day_trading()
        
        elif choice == "3":
            scan_market()
            
        elif choice == "4":
            scan_penny_stocks()
        
        elif choice == "5":
            console.print(f"[yellow]Scanning: {', '.join(BLUE_CHIP_TICKERS[:10])}...[/yellow]")
            quick_scan(BLUE_CHIP_TICKERS[:10])
        
        elif choice == "6":
            event = console.input("[cyan]Describe the event: [/cyan]").strip()
            tickers_input = console.input("[cyan]Enter tickers (comma-separated): [/cyan]").strip()
            tickers = [t.strip().upper() for t in tickers_input.split(",")]
            if event and tickers:
                analyze_event(event, tickers)
        
        elif choice == "7":
            scan_wsb_trending()
        
        elif choice == "8":
            print_model_recommendations()
            
        elif choice == "9":
            console.print("[yellow]Goodbye! 👋[/yellow]")
            break
        
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="TickTalker - Agentic AI Stock Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL              # Analyze Apple stock
  python main.py --scan            # Scan market for movers
  python main.py --quick AAPL MSFT GOOGL  # Quick scan multiple stocks
  python main.py --event "Fed raises rates" AAPL JPM  # Analyze event impact
  python main.py                   # Interactive mode
        """
    )
    
    parser.add_argument("ticker", nargs="?", help="Stock ticker to analyze")
    parser.add_argument("--scan", action="store_true", help="Scan market for top movers")
    parser.add_argument("--day-trade", action="store_true", help="Day trading scanner with buy/sell levels")
    parser.add_argument("--penny", action="store_true", help="Penny Stock scanner (100% potential)")
    parser.add_argument("--quick", nargs="+", metavar="TICKER", help="Quick scan multiple stocks")
    parser.add_argument("--event", metavar="EVENT", help="Analyze event impact")
    parser.add_argument("--blue-chips", action="store_true", help="Analyze top blue chip stocks")
    parser.add_argument("--wsb", action="store_true", help="Scan WallStreetBets for trending stocks")
    parser.add_argument("--models", action="store_true", help="Show recommended LLM models")
    
    args = parser.parse_args()
    
    print_banner()
    
    try:
        if args.day_trade:
            scan_day_trading()
            
        elif args.penny:
            scan_penny_stocks()
        
        elif args.scan:
            scan_market()
        
        elif args.quick:
            quick_scan(args.quick)
        
        elif args.event and args.ticker:
            # Event analysis with the remaining args as tickers
            tickers = [args.ticker] + (args.quick or [])
            analyze_event(args.event, tickers)
        
        elif args.blue_chips:
            console.print("[yellow]Analyzing top blue chip stocks...[/yellow]")
            quick_scan(BLUE_CHIP_TICKERS[:15])
        
        elif args.wsb:
            scan_wsb_trending()
        
        elif args.models:
            print_model_recommendations()
        
        elif args.ticker:
            analyze_stock(args.ticker)
        
        else:
            # Interactive mode
            interactive_mode()
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Goodbye! 👋[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        console.print("[yellow]If using Ollama, make sure it is running: ollama serve[/yellow]")
        console.print("[yellow]If using Gemini, make sure GOOGLE_API_KEY is set in .env[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()

