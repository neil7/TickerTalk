"""
LangGraph Workflow Orchestrator
Coordinates all agents in a multi-agent workflow
"""
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from data_sources import (
    StockDataFetcher,
    NewsFetcher,
    EconomicDataFetcher,
    SocialDataFetcher,
    MarketMoversFetcher,
    RedditDataFetcher,
)
from agents import (
    TechnicalAnalysisAgent,
    SentimentAnalysisAgent,
    FundamentalAnalysisAgent,
    RiskManagementAgent,
    PortfolioManagerAgent,
    RedditSentimentAgent,
)

console = Console()


# Define Agent State
class AgentState(TypedDict):
    """State shared across all agents"""
    messages: Annotated[List[str], operator.add]
    ticker: str
    market_data: Dict[str, Any]
    technical_analysis: Dict[str, Any]
    sentiment_data: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    reddit_analysis: Dict[str, Any]  # Reddit-specific sentiment
    fundamental_analysis: Dict[str, Any]
    economic_indicators: Dict[str, Any]
    social_data: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    final_recommendation: Dict[str, Any]


class StockAnalysisWorkflow:
    """
    Orchestrates the multi-agent stock analysis workflow
    Using LangGraph for agent coordination
    """
    
    def __init__(self):
        # Initialize data fetchers
        self.stock_fetcher = StockDataFetcher()
        self.news_fetcher = NewsFetcher()
        self.economic_fetcher = EconomicDataFetcher()
        self.social_fetcher = SocialDataFetcher()
        self.movers_fetcher = MarketMoversFetcher()
        self.reddit_fetcher = RedditDataFetcher()
        
        # Initialize agents
        self.technical_agent = TechnicalAnalysisAgent()
        self.sentiment_agent = SentimentAnalysisAgent()
        self.reddit_agent = RedditSentimentAgent()
        self.fundamental_agent = FundamentalAnalysisAgent()
        self.risk_agent = RiskManagementAgent()
        self.portfolio_agent = PortfolioManagerAgent()
        
        # Build workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("data_collection", self._data_collection_node)
        workflow.add_node("technical_analysis", self._technical_analysis_node)
        workflow.add_node("sentiment_analysis", self._sentiment_analysis_node)
        workflow.add_node("reddit_analysis", self._reddit_analysis_node)
        workflow.add_node("fundamental_analysis", self._fundamental_analysis_node)
        workflow.add_node("risk_management", self._risk_management_node)
        workflow.add_node("portfolio_manager", self._portfolio_manager_node)
        
        # Define the flow - Sequential execution to avoid concurrent state updates
        workflow.set_entry_point("data_collection")
        
        # Sequential pipeline: each agent runs after the previous completes
        workflow.add_edge("data_collection", "technical_analysis")
        workflow.add_edge("technical_analysis", "risk_management")
        workflow.add_edge("risk_management", "sentiment_analysis")
        workflow.add_edge("sentiment_analysis", "reddit_analysis")
        workflow.add_edge("reddit_analysis", "fundamental_analysis")
        workflow.add_edge("fundamental_analysis", "portfolio_manager")
        workflow.add_edge("portfolio_manager", END)
        
        return workflow.compile()
    
    def _data_collection_node(self, state: AgentState) -> AgentState:
        """Node for collecting all data"""
        ticker = state["ticker"]
        
        console.print(f"[cyan]📊 Collecting data for {ticker}...[/cyan]")
        
        # Fetch all data in parallel (could be improved with async)
        state["market_data"] = self.stock_fetcher.get_stock_data(ticker)
        
        # Fetch news and sentiment data
        news_data = self.news_fetcher.get_all_news_for_analysis(ticker)
        state["sentiment_data"] = {
            "stock_news": news_data.get("stock_specific", []),
            "white_house_news": news_data.get("white_house", []),
            "federal_news": news_data.get("federal", []),
            "market_news": news_data.get("market_general", []),
            "google_results": news_data.get("google_results", []),
        }
        
        # Fetch economic indicators
        state["economic_indicators"] = self.economic_fetcher.get_all_indicators()
        
        # Fetch social data (Reddit, trends)
        state["social_data"] = self.social_fetcher.get_social_summary(ticker)
        
        state["messages"].append(f"Data collection completed for {ticker}")
        console.print("[green]✓ Data collection completed[/green]")
        
        return state
    
    def _technical_analysis_node(self, state: AgentState) -> AgentState:
        """Node for technical analysis"""
        console.print("[cyan]📈 Performing technical analysis...[/cyan]")
        state = self.technical_agent.analyze(state)
        console.print("[green]✓ Technical analysis completed[/green]")
        return state
    
    def _sentiment_analysis_node(self, state: AgentState) -> AgentState:
        """Node for sentiment analysis"""
        console.print("[cyan]💭 Analyzing news sentiment...[/cyan]")
        state = self.sentiment_agent.analyze(state)
        console.print("[green]✓ News sentiment analysis completed[/green]")
        return state
    
    def _reddit_analysis_node(self, state: AgentState) -> AgentState:
        """Node for Reddit sentiment analysis"""
        console.print("[cyan]🔴 Analyzing Reddit sentiment...[/cyan]")
        state = self.reddit_agent.analyze(state)
        console.print("[green]✓ Reddit sentiment analysis completed[/green]")
        return state
    
    def _fundamental_analysis_node(self, state: AgentState) -> AgentState:
        """Node for fundamental analysis"""
        console.print("[cyan]📊 Analyzing fundamentals...[/cyan]")
        state = self.fundamental_agent.analyze(state)
        console.print("[green]✓ Fundamental analysis completed[/green]")
        return state
    
    def _risk_management_node(self, state: AgentState) -> AgentState:
        """Node for risk assessment"""
        console.print("[cyan]⚠️  Assessing risk...[/cyan]")
        state = self.risk_agent.analyze(state)
        console.print("[green]✓ Risk assessment completed[/green]")
        return state
    
    def _portfolio_manager_node(self, state: AgentState) -> AgentState:
        """Node for final recommendation"""
        console.print("[cyan]🎯 Generating final recommendation...[/cyan]")
        state = self.portfolio_agent.analyze(state)
        console.print("[green]✓ Final recommendation generated[/green]")
        return state
    
    def analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Run complete analysis for a stock
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            
        Returns:
            Complete analysis results including final recommendation
        """
        console.print(Panel.fit(
            f"[bold cyan]🤖 AGENTIC AI STOCK ANALYSIS: {ticker.upper()}[/bold cyan]",
            border_style="cyan"
        ))
        
        initial_state = AgentState(
            messages=[],
            ticker=ticker.upper(),
            market_data={},
            technical_analysis={},
            sentiment_data={},
            sentiment_analysis={},
            reddit_analysis={},
            fundamental_analysis={},
            economic_indicators={},
            social_data={},
            risk_assessment={},
            final_recommendation={},
        )
        
        # Run the workflow
        result = self.workflow.invoke(initial_state)
        
        return result
    
    def quick_scan(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Quick scan multiple stocks for potential opportunities
        Uses lighter analysis for speed
        """
        results = []
        
        console.print(Panel.fit(
            f"[bold cyan]🔍 QUICK SCAN: {len(tickers)} stocks[/bold cyan]",
            border_style="cyan"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Scanning...", total=len(tickers))
            
            for ticker in tickers:
                progress.update(task, description=f"Scanning {ticker}...")
                
                try:
                    market_data = self.stock_fetcher.get_stock_data(ticker)
                    
                    if "error" not in market_data:
                        quick_result = {
                            "ticker": ticker,
                            "price": market_data.get("current_price"),
                            "change_pct": market_data.get("change_percent"),
                            "volume_ratio": (market_data.get("volume", 0) / 
                                           market_data.get("avg_volume", 1) 
                                           if market_data.get("avg_volume") else 1),
                            "assessment": self.portfolio_agent.quick_assessment(ticker, market_data),
                        }
                        results.append(quick_result)
                except Exception as e:
                    results.append({
                        "ticker": ticker,
                        "error": str(e),
                    })
                
                progress.advance(task)
        
        return results
    
    def scan_market_movers(self) -> Dict[str, Any]:
        """Scan market for top movers and potential opportunities"""
        console.print(Panel.fit(
            "[bold cyan]🔍 SCANNING MARKET FOR TOP MOVERS[/bold cyan]",
            border_style="cyan"
        ))
        
        # Get market movers
        movers = self.movers_fetcher.get_daily_movers()
        sectors = self.movers_fetcher.get_sector_performance()
        momentum = self.movers_fetcher.identify_momentum_candidates()
        
        return {
            "daily_movers": movers,
            "sector_performance": sectors,
            "momentum_candidates": momentum,
        }
    
    def analyze_event_impact(self, event: str, tickers: List[str]) -> Dict[str, Any]:
        """
        Analyze the potential impact of a specific event on multiple stocks
        
        Args:
            event: Description of the event (e.g., "Fed raises interest rates")
            tickers: List of stock symbols to analyze
        """
        console.print(Panel.fit(
            f"[bold cyan]📰 EVENT IMPACT ANALYSIS[/bold cyan]\n{event}",
            border_style="cyan"
        ))
        
        results = {}
        
        for ticker in tickers:
            console.print(f"[cyan]Analyzing impact on {ticker}...[/cyan]")
            impact = self.sentiment_agent.analyze_specific_event(event, ticker)
            results[ticker] = impact
        
        return results


def create_workflow() -> StockAnalysisWorkflow:
    """Create and return a configured workflow instance"""
    return StockAnalysisWorkflow()

