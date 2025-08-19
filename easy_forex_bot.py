#!/usr/bin/env python3
"""
EASY FOREXSWING AI BOT - Simple Interface
The easiest way to interact with and use the ForexBot system
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

# Import our existing components
from ForexBot import ForexBot
from paper_trading_system import PaperTradingBot
import sys
sys.path.append('examples')
from demo_enhanced_gemini import simulate_gemini_analysis

class EasyForexBot:
    """
    Super simple interface for the ForexSwing AI system
    Everything you need in one easy-to-use class
    """
    
    def __init__(self):
        print("EASY FOREXSWING AI BOT")
        print("=" * 40)
        print("Initializing your AI trading assistant...")
        
        # Initialize core bot
        self.bot = ForexBot()
        self.available_pairs = self._load_available_pairs()
        
        print(f"[OK] AI Model loaded (55.2% accuracy)")
        print(f"[OK] Available pairs: {', '.join(self.available_pairs)}")
        print(f"[OK] Ready for trading analysis!")
        print()
    
    def _load_available_pairs(self) -> List[str]:
        """Find available currency pairs"""
        pairs = []
        data_dir = "data/MarketData"
        
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    pair = file.replace('_real_daily.csv', '').replace('_', '/')
                    pairs.append(pair)
        
        return pairs[:6]  # Limit to top 6 pairs
    
    def get_recommendation(self, pair: str = "EUR/USD") -> Dict:
        """
        Get a simple trading recommendation
        
        Args:
            pair: Currency pair (e.g., "EUR/USD", "GBP/USD")
            
        Returns:
            Simple recommendation with action and confidence
        """
        print(f"Analyzing {pair}...")
        
        # Load market data
        data = self._load_pair_data(pair)
        if data is None:
            return {"error": f"No data available for {pair}"}
        
        # Get AI recommendation
        recommendation = self.bot.get_final_recommendation(data, pair)
        
        # Simplify the response
        simple_rec = {
            "pair": pair,
            "action": recommendation['action'],
            "confidence": f"{recommendation['confidence']:.1%}",
            "strength": self._get_strength_rating(recommendation['confidence']),
            "trend": recommendation.get('trend_signal', 'neutral'),
            "advice": self._get_trading_advice(recommendation),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return simple_rec
    
    def analyze_all_pairs(self) -> List[Dict]:
        """Get recommendations for all available pairs"""
        print("Analyzing all available currency pairs...")
        print("-" * 40)
        
        results = []
        
        for pair in self.available_pairs:
            try:
                rec = self.get_recommendation(pair)
                if "error" not in rec:
                    results.append(rec)
                    print(f"{pair}: {rec['action']} ({rec['confidence']}) - {rec['strength']}")
            except Exception as e:
                print(f"{pair}: Error - {e}")
        
        # Sort by confidence
        results.sort(key=lambda x: float(x['confidence'].replace('%', '')), reverse=True)
        
        return results
    
    def get_best_opportunities(self, min_confidence: float = 0.6) -> List[Dict]:
        """Find the best trading opportunities"""
        print(f"Finding best opportunities (min {min_confidence:.0%} confidence)...")
        
        all_recommendations = self.analyze_all_pairs()
        
        # Filter by confidence
        opportunities = []
        for rec in all_recommendations:
            confidence_val = float(rec['confidence'].replace('%', '')) / 100
            if confidence_val >= min_confidence and rec['action'] != 'HOLD':
                opportunities.append(rec)
        
        print(f"\nFound {len(opportunities)} high-confidence opportunities:")
        for opp in opportunities:
            print(f"  {opp['pair']}: {opp['action']} ({opp['confidence']})")
        
        return opportunities
    
    def start_paper_trading(self, initial_balance: float = 10000.0) -> Dict:
        """Start a paper trading session"""
        print(f"\nStarting paper trading with ${initial_balance:,.2f}...")
        
        # Initialize paper trading
        paper_bot = PaperTradingBot(initial_balance)
        
        # Get market data for all pairs
        market_data = {}
        for pair in self.available_pairs:
            data = self._load_pair_data(pair)
            if data is not None:
                market_data[pair] = data
        
        # Run paper trading simulation
        results = paper_bot.simulate_trading_day(market_data)
        
        # Save results
        filename = f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        paper_bot.account.save_state(filename)
        
        return {
            "session_file": filename,
            "final_balance": results['balance'],
            "return_pct": results['return_pct'],
            "total_trades": results['total_trades'],
            "win_rate": results['win_rate'],
            "open_positions": results['open_positions']
        }
    
    def get_market_summary(self) -> Dict:
        """Get a summary of current market conditions"""
        print("Getting market summary...")
        
        recommendations = self.analyze_all_pairs()
        
        # Analyze overall sentiment
        buy_signals = sum(1 for r in recommendations if r['action'] == 'BUY')
        sell_signals = sum(1 for r in recommendations if r['action'] == 'SELL')
        hold_signals = sum(1 for r in recommendations if r['action'] == 'HOLD')
        
        # Calculate average confidence
        confidences = [float(r['confidence'].replace('%', '')) for r in recommendations]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Market sentiment
        if buy_signals > sell_signals * 1.5:
            sentiment = "BULLISH"
        elif sell_signals > buy_signals * 1.5:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        
        return {
            "market_sentiment": sentiment,
            "average_confidence": f"{avg_confidence:.1f}%",
            "signal_distribution": {
                "BUY": buy_signals,
                "SELL": sell_signals,
                "HOLD": hold_signals
            },
            "total_pairs_analyzed": len(recommendations),
            "high_confidence_signals": len([r for r in recommendations if float(r['confidence'].replace('%', '')) > 70])
        }
    
    def explain_recommendation(self, pair: str = "EUR/USD") -> str:
        """Get a detailed explanation of a recommendation"""
        rec = self.get_recommendation(pair)
        
        if "error" in rec:
            return f"Cannot analyze {pair}: {rec['error']}"
        
        explanation = f"""
FOREX AI ANALYSIS FOR {pair}
{'=' * 40}

RECOMMENDATION: {rec['action']}
Confidence: {rec['confidence']} ({rec['strength']})
Trend: {rec['trend']}

EXPLANATION:
{rec['advice']}

WHAT THIS MEANS:
- The AI model analyzed recent price patterns
- Confidence of {rec['confidence']} indicates {rec['strength'].lower()} signal strength
- Current trend is {rec['trend']}
- Generated at {rec['timestamp']}

RISK LEVEL:
{self._get_risk_explanation(rec['confidence'])}
"""
        return explanation
    
    def _load_pair_data(self, pair: str) -> Optional[pd.DataFrame]:
        """Load data for a currency pair"""
        # Convert EUR/USD to EUR_USD format
        file_pair = pair.replace('/', '_')
        file_path = f"data/MarketData/{file_pair}_real_daily.csv"
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return df.tail(120)  # Last 120 days
        return None
    
    def _get_strength_rating(self, confidence: float) -> str:
        """Convert confidence to strength rating"""
        if confidence >= 0.8:
            return "VERY STRONG"
        elif confidence >= 0.65:
            return "STRONG"
        elif confidence >= 0.5:
            return "MODERATE"
        else:
            return "WEAK"
    
    def _get_trading_advice(self, recommendation: Dict) -> str:
        """Generate simple trading advice"""
        action = recommendation['action']
        confidence = recommendation['confidence']
        
        if action == "BUY":
            if confidence > 0.7:
                return "Strong buy signal. Consider entering a long position."
            else:
                return "Moderate buy signal. Consider small position or wait for confirmation."
        elif action == "SELL":
            if confidence > 0.7:
                return "Strong sell signal. Consider entering a short position."
            else:
                return "Moderate sell signal. Consider small position or wait for confirmation."
        else:
            return "No clear direction. Consider holding current positions or staying in cash."
    
    def _get_risk_explanation(self, confidence_str: str) -> str:
        """Explain risk level based on confidence"""
        confidence = float(confidence_str.replace('%', '')) / 100
        
        if confidence >= 0.8:
            return "LOW RISK: Very high confidence signal"
        elif confidence >= 0.65:
            return "MODERATE RISK: Good confidence level"
        elif confidence >= 0.5:
            return "HIGHER RISK: Moderate confidence, use smaller positions"
        else:
            return "HIGH RISK: Low confidence, consider avoiding this trade"

def interactive_mode():
    """Interactive mode for easy bot usage"""
    bot = EasyForexBot()
    
    while True:
        print("\n" + "=" * 50)
        print("EASY FOREXSWING AI - INTERACTIVE MODE")
        print("=" * 50)
        print("1. Get recommendation for specific pair")
        print("2. Analyze all pairs")
        print("3. Find best opportunities")
        print("4. Get market summary")
        print("5. Start paper trading")
        print("6. Detailed explanation")
        print("0. Exit")
        
        try:
            choice = input("\nSelect option (0-6): ").strip()
            
            if choice == "0":
                print("Thanks for using ForexSwing AI!")
                break
            elif choice == "1":
                pair = input("Enter currency pair (e.g., EUR/USD): ").strip().upper()
                if not pair:
                    pair = "EUR/USD"
                rec = bot.get_recommendation(pair)
                print(f"\nRecommendation for {pair}:")
                for key, value in rec.items():
                    print(f"  {key}: {value}")
            elif choice == "2":
                recommendations = bot.analyze_all_pairs()
                print(f"\nAll pair analysis complete:")
                for rec in recommendations:
                    print(f"  {rec['pair']}: {rec['action']} ({rec['confidence']})")
            elif choice == "3":
                min_conf = input("Minimum confidence (0.6): ").strip()
                min_conf = float(min_conf) if min_conf else 0.6
                opportunities = bot.get_best_opportunities(min_conf)
                if opportunities:
                    print(f"\nBest opportunities:")
                    for opp in opportunities:
                        print(f"  {opp['pair']}: {opp['action']} ({opp['confidence']}) - {opp['advice']}")
                else:
                    print("No high-confidence opportunities found.")
            elif choice == "4":
                summary = bot.get_market_summary()
                print(f"\nMarket Summary:")
                for key, value in summary.items():
                    print(f"  {key}: {value}")
            elif choice == "5":
                balance = input("Starting balance (10000): ").strip()
                balance = float(balance) if balance else 10000.0
                results = bot.start_paper_trading(balance)
                print(f"\nPaper trading results:")
                for key, value in results.items():
                    print(f"  {key}: {value}")
            elif choice == "6":
                pair = input("Enter pair for detailed explanation (EUR/USD): ").strip().upper()
                if not pair:
                    pair = "EUR/USD"
                explanation = bot.explain_recommendation(pair)
                print(explanation)
            else:
                print("Invalid option. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("FOREXSWING AI - EASY BOT INTERFACE")
    print("=" * 60)
    print("Choose how you want to use the bot:")
    print("1. Interactive mode (recommended for beginners)")
    print("2. Quick single recommendation")
    print("3. Find best opportunities")
    
    try:
        mode = input("\nSelect mode (1-3): ").strip()
        
        if mode == "1":
            interactive_mode()
        elif mode == "2":
            bot = EasyForexBot()
            pair = input("Enter currency pair (EUR/USD): ").strip().upper()
            if not pair:
                pair = "EUR/USD"
            rec = bot.get_recommendation(pair)
            print(f"\nRecommendation for {pair}:")
            for key, value in rec.items():
                print(f"  {key}: {value}")
        elif mode == "3":
            bot = EasyForexBot()
            opportunities = bot.get_best_opportunities()
            if opportunities:
                print(f"\nTop opportunities:")
                for opp in opportunities:
                    print(f"{opp['pair']}: {opp['action']} ({opp['confidence']}) - {opp['strength']}")
            else:
                print("No high-confidence opportunities found.")
        else:
            print("Invalid mode. Running interactive mode...")
            interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")