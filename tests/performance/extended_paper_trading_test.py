#!/usr/bin/env python3
"""
Extended Paper Trading Test
Multi-day simulation with detailed performance tracking
"""

import pandas as pd
import numpy as np
from paper_trading_system import PaperTradingBot
import os
from datetime import datetime, timedelta

def run_extended_simulation():
    """Run extended paper trading simulation"""
    
    print("FOREXSWING AI - EXTENDED PAPER TRADING SIMULATION")
    print("=" * 70)
    
    # Initialize bot with $10,000
    bot = PaperTradingBot(initial_balance=10000.0)
    
    # Load market data
    pairs_data = {}
    pair_files = [
        ('EUR/USD', 'EUR_USD_real_daily.csv'),
        ('GBP/USD', 'GBP_USD_real_daily.csv'),
        ('USD/JPY', 'USD_JPY_real_daily.csv')
    ]
    
    for pair_name, filename in pair_files:
        file_path = f"data/MarketData/{filename}"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            pairs_data[pair_name] = df
            print(f"Loaded {len(df)} candles for {pair_name}")
    
    if not pairs_data:
        print("[ERROR] No market data available")
        return
    
    # Simulate multiple trading periods
    simulation_periods = [
        ("Week 1", -200, -150),   # 50 days ago
        ("Week 2", -150, -100),   # 
        ("Week 3", -100, -50),    # 
        ("Week 4", -50, -1),      # Most recent
    ]
    
    daily_results = []
    
    for period_name, start_idx, end_idx in simulation_periods:
        print(f"\n" + "=" * 70)
        print(f"TRADING PERIOD: {period_name}")
        print("=" * 70)
        
        # Prepare data for this period
        period_data = {}
        for pair_name, full_data in pairs_data.items():
            # Get data slice with lookback for bot analysis
            period_slice = full_data.iloc[start_idx-80:end_idx]  # Extra 80 for warmup
            period_data[pair_name] = period_slice
            
            if len(period_slice) > 80:
                print(f"{pair_name}: {len(period_slice)} candles")
        
        # Run simulation for this period
        try:
            results = bot.simulate_trading_day(period_data)
            daily_results.append({
                'period': period_name,
                'balance': results['balance'],
                'equity': results['equity'],
                'return_pct': results['return_pct'],
                'total_trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'open_positions': results['open_positions']
            })
            
            print(f"\nPeriod Summary:")
            print(f"  Balance: ${results['balance']:,.2f}")
            print(f"  Return: {results['return_pct']:+.2f}%")
            print(f"  Trades: {results['total_trades']}")
            print(f"  Win Rate: {results['win_rate']:.1f}%")
            
        except Exception as e:
            print(f"[ERROR] Simulation failed for {period_name}: {e}")
    
    # Final comprehensive summary
    print(f"\n" + "=" * 70)
    print("EXTENDED SIMULATION COMPLETE")
    print("=" * 70)
    
    final_summary = bot.account.get_account_summary()
    
    print(f"FINAL RESULTS:")
    print(f"  Starting Balance: $10,000.00")
    print(f"  Final Balance: ${final_summary['balance']:,.2f}")
    print(f"  Final Equity: ${final_summary['equity']:,.2f}")
    print(f"  Total Return: {final_summary['return_pct']:+.2f}%")
    print(f"  Total P&L: ${final_summary['total_pnl']:+.2f}")
    
    print(f"\nTRADING STATISTICS:")
    print(f"  Total Trades: {final_summary['total_trades']}")
    print(f"  Winning Trades: {final_summary['winning_trades']}")
    print(f"  Losing Trades: {final_summary['losing_trades']}")
    print(f"  Win Rate: {final_summary['win_rate']:.1f}%")
    print(f"  Max Drawdown: {final_summary['max_drawdown']:.2f}%")
    
    print(f"\nCURRENT PORTFOLIO:")
    print(f"  Open Positions: {final_summary['open_positions']}")
    print(f"  Unrealized P&L: ${final_summary['unrealized_pnl']:+.2f}")
    print(f"  Free Margin: ${final_summary['free_margin']:,.2f}")
    
    # Performance analysis
    if final_summary['total_trades'] > 0:
        avg_trade = final_summary['total_pnl'] / final_summary['total_trades']
        print(f"  Average P&L per Trade: ${avg_trade:+.2f}")
    
    # Save detailed results
    bot.account.save_state("extended_simulation_results.json")
    print(f"\nDetailed results saved to: extended_simulation_results.json")
    
    # Performance assessment
    print(f"\n" + "=" * 70)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 70)
    
    return_pct = final_summary['return_pct']
    win_rate = final_summary['win_rate']
    max_dd = final_summary['max_drawdown']
    
    if return_pct > 10 and win_rate > 60:
        grade = "EXCELLENT"
        assessment = "Outstanding performance! Bot shows strong profitability."
    elif return_pct > 5 and win_rate > 50:
        grade = "GOOD"
        assessment = "Solid performance with positive returns."
    elif return_pct > 0 and win_rate >= 45:
        grade = "FAIR"
        assessment = "Modest profits, room for improvement."
    elif return_pct > -5:
        grade = "POOR"
        assessment = "Small losses, strategy needs refinement."
    else:
        grade = "VERY POOR"
        assessment = "Significant losses, major strategy overhaul needed."
    
    print(f"Performance Grade: {grade}")
    print(f"Assessment: {assessment}")
    
    if max_dd > 20:
        print(f"WARNING: High drawdown ({max_dd:.1f}%) - risk management needed")
    elif max_dd > 10:
        print(f"CAUTION: Moderate drawdown ({max_dd:.1f}%) - monitor risk")
    else:
        print(f"GOOD: Low drawdown ({max_dd:.1f}%) - good risk control")
    
    # Next steps recommendations
    print(f"\nRECOMMENDATIONS:")
    if return_pct > 5:
        print("✓ Strategy is profitable - consider live trading")
        print("✓ Monitor performance with larger position sizes")
    else:
        print("• Optimize signal thresholds for better entry/exit")
        print("• Consider longer backtesting period")
        print("• Review risk management parameters")
    
    if win_rate < 50:
        print("• Focus on improving signal accuracy")
        print("• Consider adding more confirmation filters")
    
    if max_dd > 15:
        print("• Reduce position sizes")
        print("• Implement stricter stop-losses")
        print("• Add portfolio diversification")
    
    return final_summary

def quick_paper_trading_test():
    """Quick test of paper trading system"""
    print("QUICK PAPER TRADING TEST")
    print("=" * 40)
    
    bot = PaperTradingBot(initial_balance=5000.0)
    
    # Load just EUR/USD for quick test
    if os.path.exists("data/MarketData/EUR_USD_real_daily.csv"):
        eur_data = pd.read_csv("data/MarketData/EUR_USD_real_daily.csv")
        test_data = {'EUR/USD': eur_data.tail(100)}
        
        results = bot.simulate_trading_day(test_data)
        
        print(f"\nQuick Test Results:")
        print(f"Return: {results['return_pct']:+.2f}%")
        print(f"Trades: {results['total_trades']}")
        print(f"Positions: {results['open_positions']}")
        
        return results
    else:
        print("No EUR/USD data available for quick test")
        return None

if __name__ == "__main__":
    # Run extended simulation
    run_extended_simulation()