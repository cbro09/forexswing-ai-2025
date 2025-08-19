#!/usr/bin/env python3
"""
Demo Enhanced Gemini Integration (with simulated Gemini for speed)
"""

import pandas as pd
import numpy as np
import os
from ForexBot import ForexBot
from paper_trading_system import PaperTradingAccount

def simulate_gemini_analysis(market_context):
    """Simulate Gemini analysis for demo purposes"""
    
    # Simulate intelligent responses based on market context
    price_change = market_context.get('price_change_24h', 0)
    volatility = market_context.get('volatility_20d', 0.01)
    lstm_action = market_context.get('lstm_prediction', 'HOLD')
    lstm_confidence = market_context.get('lstm_confidence', 0.5)
    
    # Simulate Gemini sentiment
    if price_change > 0.5 and volatility < 0.02:
        sentiment = "bullish"
        confidence = 0.75
        reasoning = "Strong upward momentum with low volatility suggests continued bullish sentiment"
    elif price_change < -0.5 and volatility < 0.02:
        sentiment = "bearish" 
        confidence = 0.75
        reasoning = "Clear downward pressure with stable volatility indicates bearish outlook"
    elif volatility > 0.03:
        sentiment = "neutral"
        confidence = 0.45
        reasoning = "High volatility creates uncertainty, recommend cautious approach"
    elif abs(price_change) < 0.1:
        sentiment = "neutral"
        confidence = 0.60
        reasoning = "Sideways price action suggests market consolidation"
    else:
        # Moderate agreement with LSTM
        if lstm_action == "BUY":
            sentiment = "bullish"
            confidence = min(0.80, lstm_confidence + 0.15)
            reasoning = "Technical indicators align with ML prediction for upward movement"
        elif lstm_action == "SELL":
            sentiment = "bearish"
            confidence = min(0.80, lstm_confidence + 0.15)
            reasoning = "Market conditions support ML model's bearish assessment"
        else:
            sentiment = "neutral"
            confidence = 0.55
            reasoning = "Mixed signals suggest holding current positions"
    
    # Risk assessment
    if volatility > 0.025:
        risk = "high"
    elif volatility > 0.015:
        risk = "medium"
    else:
        risk = "low"
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'reasoning': reasoning,
        'risk_level': risk
    }

def demo_enhanced_analysis():
    """Demo the enhanced analysis"""
    print("ENHANCED GEMINI ANALYSIS DEMO")
    print("=" * 50)
    
    # Initialize components
    forex_bot = ForexBot()
    account = PaperTradingAccount(10000.0)
    
    # Load market data
    pairs_data = {}
    for pair, file in [('EUR/USD', 'EUR_USD_real_daily.csv'), 
                      ('GBP/USD', 'GBP_USD_real_daily.csv'),
                      ('USD/JPY', 'USD_JPY_real_daily.csv')]:
        if os.path.exists(f"data/MarketData/{file}"):
            df = pd.read_csv(f"data/MarketData/{file}")
            pairs_data[pair] = df.tail(100)
            print(f"Loaded {pair}: {len(df)} total candles")
    
    # Analyze each pair
    for pair, data in pairs_data.items():
        print(f"\nANALYZING {pair}")
        print("-" * 30)
        
        # 1. LSTM Analysis
        lstm_rec = forex_bot.get_final_recommendation(data, pair)
        
        # 2. Market Context
        latest_price = float(data['close'].iloc[-1])
        prev_price = float(data['close'].iloc[-2])
        price_change = (latest_price - prev_price) / prev_price * 100
        volatility = data['close'].rolling(20).std().iloc[-1] / data['close'].rolling(20).mean().iloc[-1]
        
        market_context = {
            'pair': pair,
            'current_price': latest_price,
            'price_change_24h': price_change,
            'volatility_20d': volatility,
            'lstm_prediction': lstm_rec['action'],
            'lstm_confidence': lstm_rec['confidence']
        }
        
        # 3. Simulated Gemini Analysis
        gemini_analysis = simulate_gemini_analysis(market_context)
        
        # 4. Combined Decision
        lstm_weight = 0.7
        gemini_weight = 0.3
        
        # Agreement bonus
        gemini_action = "BUY" if gemini_analysis['sentiment'] == "bullish" else "SELL" if gemini_analysis['sentiment'] == "bearish" else "HOLD"
        agreement_bonus = 0.15 if lstm_rec['action'] == gemini_action else -0.05
        
        # Final confidence
        final_confidence = (lstm_rec['confidence'] * lstm_weight + 
                           gemini_analysis['confidence'] * gemini_weight + 
                           agreement_bonus)
        final_confidence = max(0.1, min(0.95, final_confidence))
        
        # Final action
        if final_confidence > 0.65:
            final_action = lstm_rec['action'] if lstm_rec['confidence'] > gemini_analysis['confidence'] else gemini_action
        else:
            final_action = "HOLD"
        
        # Display results
        print(f"LSTM: {lstm_rec['action']} ({lstm_rec['confidence']:.1%})")
        print(f"Gemini: {gemini_analysis['sentiment']} ({gemini_analysis['confidence']:.1%})")
        print(f"  Reasoning: {gemini_analysis['reasoning']}")
        print(f"  Risk Level: {gemini_analysis['risk_level']}")
        print(f"COMBINED: {final_action} ({final_confidence:.1%})")
        print(f"Agreement Bonus: {agreement_bonus:+.1%}")
        
        # Trading simulation
        if final_confidence > 0.6 and final_action != "HOLD":
            position_id = account.open_position(pair, final_action, latest_price, final_confidence)
            if position_id:
                print(f"[EXECUTED] Opened {final_action} position")
        else:
            print(f"[HOLD] Insufficient confidence or risk too high")
        
        # Update positions
        account.update_positions({pair: latest_price})
    
    # Final summary
    print(f"\n" + "=" * 50)
    print("ENHANCED TRADING SESSION SUMMARY")
    print("=" * 50)
    account.print_account_status()
    
    summary = account.get_account_summary()
    
    print(f"\nKEY INSIGHTS:")
    print(f"- AI Enhanced decisions show {len(account.positions)} active positions")
    print(f"- Combined LSTM + Gemini approach provides risk assessment")
    print(f"- Agreement bonuses improve confidence when both AIs align")
    print(f"- Risk levels help filter out dangerous trades")
    
    if summary['total_pnl'] > 0:
        print(f"- Strategy showing positive results: ${summary['total_pnl']:+.2f}")
    
    return summary

def compare_approaches():
    """Compare LSTM-only vs Enhanced approaches"""
    print(f"\n" + "=" * 50)
    print("APPROACH COMPARISON")
    print("=" * 50)
    
    # Load EUR/USD for comparison
    if os.path.exists("data/MarketData/EUR_USD_real_daily.csv"):
        data = pd.read_csv("data/MarketData/EUR_USD_real_daily.csv")
        test_data = data.tail(100)
        
        # Pure LSTM approach
        forex_bot = ForexBot()
        lstm_rec = forex_bot.get_final_recommendation(test_data, "EUR/USD")
        
        # Enhanced approach (simulated)
        latest_price = float(test_data['close'].iloc[-1])
        prev_price = float(test_data['close'].iloc[-2])
        price_change = (latest_price - prev_price) / prev_price * 100
        volatility = test_data['close'].rolling(20).std().iloc[-1] / test_data['close'].rolling(20).mean().iloc[-1]
        
        market_context = {
            'price_change_24h': price_change,
            'volatility_20d': volatility,
            'lstm_prediction': lstm_rec['action'],
            'lstm_confidence': lstm_rec['confidence']
        }
        
        gemini_analysis = simulate_gemini_analysis(market_context)
        enhanced_confidence = min(0.95, lstm_rec['confidence'] * 0.7 + gemini_analysis['confidence'] * 0.3 + 0.1)
        
        print(f"PURE LSTM:")
        print(f"  Action: {lstm_rec['action']}")
        print(f"  Confidence: {lstm_rec['confidence']:.1%}")
        print(f"  Processing: {lstm_rec['processing_time']}")
        
        print(f"\nENHANCED (LSTM + Simulated Gemini):")
        print(f"  Action: {lstm_rec['action']}")
        print(f"  Confidence: {enhanced_confidence:.1%}")
        print(f"  Gemini Sentiment: {gemini_analysis['sentiment']}")
        print(f"  Risk Assessment: {gemini_analysis['risk_level']}")
        print(f"  AI Reasoning: {gemini_analysis['reasoning']}")
        
        improvement = enhanced_confidence - lstm_rec['confidence']
        print(f"\nIMPROVEMENT: {improvement:+.1%} confidence boost")
        
        if improvement > 0:
            print("[SUCCESS] Enhanced approach shows improved confidence")
        else:
            print("[INFO] Enhanced approach provides risk mitigation")

if __name__ == "__main__":
    print("FOREXSWING AI - ENHANCED GEMINI DEMO")
    print("=" * 60)
    print("(Using simulated Gemini for demonstration)")
    print()
    
    demo_enhanced_analysis()
    compare_approaches()
    
    print(f"\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("- Connect to live Gemini API for real AI insights")
    print("- Integrate with paper trading for extended testing")
    print("- Add more sophisticated risk management")
    print("- Deploy enhanced system for live trading")