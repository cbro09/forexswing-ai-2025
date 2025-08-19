#!/usr/bin/env python3
"""
Enhanced Gemini Trading System
Integrates Gemini AI with ForexBot, Paper Trading, and JAX indicators
"""

import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import our existing components
from ForexBot import ForexBot
from src.integrations.optimized_gemini import OptimizedGeminiInterpreter
from paper_trading_system import PaperTradingBot, PaperTradingAccount

@dataclass
class MarketAnalysis:
    """Comprehensive market analysis result"""
    pair: str
    timestamp: str
    
    # LSTM Analysis
    lstm_action: str
    lstm_confidence: float
    lstm_probabilities: Dict[str, float]
    
    # Gemini Analysis
    gemini_sentiment: str
    gemini_confidence: float
    gemini_reasoning: str
    gemini_risk_assessment: str
    
    # Combined Analysis
    final_action: str
    final_confidence: float
    confidence_boost: float
    risk_level: str
    
    # Technical Indicators
    trend_direction: str
    volatility_level: str
    support_resistance: Dict[str, float]

class EnhancedGeminiTradingSystem:
    """
    Advanced trading system that combines:
    1. ForexBot LSTM predictions
    2. Gemini AI market interpretation
    3. Paper trading execution
    4. JAX indicators (when available)
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        print("INITIALIZING ENHANCED GEMINI TRADING SYSTEM")
        print("=" * 60)
        
        # Initialize core components
        self.forex_bot = ForexBot()
        self.gemini = OptimizedGeminiInterpreter(cache_size=100, cache_duration_minutes=30)
        self.paper_account = PaperTradingAccount(initial_balance)
        
        # Configuration
        self.gemini_weight = 0.3  # Weight for Gemini in final decision
        self.min_confidence_threshold = 0.55
        self.enable_advanced_analysis = True
        
        print(f"[OK] ForexBot: LSTM model loaded")
        print(f"[OK] Gemini AI: {'Available' if self.gemini.gemini_available else 'Not Available'}")
        print(f"[OK] Paper Trading: ${initial_balance:,.2f} starting balance")
        print(f"[OK] Gemini weight in decisions: {self.gemini_weight:.1%}")
    
    def analyze_market_with_gemini(self, data: pd.DataFrame, pair: str) -> MarketAnalysis:
        """
        Comprehensive market analysis using both LSTM and Gemini
        """
        print(f"\nANALYZING {pair} WITH ENHANCED AI")
        print("-" * 40)
        
        # 1. Get LSTM prediction
        start_time = time.time()
        lstm_recommendation = self.forex_bot.get_final_recommendation(data, pair)
        lstm_time = time.time() - start_time
        
        lstm_action = lstm_recommendation['action']
        lstm_confidence = lstm_recommendation['confidence']
        
        print(f"LSTM Analysis ({lstm_time:.3f}s):")
        print(f"  Action: {lstm_action}")
        print(f"  Confidence: {lstm_confidence:.1%}")
        
        # 2. Prepare market context for Gemini
        latest_price = float(data['close'].iloc[-1])
        prev_price = float(data['close'].iloc[-2])
        price_change = (latest_price - prev_price) / prev_price * 100
        
        # Calculate additional context
        volatility = data['close'].rolling(20).std().iloc[-1] / data['close'].rolling(20).mean().iloc[-1]
        price_range_20d = (data['close'].rolling(20).max().iloc[-1] - data['close'].rolling(20).min().iloc[-1]) / data['close'].rolling(20).mean().iloc[-1]
        volume_avg = data['volume'].rolling(10).mean().iloc[-1] if 'volume' in data.columns else 0
        
        market_context = {
            "pair": pair,
            "current_price": latest_price,
            "price_change_24h": price_change,
            "volatility_20d": volatility,
            "price_range_20d": price_range_20d,
            "volume_average": volume_avg,
            "lstm_prediction": lstm_action,
            "lstm_confidence": lstm_confidence,
            "trend": lstm_recommendation.get('trend_signal', 'neutral')
        }
        
        # 3. Get Gemini analysis
        gemini_sentiment = "neutral"
        gemini_confidence = 0.5
        gemini_reasoning = "Gemini not available"
        risk_assessment = "medium"
        
        if self.gemini.gemini_available and self.enable_advanced_analysis:
            try:
                # Market interpretation
                gemini_start = time.time()
                market_interpretation = self.gemini.interpret_market_quickly(market_context, pair)
                
                # Signal validation
                signal_data = {
                    'ml_prediction': 0.7 if lstm_action == 'BUY' else 0.3 if lstm_action == 'SELL' else 0.5,
                    'ml_confidence': lstm_confidence,
                    'rsi': 50 + (price_change * 2),  # Simplified RSI approximation
                    'trend_direction': lstm_recommendation.get('trend_signal', 'neutral'),
                    'volatility': volatility,
                    'price_momentum': price_change
                }
                
                validation_result = self.gemini.validate_signal_fast(signal_data)
                gemini_time = time.time() - gemini_start
                
                # Extract Gemini insights
                gemini_sentiment = market_interpretation.get('sentiment', 'neutral')
                gemini_confidence = market_interpretation.get('confidence', 50) / 100.0
                gemini_reasoning = market_interpretation.get('key_factor', 'Standard analysis')
                risk_assessment = validation_result.get('risk_level', 'medium')
                
                print(f"\nGemini Analysis ({gemini_time:.3f}s):")
                print(f"  Sentiment: {gemini_sentiment}")
                print(f"  Confidence: {gemini_confidence:.1%}")
                print(f"  Reasoning: {gemini_reasoning}")
                print(f"  Risk Level: {risk_assessment}")
                
            except Exception as e:
                print(f"  Gemini analysis failed: {e}")
        
        # 4. Combine LSTM and Gemini decisions
        final_action, final_confidence, confidence_boost = self._combine_ai_decisions(
            lstm_action, lstm_confidence, gemini_sentiment, gemini_confidence
        )
        
        print(f"\nCOMBINED DECISION:")
        print(f"  Final Action: {final_action}")
        print(f"  Final Confidence: {final_confidence:.1%}")
        print(f"  Confidence Boost: {confidence_boost:+.1%}")
        
        # 5. Technical analysis
        trend_direction = self._analyze_trend(data)
        volatility_level = self._classify_volatility(volatility)
        support_resistance = self._find_support_resistance(data)
        
        # Create comprehensive analysis
        analysis = MarketAnalysis(
            pair=pair,
            timestamp=datetime.now().isoformat(),
            lstm_action=lstm_action,
            lstm_confidence=lstm_confidence,
            lstm_probabilities={
                'HOLD': float(lstm_recommendation.get('hold_prob', 0.33)),
                'BUY': float(lstm_recommendation.get('buy_prob', 0.33)),
                'SELL': float(lstm_recommendation.get('sell_prob', 0.33))
            },
            gemini_sentiment=gemini_sentiment,
            gemini_confidence=gemini_confidence,
            gemini_reasoning=gemini_reasoning,
            gemini_risk_assessment=risk_assessment,
            final_action=final_action,
            final_confidence=final_confidence,
            confidence_boost=confidence_boost,
            risk_level=risk_assessment,
            trend_direction=trend_direction,
            volatility_level=volatility_level,
            support_resistance=support_resistance
        )
        
        return analysis
    
    def _combine_ai_decisions(self, lstm_action: str, lstm_conf: float, 
                            gemini_sentiment: str, gemini_conf: float) -> Tuple[str, float, float]:
        """
        Intelligent fusion of LSTM and Gemini decisions
        """
        
        # Convert Gemini sentiment to action
        gemini_action = "HOLD"
        if gemini_sentiment == "bullish":
            gemini_action = "BUY"
        elif gemini_sentiment == "bearish":
            gemini_action = "SELL"
        
        # Agreement scoring
        agreement_bonus = 0.0
        if lstm_action == gemini_action:
            agreement_bonus = 0.15  # 15% bonus for agreement
        elif (lstm_action == "HOLD") or (gemini_action == "HOLD"):
            agreement_bonus = 0.05  # 5% bonus for partial agreement
        else:
            agreement_bonus = -0.10  # 10% penalty for disagreement
        
        # Weighted combination
        lstm_weight = 1.0 - self.gemini_weight
        gemini_weight = self.gemini_weight
        
        # Calculate weighted confidence
        base_confidence = (lstm_conf * lstm_weight) + (gemini_conf * gemini_weight)
        
        # Apply agreement bonus
        final_confidence = min(0.95, max(0.05, base_confidence + agreement_bonus))
        
        # Determine final action based on stronger signal
        if lstm_conf > gemini_conf:
            final_action = lstm_action
        elif gemini_conf > lstm_conf:
            final_action = gemini_action
        else:
            # Equal confidence - prefer LSTM for consistency
            final_action = lstm_action
        
        # Override for very low confidence
        if final_confidence < self.min_confidence_threshold:
            final_action = "HOLD"
        
        confidence_boost = agreement_bonus
        
        return final_action, final_confidence, confidence_boost
    
    def _analyze_trend(self, data: pd.DataFrame) -> str:
        """Analyze overall trend direction"""
        if len(data) < 20:
            return "insufficient_data"
        
        ma_5 = data['close'].rolling(5).mean().iloc[-1]
        ma_20 = data['close'].rolling(20).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        if current_price > ma_5 > ma_20:
            return "strong_uptrend"
        elif current_price > ma_5 and ma_5 > ma_20 * 0.999:
            return "uptrend"
        elif current_price < ma_5 < ma_20:
            return "strong_downtrend"
        elif current_price < ma_5 and ma_5 < ma_20 * 1.001:
            return "downtrend"
        else:
            return "sideways"
    
    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility level"""
        if volatility > 0.02:
            return "high"
        elif volatility > 0.01:
            return "medium"
        else:
            return "low"
    
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Find key support and resistance levels"""
        if len(data) < 50:
            return {"support": 0.0, "resistance": 0.0}
        
        recent_data = data.tail(50)
        support = float(recent_data['close'].min())
        resistance = float(recent_data['close'].max())
        current = float(data['close'].iloc[-1])
        
        return {
            "support": support,
            "resistance": resistance,
            "current": current,
            "support_distance": (current - support) / current,
            "resistance_distance": (resistance - current) / current
        }
    
    def execute_enhanced_trading(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Execute trading with enhanced AI analysis
        """
        print(f"\n" + "=" * 70)
        print(f"ENHANCED GEMINI TRADING EXECUTION")
        print("=" * 70)
        
        trading_results = []
        
        for pair, data in market_data.items():
            if len(data) < 100:
                print(f"[SKIP] {pair}: Insufficient data")
                continue
            
            # Comprehensive analysis
            analysis = self.analyze_market_with_gemini(data, pair)
            
            # Trading decision
            current_price = float(data['close'].iloc[-1])
            
            # Check for existing positions
            existing_position = None
            for pos_id, pos in self.paper_account.positions.items():
                if pos.pair == pair:
                    existing_position = pos
                    break
            
            # Enhanced trading logic
            action_taken = "NONE"
            
            if analysis.final_confidence >= self.min_confidence_threshold:
                
                if analysis.final_action == "BUY" and analysis.risk_level != "high":
                    if existing_position and existing_position.side == "SELL":
                        self.paper_account.close_position(existing_position.id, current_price, "AI_REVERSAL")
                        action_taken = "CLOSE_SELL"
                    
                    if not existing_position or existing_position.side != "BUY":
                        # Enhanced position sizing based on combined confidence
                        position_id = self.paper_account.open_position(
                            pair, "BUY", current_price, analysis.final_confidence,
                            stop_loss_pips=40, take_profit_pips=80
                        )
                        if position_id:
                            action_taken = "OPEN_BUY"
                
                elif analysis.final_action == "SELL" and analysis.risk_level != "high":
                    if existing_position and existing_position.side == "BUY":
                        self.paper_account.close_position(existing_position.id, current_price, "AI_REVERSAL")
                        action_taken = "CLOSE_BUY"
                    
                    if not existing_position or existing_position.side != "SELL":
                        position_id = self.paper_account.open_position(
                            pair, "SELL", current_price, analysis.final_confidence,
                            stop_loss_pips=40, take_profit_pips=80
                        )
                        if position_id:
                            action_taken = "OPEN_SELL"
                
                elif analysis.final_action == "HOLD" and existing_position:
                    # Consider closing if confidence is very low or risk is high
                    if analysis.final_confidence < 0.4 or analysis.risk_level == "high":
                        self.paper_account.close_position(existing_position.id, current_price, "LOW_CONFIDENCE")
                        action_taken = "CLOSE_LOW_CONF"
            
            # Update positions
            self.paper_account.update_positions({pair: current_price})
            
            # Record results
            trading_results.append({
                'pair': pair,
                'analysis': analysis,
                'action_taken': action_taken,
                'current_price': current_price
            })
        
        # Account summary
        self.paper_account.print_account_status()
        
        return {
            'results': trading_results,
            'account_summary': self.paper_account.get_account_summary()
        }
    
    def save_trading_session(self, results: Dict, filename: str = None):
        """Save detailed trading session results"""
        if filename is None:
            filename = f"enhanced_trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare serializable data
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'account_summary': results['account_summary'],
            'trading_results': []
        }
        
        for result in results['results']:
            analysis = result['analysis']
            session_data['trading_results'].append({
                'pair': result['pair'],
                'action_taken': result['action_taken'],
                'current_price': result['current_price'],
                'analysis': {
                    'lstm_action': analysis.lstm_action,
                    'lstm_confidence': analysis.lstm_confidence,
                    'gemini_sentiment': analysis.gemini_sentiment,
                    'gemini_confidence': analysis.gemini_confidence,
                    'gemini_reasoning': analysis.gemini_reasoning,
                    'final_action': analysis.final_action,
                    'final_confidence': analysis.final_confidence,
                    'confidence_boost': analysis.confidence_boost,
                    'risk_level': analysis.risk_level,
                    'trend_direction': analysis.trend_direction,
                    'volatility_level': analysis.volatility_level
                }
            })
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"\nSession saved to: {filename}")

def run_enhanced_gemini_demo():
    """Run enhanced Gemini trading demonstration"""
    
    print("FOREXSWING AI - ENHANCED GEMINI TRADING DEMO")
    print("=" * 70)
    
    # Initialize enhanced system
    trading_system = EnhancedGeminiTradingSystem(initial_balance=15000.0)
    
    # Load market data
    market_data = {}
    pairs = [
        ('EUR/USD', 'EUR_USD_real_daily.csv'),
        ('GBP/USD', 'GBP_USD_real_daily.csv'),
        ('USD/JPY', 'USD_JPY_real_daily.csv')
    ]
    
    for pair_name, filename in pairs:
        file_path = f"data/MarketData/{filename}"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            market_data[pair_name] = df.tail(120)  # Use last 120 days
            print(f"Loaded {len(df)} candles for {pair_name}")
    
    if not market_data:
        print("[ERROR] No market data available")
        return
    
    # Execute enhanced trading
    results = trading_system.execute_enhanced_trading(market_data)
    
    # Save results
    trading_system.save_trading_session(results)
    
    # Performance summary
    summary = results['account_summary']
    print(f"\n" + "=" * 70)
    print("ENHANCED GEMINI TRADING RESULTS")
    print("=" * 70)
    print(f"Return: {summary['return_pct']:+.2f}%")
    print(f"Total P&L: ${summary['total_pnl']:+.2f}")
    print(f"Win Rate: {summary['win_rate']:.1f}%")
    print(f"Open Positions: {summary['open_positions']}")
    print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")

if __name__ == "__main__":
    run_enhanced_gemini_demo()