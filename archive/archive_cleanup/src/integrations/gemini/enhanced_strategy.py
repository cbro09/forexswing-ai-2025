#!/usr/bin/env python3
"""
Enhanced ForexSwing Strategy with Gemini AI Integration
Combines 55.2% accurate LSTM with Gemini's advanced reasoning
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import sys
import os

# Import existing components
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.forex_lstm import HybridForexPredictor
from integrations.gemini_data_interpreter import GeminiDataInterpreter

class EnhancedForexStrategy:
    """
    Enhanced strategy combining:
    1. Optimized LSTM (55.2% accuracy)
    2. JAX-accelerated indicators (65K+ calc/sec)
    3. Gemini AI interpretation
    """
    
    def __init__(self, model_path: str = None):
        # Initialize core AI predictor
        self.ml_predictor = HybridForexPredictor(model_path)
        
        # Initialize Gemini interpreter
        self.gemini_interpreter = GeminiDataInterpreter()
        
        # Strategy parameters
        self.ml_threshold_buy = 0.65
        self.ml_threshold_sell = 0.35
        self.confidence_threshold = 0.70
        self.gemini_weight = 0.3  # Weight for Gemini consensus
        
        print(f"Enhanced Strategy Initialized:")
        print(f"  - LSTM Model: {'Loaded' if self.ml_predictor.is_trained else 'Untrained'}")
        print(f"  - Gemini AI: {'Available' if self.gemini_interpreter.gemini_available else 'Unavailable'}")
    
    def analyze_market(self, dataframe: pd.DataFrame, pair: str) -> Dict:
        """
        Comprehensive market analysis using both AI systems
        """
        analysis = {
            "pair": pair,
            "timestamp": pd.Timestamp.now().isoformat(),
            "ml_analysis": {},
            "gemini_analysis": {},
            "combined_signal": {}
        }
        
        # 1. Get LSTM predictions
        try:
            ml_predictions = self.ml_predictor.predict(dataframe)
            latest_ml_signal = ml_predictions[-1] if len(ml_predictions) > 0 else 0.5
            
            analysis["ml_analysis"] = {
                "signal": float(latest_ml_signal),
                "prediction": self._interpret_ml_signal(latest_ml_signal),
                "confidence": float(abs(latest_ml_signal - 0.5) * 2),
                "model_info": self.ml_predictor.get_model_info()
            }
            
        except Exception as e:
            analysis["ml_analysis"] = {"error": str(e)}
            latest_ml_signal = 0.5
        
        # 2. Get Gemini interpretation
        if self.gemini_interpreter.gemini_available:
            try:
                gemini_result = self.gemini_interpreter.interpret_market_data(dataframe, pair)
                analysis["gemini_analysis"] = gemini_result
                
                # Validate ML signal with Gemini
                technical_indicators = self._extract_technical_indicators(dataframe)
                validation = self.gemini_interpreter.validate_trading_signal(
                    latest_ml_signal, technical_indicators, {}
                )
                analysis["gemini_validation"] = validation
                
            except Exception as e:
                analysis["gemini_analysis"] = {"error": str(e)}
        
        # 3. Combine signals for final decision
        combined_signal = self._combine_signals(analysis)
        analysis["combined_signal"] = combined_signal
        
        return analysis
    
    def _interpret_ml_signal(self, signal: float) -> str:
        """Convert ML signal to trading action"""
        if signal >= self.ml_threshold_buy:
            return "BUY"
        elif signal <= self.ml_threshold_sell:
            return "SELL"
        else:
            return "HOLD"
    
    def _extract_technical_indicators(self, dataframe: pd.DataFrame) -> Dict:
        """Extract technical indicators for Gemini validation"""
        if len(dataframe) == 0:
            return {}
        
        latest_row = dataframe.iloc[-1]
        
        return {
            "rsi": latest_row.get('rsi', 50),
            "macd": latest_row.get('macd', 0),
            "macd_signal": latest_row.get('macd_signal', 0),
            "ml_confidence": latest_row.get('ml_confidence', 0.5),
            "trend_alignment": latest_row.get('trend_alignment', False),
            "volatility": latest_row.get('volatility', 0.01),
            "volume_ratio": latest_row.get('volume_ratio', 1.0)
        }
    
    def _combine_signals(self, analysis: Dict) -> Dict:
        """
        Combine LSTM and Gemini signals for enhanced decision making
        """
        ml_analysis = analysis.get("ml_analysis", {})
        gemini_analysis = analysis.get("gemini_analysis", {})
        gemini_validation = analysis.get("gemini_validation", {})
        
        # Base signal from ML
        ml_signal = ml_analysis.get("signal", 0.5)
        ml_confidence = ml_analysis.get("confidence", 0.5)
        ml_prediction = ml_analysis.get("prediction", "HOLD")
        
        # Gemini enhancement
        gemini_confidence = gemini_validation.get("enhanced_confidence", 0.5)
        gemini_risks = gemini_validation.get("risk_assessment", [])
        
        # Combined confidence calculation
        if self.gemini_interpreter.gemini_available and gemini_confidence > 0:
            # Weighted average of ML and Gemini confidence
            combined_confidence = (
                ml_confidence * (1 - self.gemini_weight) + 
                gemini_confidence * self.gemini_weight
            )
        else:
            combined_confidence = ml_confidence
        
        # Risk adjustment
        risk_penalty = min(len(gemini_risks) * 0.1, 0.3)  # Max 30% penalty
        adjusted_confidence = max(combined_confidence - risk_penalty, 0.1)
        
        # Final signal decision
        final_prediction = ml_prediction
        
        # Override if confidence is too low
        if adjusted_confidence < self.confidence_threshold:
            final_prediction = "HOLD"
        
        return {
            "prediction": final_prediction,
            "confidence": float(adjusted_confidence),
            "ml_signal": float(ml_signal),
            "ml_confidence": float(ml_confidence),
            "gemini_confidence": float(gemini_confidence),
            "risk_factors": len(gemini_risks),
            "reasoning": self._generate_reasoning(ml_analysis, gemini_analysis, final_prediction)
        }
    
    def _generate_reasoning(self, ml_analysis: Dict, gemini_analysis: Dict, final_prediction: str) -> str:
        """Generate human-readable reasoning for the trading decision"""
        
        ml_pred = ml_analysis.get("prediction", "UNKNOWN")
        ml_conf = ml_analysis.get("confidence", 0)
        
        reasoning = f"LSTM predicts {ml_pred} with {ml_conf:.1%} confidence. "
        
        if self.gemini_interpreter.gemini_available:
            gemini_sentiment = gemini_analysis.get("sentiment", "neutral")
            reasoning += f"Gemini analysis shows {gemini_sentiment} market sentiment. "
        
        reasoning += f"Final decision: {final_prediction}"
        
        return reasoning
    
    def get_trading_recommendation(self, dataframe: pd.DataFrame, pair: str) -> Dict:
        """
        Get complete trading recommendation with reasoning
        """
        # Full market analysis
        analysis = self.analyze_market(dataframe, pair)
        
        # Extract final recommendation
        combined_signal = analysis["combined_signal"]
        
        recommendation = {
            "pair": pair,
            "action": combined_signal["prediction"],
            "confidence": combined_signal["confidence"],
            "reasoning": combined_signal["reasoning"],
            "risk_level": "High" if combined_signal["risk_factors"] > 2 else "Medium" if combined_signal["risk_factors"] > 0 else "Low",
            "timestamp": analysis["timestamp"]
        }
        
        # Add position sizing suggestion
        if combined_signal["prediction"] != "HOLD":
            position_size = self._calculate_position_size(combined_signal["confidence"])
            recommendation["suggested_position_size"] = position_size
        
        # Add commentary if Gemini is available
        if self.gemini_interpreter.gemini_available:
            try:
                market_data = self._extract_technical_indicators(dataframe)
                commentary = self.gemini_interpreter.generate_trading_commentary(
                    pair, combined_signal["prediction"], combined_signal["confidence"], market_data
                )
                recommendation["commentary"] = commentary
            except:
                recommendation["commentary"] = "Commentary unavailable"
        
        return recommendation
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence"""
        # Conservative position sizing: 0.5% to 2% of account
        base_size = 0.01  # 1% base position
        confidence_multiplier = min(confidence * 2, 2.0)  # Max 2x multiplier
        
        return base_size * confidence_multiplier
    
    def monitor_anomalies(self, dataframe: pd.DataFrame, pair: str) -> Dict:
        """Monitor for market anomalies using Gemini"""
        if self.gemini_interpreter.gemini_available:
            return self.gemini_interpreter.analyze_market_anomalies(dataframe, pair)
        else:
            return {"anomalies": [], "analysis": "Monitoring unavailable - Gemini not available"}

# Example usage and testing
def test_enhanced_strategy():
    """Test the enhanced strategy"""
    print("Testing Enhanced ForexSwing Strategy...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'close': np.random.randn(200).cumsum() + 1.1000,
        'volume': np.random.randint(1000, 5000, 200),
        'high': np.random.randn(200) * 0.001 + 1.1000,
        'low': np.random.randn(200) * 0.001 + 1.1000,
        'rsi': np.random.uniform(30, 70, 200),
        'macd': np.random.randn(200) * 0.0001,
        'macd_signal': np.random.randn(200) * 0.0001,
    })
    
    # Initialize strategy
    strategy = EnhancedForexStrategy()
    
    # Get trading recommendation
    recommendation = strategy.get_trading_recommendation(sample_data, "EUR/USD")
    
    print(f"\nTrading Recommendation:")
    print(f"  Pair: {recommendation['pair']}")
    print(f"  Action: {recommendation['action']}")
    print(f"  Confidence: {recommendation['confidence']:.1%}")
    print(f"  Risk Level: {recommendation['risk_level']}")
    print(f"  Reasoning: {recommendation['reasoning']}")
    
    if 'suggested_position_size' in recommendation:
        print(f"  Position Size: {recommendation['suggested_position_size']:.2%}")
    
    if 'commentary' in recommendation:
        print(f"  Commentary: {recommendation['commentary']}")
    
    # Test anomaly detection
    anomalies = strategy.monitor_anomalies(sample_data, "EUR/USD")
    print(f"\nAnomaly Detection: {len(anomalies.get('anomalies', []))} anomalies found")
    
    return strategy

if __name__ == "__main__":
    test_enhanced_strategy()