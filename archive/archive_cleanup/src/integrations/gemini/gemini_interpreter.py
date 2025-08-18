#!/usr/bin/env python3
"""
Gemini CLI Integration for ForexSwing AI 2025
Enhances trading decisions with Google's Gemini AI interpretation
"""

import subprocess
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import sys
from datetime import datetime

class GeminiDataInterpreter:
    """
    Integrates Gemini CLI to provide advanced market data interpretation
    """
    
    def __init__(self):
        self.gemini_available = self._check_gemini_cli()
        self.interpretation_cache = {}
        
    def _check_gemini_cli(self) -> bool:
        """Check if Gemini CLI is available"""
        try:
            result = subprocess.run(['npx', '@google/gemini-cli', '--version'], 
                                  capture_output=True, text=True, timeout=10, shell=True)
            return result.returncode == 0
        except:
            return False
    
    def _call_gemini(self, prompt: str, context_data: str = "") -> Optional[str]:
        """Call Gemini CLI with trading context"""
        if not self.gemini_available:
            return None
            
        try:
            # Prepare full prompt with forex trading context
            full_prompt = f"""
            You are an expert forex market analyst. Analyze the following market data and provide insights:
            
            Context: {context_data}
            
            Question: {prompt}
            
            Provide a concise analysis focusing on:
            1. Market sentiment interpretation
            2. Key patterns or anomalies
            3. Risk factors
            4. Confidence level (1-10)
            
            Response format: JSON with fields: sentiment, patterns, risks, confidence, reasoning
            """
            
            # Call Gemini CLI
            result = subprocess.run(
                ['npx', '@google/gemini-cli', '--prompt', full_prompt],
                capture_output=True, text=True, timeout=30, shell=True
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Gemini CLI error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return None
    
    def interpret_market_data(self, dataframe: pd.DataFrame, pair: str) -> Dict:
        """
        Interpret market data using Gemini's advanced reasoning
        """
        if not self.gemini_available:
            return {"available": False, "message": "Gemini CLI not available"}
        
        try:
            # Prepare market context
            recent_data = dataframe.tail(20)
            
            market_context = {
                "pair": pair,
                "current_price": float(recent_data['close'].iloc[-1]),
                "price_change_24h": float((recent_data['close'].iloc[-1] - recent_data['close'].iloc[-2]) / recent_data['close'].iloc[-2] * 100),
                "volatility": float(recent_data['close'].std()),
                "volume_trend": "increasing" if recent_data['volume'].iloc[-1] > recent_data['volume'].mean() else "decreasing",
                "rsi": float(recent_data['rsi'].iloc[-1]) if 'rsi' in recent_data.columns else None,
                "trend": "bullish" if recent_data['close'].iloc[-1] > recent_data['close'].rolling(5).mean().iloc[-1] else "bearish"
            }
            
            context_str = json.dumps(market_context, indent=2)
            
            prompt = f"""
            Analyze this {pair} forex market data. What are the key insights for trading decisions?
            Consider technical patterns, volume analysis, and market structure.
            """
            
            response = self._call_gemini(prompt, context_str)
            
            if response:
                try:
                    # Try to parse JSON response
                    interpretation = json.loads(response)
                    interpretation["timestamp"] = datetime.now().isoformat()
                    interpretation["pair"] = pair
                    return interpretation
                except json.JSONDecodeError:
                    # Fallback to text response
                    return {
                        "raw_analysis": response,
                        "timestamp": datetime.now().isoformat(),
                        "pair": pair
                    }
            
            return {"error": "No response from Gemini"}
            
        except Exception as e:
            return {"error": f"Market interpretation failed: {e}"}
    
    def validate_trading_signal(self, 
                              ml_prediction: float, 
                              technical_indicators: Dict, 
                              market_context: Dict) -> Dict:
        """
        Use Gemini to validate and enhance trading signals
        """
        if not self.gemini_available:
            return {"validation": "unavailable", "confidence": 0.5}
        
        try:
            # Prepare validation context
            signal_data = {
                "ml_prediction": float(ml_prediction),
                "ml_confidence": float(technical_indicators.get('ml_confidence', 0.5)),
                "rsi": float(technical_indicators.get('rsi', 50)),
                "macd_signal": "bullish" if technical_indicators.get('macd', 0) > technical_indicators.get('macd_signal', 0) else "bearish",
                "trend_alignment": bool(technical_indicators.get('trend_alignment', False)),
                "volatility": float(technical_indicators.get('volatility', 0.01)),
                "volume_ratio": float(technical_indicators.get('volume_ratio', 1.0))
            }
            
            context_str = json.dumps(signal_data, indent=2)
            
            prompt = f"""
            Review this forex trading signal for accuracy and risk:
            
            ML Prediction: {ml_prediction:.3f} ({'BUY' if ml_prediction > 0.6 else 'SELL' if ml_prediction < 0.4 else 'HOLD'})
            
            Should I trust this signal? What are the risks? Rate confidence 1-10.
            """
            
            response = self._call_gemini(prompt, context_str)
            
            if response:
                try:
                    validation = json.loads(response)
                    return {
                        "gemini_validation": validation,
                        "enhanced_confidence": validation.get('confidence', 5) / 10.0,
                        "risk_assessment": validation.get('risks', []),
                        "timestamp": datetime.now().isoformat()
                    }
                except json.JSONDecodeError:
                    return {
                        "gemini_analysis": response,
                        "timestamp": datetime.now().isoformat()
                    }
            
            return {"validation": "no_response"}
            
        except Exception as e:
            return {"validation": "error", "message": str(e)}
    
    def analyze_market_anomalies(self, dataframe: pd.DataFrame, pair: str) -> Dict:
        """
        Detect and interpret market anomalies using Gemini
        """
        if not self.gemini_available:
            return {"anomalies": [], "analysis": "Gemini unavailable"}
        
        try:
            # Detect potential anomalies
            recent_data = dataframe.tail(50)
            
            anomalies = []
            
            # Price spike detection
            price_changes = recent_data['close'].pct_change()
            large_moves = price_changes[abs(price_changes) > 0.02]  # >2% moves
            
            if len(large_moves) > 0:
                anomalies.append({
                    "type": "price_spike",
                    "magnitude": float(large_moves.iloc[-1]),
                    "timestamp": str(large_moves.index[-1])
                })
            
            # Volume anomalies
            volume_mean = recent_data['volume'].rolling(20).mean()
            volume_spikes = recent_data[recent_data['volume'] > volume_mean * 2]
            
            if len(volume_spikes) > 0:
                anomalies.append({
                    "type": "volume_spike",
                    "ratio": float(volume_spikes['volume'].iloc[-1] / volume_mean.iloc[-1]),
                    "timestamp": str(volume_spikes.index[-1])
                })
            
            if anomalies:
                anomaly_context = json.dumps(anomalies, indent=2)
                
                prompt = f"""
                Unusual market activity detected in {pair}:
                
                {anomaly_context}
                
                What could be causing these anomalies? How should traders respond?
                """
                
                response = self._call_gemini(prompt)
                
                return {
                    "anomalies": anomalies,
                    "gemini_interpretation": response,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {"anomalies": [], "analysis": "No significant anomalies detected"}
            
        except Exception as e:
            return {"error": f"Anomaly analysis failed: {e}"}
    
    def generate_trading_commentary(self, 
                                  pair: str,
                                  prediction: str,
                                  confidence: float,
                                  market_data: Dict) -> str:
        """
        Generate human-readable trading commentary using Gemini
        """
        if not self.gemini_available:
            return f"Basic signal: {prediction} for {pair} (confidence: {confidence:.1%})"
        
        try:
            context = json.dumps(market_data, indent=2)
            
            prompt = f"""
            Generate a brief, professional trading commentary for:
            
            Pair: {pair}
            Signal: {prediction} 
            Confidence: {confidence:.1%}
            
            Market Data: {context}
            
            Write 2-3 sentences explaining the reasoning and key factors.
            """
            
            response = self._call_gemini(prompt)
            
            if response:
                return response.strip()
            else:
                return f"AI recommends {prediction} for {pair} with {confidence:.1%} confidence based on technical analysis."
                
        except Exception as e:
            return f"Signal: {prediction} for {pair} (confidence: {confidence:.1%})"

# Test the integration
def test_gemini_integration():
    """Test Gemini CLI integration"""
    print("Testing Gemini CLI Integration...")
    
    interpreter = GeminiDataInterpreter()
    
    if not interpreter.gemini_available:
        print("⚠️ Gemini CLI not available")
        print("Install with: npm install -g @google/gemini-cli")
        return False
    
    print("✅ Gemini CLI available")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 1.1000,
        'volume': np.random.randint(1000, 5000, 100),
        'rsi': np.random.uniform(30, 70, 100)
    })
    
    # Test market interpretation
    print("\nTesting market interpretation...")
    result = interpreter.interpret_market_data(sample_data, "EUR/USD")
    print(f"Result: {result}")
    
    return True

if __name__ == "__main__":
    test_gemini_integration()