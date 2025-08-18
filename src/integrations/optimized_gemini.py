#!/usr/bin/env python3
"""
Optimized Gemini CLI Integration for ForexSwing AI
High-performance market interpretation with caching and async processing
"""

import subprocess
import json
import asyncio
import threading
import time
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from functools import lru_cache
import os

class OptimizedGeminiInterpreter:
    """
    High-performance Gemini CLI integration with optimization features:
    - Response caching
    - Async processing
    - Batch requests
    - Intelligent prompting
    """
    
    def __init__(self, cache_size: int = 100, cache_duration_minutes: int = 30):
        self.gemini_available = self._check_gemini_cli()
        self.cache_size = cache_size
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        
        # Response cache with timestamps
        self.response_cache = {}
        self.cache_timestamps = {}
        
        # Performance tracking
        self.call_count = 0
        self.cache_hits = 0
        self.total_response_time = 0
        
        # Rate limiting
        self.last_call_time = 0
        self.min_call_interval = 1.0  # seconds
        
        print(f"OptimizedGeminiInterpreter initialized:")
        print(f"  - Gemini CLI: {'Available' if self.gemini_available else 'Not available'}")
        print(f"  - Cache size: {cache_size} entries")
        print(f"  - Cache duration: {cache_duration_minutes} minutes")
    
    def _check_gemini_cli(self) -> bool:
        """Check if Gemini CLI is available with timeout"""
        try:
            result = subprocess.run(
                ['npx', '@google/gemini-cli', '--version'], 
                capture_output=True, text=True, timeout=5, shell=True
            )
            return result.returncode == 0
        except:
            return False
    
    def _generate_cache_key(self, prompt: str, context_data: str = "") -> str:
        """Generate cache key for request"""
        combined = f"{prompt}|{context_data}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached response is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        timestamp = self.cache_timestamps[cache_key]
        return datetime.now() - timestamp < self.cache_duration
    
    def _call_gemini_raw(self, prompt: str, timeout: int = 20) -> Optional[str]:
        """Raw Gemini CLI call with rate limiting"""
        if not self.gemini_available:
            return None
        
        # Rate limiting
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.min_call_interval:
            time.sleep(self.min_call_interval - time_since_last_call)
        
        try:
            start_time = time.time()
            
            result = subprocess.run(
                ['npx', '@google/gemini-cli', '--prompt', prompt],
                capture_output=True, text=True, timeout=timeout, shell=True
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            self.call_count += 1
            self.total_response_time += response_time
            self.last_call_time = end_time
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Gemini CLI error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("Gemini CLI timeout")
            return None
        except Exception as e:
            print(f"Gemini CLI error: {e}")
            return None
    
    def _call_gemini_cached(self, prompt: str, context_data: str = "", force_refresh: bool = False) -> Optional[str]:
        """Call Gemini with caching"""
        cache_key = self._generate_cache_key(prompt, context_data)
        
        # Check cache first
        if not force_refresh and cache_key in self.response_cache and self._is_cache_valid(cache_key):
            self.cache_hits += 1
            return self.response_cache[cache_key]
        
        # Call Gemini
        response = self._call_gemini_raw(prompt)
        
        if response:
            # Cache the response
            self.response_cache[cache_key] = response
            self.cache_timestamps[cache_key] = datetime.now()
            
            # Clean old cache entries if needed
            if len(self.response_cache) > self.cache_size:
                self._clean_cache()
        
        return response
    
    def _clean_cache(self):
        """Remove oldest cache entries"""
        # Sort by timestamp and remove oldest
        sorted_keys = sorted(self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k])
        
        # Remove oldest 20% of entries
        remove_count = max(1, len(sorted_keys) // 5)
        for key in sorted_keys[:remove_count]:
            del self.response_cache[key]
            del self.cache_timestamps[key]
    
    def interpret_market_quickly(self, price_data: Dict, pair: str) -> Dict:
        """
        Fast market interpretation with optimized prompting
        """
        if not self.gemini_available:
            return {"available": False, "message": "Gemini CLI not available"}
        
        try:
            # Create concise context
            context = {
                "pair": pair,
                "price": f"{price_data.get('current_price', 0):.5f}",
                "change": f"{price_data.get('price_change_24h', 0):+.2f}%",
                "trend": price_data.get('trend', 'neutral'),
                "volatility": price_data.get('volatility', 0)
            }
            
            # Optimized prompt for faster response
            prompt = f"""Quick forex analysis for {pair}:
Price: {context['price']} ({context['change']})
Trend: {context['trend']}

Respond with JSON: {{"sentiment": "bullish/bearish/neutral", "confidence": 0-100, "key_factor": "brief reason"}}"""
            
            response = self._call_gemini_cached(prompt, json.dumps(context))
            
            if response:
                try:
                    interpretation = json.loads(response)
                    interpretation["timestamp"] = datetime.now().isoformat()
                    interpretation["pair"] = pair
                    interpretation["cached"] = cache_key in self.response_cache
                    return interpretation
                except json.JSONDecodeError:
                    return {
                        "sentiment": "neutral",
                        "confidence": 50,
                        "key_factor": "parsing_error",
                        "raw_response": response[:100],
                        "timestamp": datetime.now().isoformat(),
                        "pair": pair
                    }
            
            return {"error": "No response from Gemini"}
            
        except Exception as e:
            return {"error": f"Market interpretation failed: {e}"}
    
    def validate_signal_fast(self, signal_data: Dict) -> Dict:
        """
        Fast signal validation with minimal prompting
        """
        if not self.gemini_available:
            return {"validation": "unavailable", "confidence": 0.5}
        
        try:
            # Simplified validation prompt
            ml_prediction = signal_data.get('ml_prediction', 0.5)
            ml_confidence = signal_data.get('ml_confidence', 0.5)
            
            action = "BUY" if ml_prediction > 0.6 else "SELL" if ml_prediction < 0.4 else "HOLD"
            
            prompt = f"""Forex signal check:
Action: {action} (confidence: {ml_confidence:.1%})
RSI: {signal_data.get('rsi', 50):.0f}
Trend: {signal_data.get('trend_direction', 'neutral')}

Quick assessment - JSON: {{"valid": true/false, "risk": "low/medium/high", "adjust": "none/reduce/increase"}}"""
            
            response = self._call_gemini_cached(prompt, json.dumps(signal_data))
            
            if response:
                try:
                    validation = json.loads(response)
                    
                    # Convert to expected format
                    return {
                        "gemini_validation": validation,
                        "enhanced_confidence": ml_confidence * (1.1 if validation.get('valid', True) else 0.9),
                        "risk_level": validation.get('risk', 'medium'),
                        "timestamp": datetime.now().isoformat()
                    }
                except json.JSONDecodeError:
                    return {
                        "validation": "error",
                        "enhanced_confidence": ml_confidence,
                        "raw_response": response[:50],
                        "timestamp": datetime.now().isoformat()
                    }
            
            return {"validation": "no_response", "enhanced_confidence": ml_confidence}
            
        except Exception as e:
            return {"validation": "error", "message": str(e)}
    
    def batch_analyze_scenarios(self, scenarios: List[Dict]) -> List[Dict]:
        """
        Batch analyze multiple market scenarios efficiently
        """
        if not self.gemini_available:
            return [{"error": "Gemini unavailable"}] * len(scenarios)
        
        results = []
        
        # Group similar scenarios to optimize caching
        for scenario in scenarios:
            try:
                # Quick analysis for each scenario
                result = self.interpret_market_quickly(scenario, scenario.get('pair', 'UNKNOWN'))
                results.append(result)
                
                # Small delay between batch requests
                time.sleep(0.1)
                
            except Exception as e:
                results.append({"error": str(e)})
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_response_time = (self.total_response_time / self.call_count) if self.call_count > 0 else 0
        cache_hit_rate = (self.cache_hits / self.call_count) if self.call_count > 0 else 0
        
        return {
            "total_calls": self.call_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{cache_hit_rate:.1%}",
            "avg_response_time": f"{avg_response_time:.2f}s",
            "cache_entries": len(self.response_cache),
            "gemini_available": self.gemini_available
        }
    
    def clear_cache(self):
        """Clear all cached responses"""
        self.response_cache.clear()
        self.cache_timestamps.clear()
        print("Cache cleared")

class FastForexStrategy:
    """
    Fast trading strategy combining optimized LSTM + Gemini
    """
    
    def __init__(self, model_path: str = None):
        # Import here to avoid circular imports
        from test_optimized_model import SimpleOptimizedLSTM, create_simple_features
        
        self.model = SimpleOptimizedLSTM(input_size=20, hidden_size=128, num_layers=3, dropout=0.4)
        self.create_features = create_simple_features
        self.gemini = OptimizedGeminiInterpreter(cache_size=50, cache_duration_minutes=15)
        
        # Load model
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(checkpoint, strict=False)
                self.model.eval()
                print("Model loaded successfully")
            except Exception as e:
                print(f"Model loading failed: {e}")
        
        # Performance settings
        self.enable_gemini = True
        self.gemini_weight = 0.2  # Reduced weight for speed
        
    def get_fast_recommendation(self, dataframe: pd.DataFrame, pair: str) -> Dict:
        """
        Generate fast trading recommendation
        """
        start_time = time.time()
        
        try:
            # Fast LSTM prediction
            features = self.create_features(dataframe, target_features=20)
            
            if len(features) >= 80:
                sequence = features[-80:]
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                
                with torch.no_grad():
                    output = self.model(sequence_tensor)
                    probs = output[0].numpy()
                    
                    # Quick signal determination
                    if probs[1] > 0.45:  # Adjusted threshold
                        ml_signal = "BUY"
                        ml_confidence = float(probs[1])
                    elif probs[2] > 0.45:
                        ml_signal = "SELL"
                        ml_confidence = float(probs[2])
                    else:
                        ml_signal = "HOLD"
                        ml_confidence = float(probs[0])
            else:
                ml_signal = "HOLD"
                ml_confidence = 0.5
            
            # Fast Gemini validation (if enabled)
            gemini_result = {"confidence_adjustment": 1.0, "risk_level": "medium"}
            
            if self.enable_gemini and self.gemini.gemini_available:
                try:
                    # Quick market context
                    latest_price = float(dataframe['close'].iloc[-1])
                    prev_price = float(dataframe['close'].iloc[-2])
                    price_change = (latest_price - prev_price) / prev_price * 100
                    
                    market_context = {
                        "current_price": latest_price,
                        "price_change_24h": price_change,
                        "trend": "bullish" if price_change > 0 else "bearish" if price_change < 0 else "neutral",
                        "volatility": float(dataframe['close'].rolling(10).std().iloc[-1])
                    }
                    
                    gemini_analysis = self.gemini.interpret_market_quickly(market_context, pair)
                    
                    if "confidence" in gemini_analysis:
                        gemini_confidence = gemini_analysis["confidence"] / 100.0
                        confidence_adjustment = 0.8 + 0.4 * gemini_confidence  # 0.8 to 1.2 range
                        gemini_result["confidence_adjustment"] = confidence_adjustment
                    
                except Exception as e:
                    print(f"Gemini analysis error: {e}")
            
            # Combine results
            final_confidence = ml_confidence * gemini_result["confidence_adjustment"]
            final_confidence = min(max(final_confidence, 0.1), 0.95)  # Clamp to reasonable range
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            return {
                "pair": pair,
                "action": ml_signal,
                "confidence": final_confidence,
                "processing_time": f"{processing_time:.2f}s",
                "ml_signal": ml_signal,
                "ml_confidence": ml_confidence,
                "gemini_enhanced": self.enable_gemini and self.gemini.gemini_available,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            return {
                "pair": pair,
                "action": "HOLD",
                "confidence": 0.5,
                "processing_time": f"{processing_time:.2f}s",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_performance_stats(self) -> Dict:
        """Get combined performance statistics"""
        return {
            "strategy": "FastForexStrategy",
            "gemini_stats": self.gemini.get_performance_stats(),
            "gemini_enabled": self.enable_gemini
        }

# Test function
def test_optimized_gemini():
    """Test optimized Gemini integration"""
    print("TESTING OPTIMIZED GEMINI INTEGRATION")
    print("=" * 50)
    
    # Test basic functionality
    gemini = OptimizedGeminiInterpreter(cache_size=10, cache_duration_minutes=5)
    
    print(f"Gemini available: {gemini.gemini_available}")
    
    if gemini.gemini_available:
        # Test quick market interpretation
        test_data = {
            "current_price": 1.0850,
            "price_change_24h": 0.15,
            "trend": "bullish",
            "volatility": 0.012
        }
        
        print("\nTesting quick market interpretation...")
        start_time = time.time()
        result1 = gemini.interpret_market_quickly(test_data, "EUR/USD")
        end_time = time.time()
        
        print(f"First call: {end_time - start_time:.2f}s")
        print(f"Result: {result1}")
        
        # Test caching
        print("\nTesting cache performance...")
        start_time = time.time()
        result2 = gemini.interpret_market_quickly(test_data, "EUR/USD")
        end_time = time.time()
        
        print(f"Cached call: {end_time - start_time:.2f}s")
        print(f"Cache hit: {result2.get('cached', False)}")
        
        # Performance stats
        stats = gemini.get_performance_stats()
        print(f"\nPerformance stats: {stats}")
    
    # Test fast strategy
    print(f"\n" + "="*50)
    print("TESTING FAST STRATEGY")
    print("="*50)
    
    try:
        import torch
        strategy = FastForexStrategy("data/models/optimized_forex_ai.pth")
        
        # Create test data
        test_df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 1.0850,
            'volume': np.random.randint(50000, 200000, 100),
            'high': np.random.randn(100) * 0.002 + 1.0850,
            'low': np.random.randn(100) * 0.002 + 1.0850,
        })
        
        print("Testing fast recommendation...")
        recommendation = strategy.get_fast_recommendation(test_df, "EUR/USD")
        
        print(f"Recommendation: {recommendation}")
        
        strategy_stats = strategy.get_performance_stats()
        print(f"Strategy stats: {strategy_stats}")
        
    except Exception as e:
        print(f"Strategy test failed: {e}")

if __name__ == "__main__":
    test_optimized_gemini()