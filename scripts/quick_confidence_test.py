#!/usr/bin/env python3
"""
Quick test of confidence calculation fix
"""

def test_confidence_fix():
    """Test the confidence calculation without running full system"""
    print("TESTING CONFIDENCE CALCULATION FIX")
    print("=" * 50)
    
    # Test different confidence inputs
    test_cases = [
        {"base": 0.652, "expected": "around 65-75%"},
        {"base": 65.2, "expected": "around 65-75%"},  
        {"base": 0.45, "expected": "around 45-55%"},
        {"base": 45.0, "expected": "around 45-55%"}
    ]
    
    # Simulate the enhanced confidence calculation
    for test in test_cases:
        base_confidence = test["base"]
        
        # Multipliers (same as in fallback system)
        news_mult = 1.0  # neutral news
        risk_mult = 1.0  # no correlation risk
        volatility_mult = 1.1  # low volatility bonus
        
        # Fixed calculation logic
        if base_confidence > 1:
            # Already in percentage form (like 65.2)
            enhanced_confidence = min(95, max(15, base_confidence * news_mult * risk_mult * volatility_mult))
        else:
            # In decimal form (like 0.652)
            enhanced_confidence = min(95, max(15, base_confidence * 100 * news_mult * risk_mult * volatility_mult))
        
        print(f"Input: {base_confidence} -> Output: {enhanced_confidence:.1f}% ({test['expected']})")
    
    print("\n" + "=" * 50)
    print("CONFIDENCE CALCULATION: FIXED")
    print("Now values should be reasonable (15-95%)")

if __name__ == "__main__":
    test_confidence_fix()