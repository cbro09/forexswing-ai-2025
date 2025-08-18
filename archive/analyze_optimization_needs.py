#!/usr/bin/env python3
"""
Analyze Optimization Needs for ForexSwing AI 2025
Identify performance gaps and optimization opportunities
"""

import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import time

def analyze_model_compatibility():
    """Check if current model architecture matches trained weights"""
    print("ANALYZING MODEL COMPATIBILITY")
    print("=" * 50)
    
    model_path = "data/models/optimized_forex_ai.pth"
    
    if not os.path.exists(model_path):
        print("[ERROR] Optimized model not found")
        return False
    
    try:
        # Load the saved model state
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"[OK] Model file loaded: {os.path.getsize(model_path):,} bytes")
        
        # Analyze model structure
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("[INFO] Model has checkpoint structure")
            else:
                state_dict = checkpoint
                print("[INFO] Model is raw state dict")
        else:
            print("[WARNING] Unexpected model format")
            return False
        
        # Analyze layer structure
        layers = list(state_dict.keys())
        print(f"[INFO] Model has {len(layers)} parameter layers")
        
        # Check for known layer patterns
        has_lstm = any('lstm' in key for key in layers)
        has_attention = any('attention' in key for key in layers)
        has_classifier = any('classifier' in key or 'fc' in key for key in layers)
        
        print(f"[INFO] Architecture analysis:")
        print(f"  - LSTM layers: {'Yes' if has_lstm else 'No'}")
        print(f"  - Attention mechanism: {'Yes' if has_attention else 'No'}")
        print(f"  - Classification head: {'Yes' if has_classifier else 'No'}")
        
        # Check parameter count
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"[INFO] Total parameters: {total_params:,}")
        
        # Expected vs actual
        expected_params = 397519  # From documentation
        if abs(total_params - expected_params) < 1000:
            print(f"[OK] Parameter count matches expected ({expected_params:,})")
            return True
        else:
            print(f"[WARNING] Parameter mismatch - Expected: {expected_params:,}, Got: {total_params:,}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Model analysis failed: {e}")
        return False

def analyze_performance_gaps():
    """Analyze current performance vs targets"""
    print("\nANALYZING PERFORMANCE GAPS")
    print("=" * 50)
    
    # Current documented performance
    current_performance = {
        "accuracy": 55.2,  # %
        "processing_time": 12.3,  # seconds per recommendation
        "signal_bias": "100% HOLD",  # From recent tests
        "gemini_integration": "Working but slow",
        "model_loading": "Architecture mismatch"
    }
    
    # Target performance for production
    target_performance = {
        "accuracy": 60.0,  # Target institutional level
        "processing_time": 5.0,  # Max acceptable for live trading
        "signal_bias": "Balanced HOLD/BUY/SELL",
        "gemini_integration": "Optimized and fast",
        "model_loading": "Seamless compatibility"
    }
    
    print("Current vs Target Performance:")
    print("-" * 40)
    
    gaps_identified = []
    
    for metric in current_performance:
        current = current_performance[metric]
        target = target_performance[metric]
        
        if metric == "accuracy":
            gap = target - current
            print(f"Accuracy: {current}% → {target}% (gap: {gap:+.1f}%)")
            if gap > 0:
                gaps_identified.append(f"Accuracy needs +{gap:.1f}% improvement")
        
        elif metric == "processing_time":
            ratio = current / target
            print(f"Processing Time: {current}s → {target}s (ratio: {ratio:.1f}x)")
            if ratio > 1:
                gaps_identified.append(f"Speed needs {ratio:.1f}x improvement")
        
        else:
            print(f"{metric.replace('_', ' ').title()}: {current} → {target}")
            if current != target:
                gaps_identified.append(f"{metric.replace('_', ' ').title()} needs improvement")
    
    print(f"\nGaps Identified: {len(gaps_identified)}")
    for i, gap in enumerate(gaps_identified, 1):
        print(f"{i}. {gap}")
    
    return gaps_identified

def analyze_data_opportunities():
    """Analyze data and training opportunities"""
    print("\nANALYZING DATA OPPORTUNITIES")
    print("=" * 50)
    
    data_dir = "data/market"
    
    if not os.path.exists(data_dir):
        print("[ERROR] Market data directory not found")
        return []
    
    # Analyze available data
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.feather')]
    print(f"[INFO] Available data files: {len(data_files)}")
    
    opportunities = []
    
    if data_files:
        # Load first file to analyze data quality
        try:
            sample_file = data_files[0]
            df = pd.read_feather(os.path.join(data_dir, sample_file))
            
            print(f"[INFO] Sample data analysis ({sample_file}):")
            print(f"  - Records: {len(df):,}")
            print(f"  - Columns: {list(df.columns)}")
            
            if 'Date' in df.columns:
                df = df.set_index('Date')
            
            # Check data timespan
            if len(df) > 0:
                date_range = df.index[-1] - df.index[0]
                print(f"  - Timespan: {date_range.days} days")
                
                # Data quality checks
                missing_data = df.isnull().sum().sum()
                print(f"  - Missing values: {missing_data}")
                
                if len(df) >= 1000:
                    opportunities.append("Sufficient data for advanced training")
                else:
                    opportunities.append("Need more historical data")
                
                if missing_data == 0:
                    opportunities.append("Clean data - ready for training")
                else:
                    opportunities.append("Data cleaning required")
                
                # Check for multiple timeframes
                if len(data_files) >= 5:
                    opportunities.append("Multiple currency pairs available")
                
                # Check recency
                latest_date = df.index[-1]
                days_old = (pd.Timestamp.now() - latest_date).days
                print(f"  - Data age: {days_old} days old")
                
                if days_old < 30:
                    opportunities.append("Recent data available")
                else:
                    opportunities.append("Data needs updating")
                    
        except Exception as e:
            print(f"[ERROR] Data analysis failed: {e}")
            opportunities.append("Data format needs investigation")
    
    print(f"\nData Opportunities: {len(opportunities)}")
    for i, opp in enumerate(opportunities, 1):
        print(f"{i}. {opp}")
    
    return opportunities

def analyze_architecture_improvements():
    """Analyze potential architecture improvements"""
    print("\nANALYZING ARCHITECTURE IMPROVEMENTS")
    print("=" * 50)
    
    current_architecture = {
        "layers": "3-layer bidirectional LSTM",
        "attention": "8-head multi-head attention",
        "parameters": "397,519",
        "features": "20 technical indicators",
        "sequence_length": "80 periods",
        "prediction_horizon": "8 periods ahead"
    }
    
    print("Current Architecture:")
    for component, details in current_architecture.items():
        print(f"  {component.replace('_', ' ').title()}: {details}")
    
    # Identify improvement opportunities
    improvements = []
    
    # Model size optimization
    improvements.append("Consider model distillation for faster inference")
    improvements.append("Implement dynamic sequence length for varying data")
    improvements.append("Add ensemble methods for higher accuracy")
    improvements.append("Optimize attention mechanism for speed")
    improvements.append("Implement incremental learning for live adaptation")
    
    # Gemini integration optimization
    improvements.append("Cache Gemini responses to reduce API calls")
    improvements.append("Implement async Gemini processing")
    improvements.append("Add Gemini response validation")
    improvements.append("Optimize prompt engineering for better results")
    
    print(f"\nPotential Improvements: {len(improvements)}")
    for i, improvement in enumerate(improvements, 1):
        print(f"{i}. {improvement}")
    
    return improvements

def generate_optimization_plan():
    """Generate comprehensive optimization plan"""
    print(f"\n" + "=" * 60)
    print("OPTIMIZATION PLAN GENERATION")
    print("=" * 60)
    
    # Run all analyses
    model_compatible = analyze_model_compatibility()
    performance_gaps = analyze_performance_gaps()
    data_opportunities = analyze_data_opportunities()
    architecture_improvements = analyze_architecture_improvements()
    
    # Priority assessment
    print(f"\nPRIORITY OPTIMIZATION AREAS")
    print("-" * 40)
    
    high_priority = []
    medium_priority = []
    low_priority = []
    
    # Categorize by impact and effort
    if not model_compatible:
        high_priority.append("Fix model architecture compatibility")
    
    if "Speed needs" in str(performance_gaps):
        high_priority.append("Optimize processing speed (12s → 5s)")
    
    if "signal_bias" in str(performance_gaps):
        high_priority.append("Fix signal bias (100% HOLD → Balanced)")
    
    if "Accuracy needs" in str(performance_gaps):
        medium_priority.append("Improve accuracy (55.2% → 60%)")
    
    medium_priority.extend([
        "Optimize Gemini integration",
        "Implement response caching",
        "Add incremental learning"
    ])
    
    low_priority.extend([
        "Model distillation",
        "Ensemble methods",
        "Advanced feature engineering"
    ])
    
    print("HIGH PRIORITY (Immediate):")
    for i, item in enumerate(high_priority, 1):
        print(f"  {i}. {item}")
    
    print(f"\nMEDIUM PRIORITY (Short-term):")
    for i, item in enumerate(medium_priority, 1):
        print(f"  {i}. {item}")
    
    print(f"\nLOW PRIORITY (Long-term):")
    for i, item in enumerate(low_priority, 1):
        print(f"  {i}. {item}")
    
    # Generate action plan
    action_plan = {
        "immediate_actions": high_priority,
        "short_term_goals": medium_priority,
        "long_term_vision": low_priority,
        "estimated_effort": {
            "high_priority": "1-2 weeks",
            "medium_priority": "2-4 weeks", 
            "low_priority": "1-3 months"
        }
    }
    
    return action_plan

def main():
    """Run comprehensive optimization analysis"""
    
    print("FOREXSWING AI 2025 - OPTIMIZATION ANALYSIS")
    print("=" * 60)
    print("Analyzing current state and optimization opportunities...")
    print()
    
    # Generate comprehensive plan
    plan = generate_optimization_plan()
    
    # Final recommendations
    print(f"\n" + "=" * 60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    
    if plan["immediate_actions"]:
        print("🚨 IMMEDIATE ACTION REQUIRED:")
        print("  The system needs critical fixes before production use")
        print("  Focus on high-priority items first")
    else:
        print("✅ SYSTEM BASELINE ACCEPTABLE:")
        print("  Ready for optimization and enhancement")
    
    print(f"\nRECOMMENDED APPROACH:")
    print("1. Address high-priority issues (critical fixes)")
    print("2. Implement medium-priority optimizations (performance)")
    print("3. Consider long-term enhancements (advanced features)")
    
    print(f"\nESTIMATED TIMELINE:")
    print(f"  Production Ready: {plan['estimated_effort']['high_priority']}")
    print(f"  Optimized System: {plan['estimated_effort']['medium_priority']}")
    print(f"  Advanced Features: {plan['estimated_effort']['low_priority']}")
    
    print(f"\nNext Step: Choose optimization approach")
    print("  Option A: Quick fixes for immediate deployment")
    print("  Option B: Comprehensive optimization for best performance")
    
    return plan

if __name__ == "__main__":
    main()