#!/usr/bin/env python3
"""
Quick Optimization Progress Checker
"""

import os
import time
from datetime import datetime

def check_optimization_status():
    """Check if optimization is complete"""
    
    print("FOREXSWING AI OPTIMIZATION STATUS")
    print("=" * 40)
    print(f"Check Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Check for optimized model
    optimized_model = "models/optimized_forex_ai.pth"
    optimized_scaler = "models/optimized_scaler.pkl"
    
    if os.path.exists(optimized_model):
        print("STATUS: OPTIMIZATION COMPLETE!")
        print(f"Optimized model found: {optimized_model}")
        
        if os.path.exists(optimized_scaler):
            print(f"Optimized scaler found: {optimized_scaler}")
        
        # Check file timestamps
        model_time = os.path.getmtime(optimized_model)
        model_datetime = datetime.fromtimestamp(model_time)
        
        print(f"Model created: {model_datetime.strftime('%H:%M:%S')}")
        print()
        print("READY TO TEST OPTIMIZED AI!")
        print("Run: python test_optimized_ai.py")
        
        return True
    else:
        print("STATUS: OPTIMIZATION IN PROGRESS...")
        print("Training is still running in the background.")
        print("Estimated completion: 15-30 minutes")
        print()
        print("Check again in a few minutes or wait for completion.")
        
        return False

if __name__ == "__main__":
    complete = check_optimization_status()
    
    if complete:
        print("\nOptimization complete! Your AI has been upgraded.")
    else:
        print("\nOptimization still running. Please wait...")