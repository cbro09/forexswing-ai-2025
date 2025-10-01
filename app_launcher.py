#!/usr/bin/env python3
"""
ForexSwing AI 2025 - Application Launcher
Final deployment script for the all-in-one trading application
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "data/MarketData/EUR_USD_real_daily.csv",
        "data/MarketData/GBP_USD_real_daily.csv", 
        "data/MarketData/USD_JPY_real_daily.csv",
        "data/models/optimized_forex_ai.pth",
        "data/models/optimized_scaler.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    return missing_files

def check_core_files():
    """Check if core application files exist"""
    core_files = [
        "ForexBot.py",
        "easy_forex_bot.py", 
        "paper_trading_system.py",
        "forexswing_app.py"
    ]
    
    missing_files = []
    for file_path in core_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    return missing_files

def test_imports():
    """Test critical imports"""
    import_tests = [
        ("streamlit", "Streamlit web framework"),
        ("plotly", "Plotly charting library"),
        ("pandas", "Data manipulation"),
        ("numpy", "Numerical computing")
    ]
    
    failed_imports = []
    
    for module, description in import_tests:
        try:
            __import__(module)
        except ImportError:
            failed_imports.append((module, description))
    
    return failed_imports

def run_app():
    """Launch the Streamlit application"""
    
    print("=" * 60)
    print("FOREXSWING AI 2025 - PROFESSIONAL TRADING APPLICATION")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("ForexBot.py").exists():
        print("[ERROR] Please run this script from the ForexSwing-AI directory")
        print("        Current directory should contain ForexBot.py")
        return False
    
    print("\n[STEP 1] Checking core application files...")
    missing_core = check_core_files()
    if missing_core:
        print(f"[ERROR] Missing core files: {missing_core}")
        return False
    print("[OK] All core files present")
    
    print("\n[STEP 2] Checking data files...")
    missing_data = check_data_files()
    if missing_data:
        print(f"[WARNING] Missing data files: {len(missing_data)} files")
        for file in missing_data[:3]:  # Show first 3
            print(f"         - {file}")
        if len(missing_data) > 3:
            print(f"         ... and {len(missing_data) - 3} more")
        print("[INFO] App will run but some features may be limited")
    else:
        print("[OK] All data files present")
    
    print("\n[STEP 3] Checking Python dependencies...")
    failed_imports = test_imports()
    if failed_imports:
        print("[ERROR] Missing Python packages:")
        for module, desc in failed_imports:
            print(f"         - {module}: {desc}")
        print("\n[FIX] Install missing packages:")
        print("      pip install streamlit plotly pandas numpy")
        return False
    print("[OK] All dependencies available")
    
    print("\n[STEP 4] Launching web application...")
    print("=" * 60)
    print("LAUNCHING FOREXSWING AI 2025 WEB APPLICATION")
    print("=" * 60)
    print("Access URL: http://localhost:8502")
    print("Features: Real-time charts, AI recommendations, paper trading")
    print("Status: Professional trading interface active")
    print("Press Ctrl+C to stop the application")
    print("-" * 60)
    
    # Launch Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", "forexswing_app.py",
        "--server.port", "8502",
        "--server.address", "localhost", 
        "--browser.gatherUsageStats", "false",
        "--logger.level", "error"  # Reduce log noise
    ]
    
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("APPLICATION STOPPED")
        print("=" * 60)
        print("ForexSwing AI 2025 web application has been stopped.")
        print("Thank you for using our professional trading system!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to launch application: {e}")
        print("\n[TROUBLESHOOTING]")
        print("1. Ensure Streamlit is installed: pip install streamlit")
        print("2. Check port 8502 is not in use")
        print("3. Try: streamlit run forexswing_app.py --server.port 8503")
        return False

if __name__ == "__main__":
    success = run_app()
    if not success:
        print("\n[FAILED] Application could not start")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Application session completed")
        sys.exit(0)