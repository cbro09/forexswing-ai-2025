#!/usr/bin/env python3
"""
Setup script for Gemini CLI integration with ForexSwing AI
Installs and configures Gemini CLI for enhanced trading analysis
"""

import subprocess
import os
import sys
from pathlib import Path

def check_nodejs():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"[OK] Node.js found: {version}")
            
            # Check if version is 20+
            version_num = int(version.split('.')[0][1:])  # Remove 'v' and get major version
            if version_num >= 20:
                return True
            else:
                print(f"[WARNING] Node.js version {version} found, but version 20+ is required")
                return False
        else:
            print("[ERROR] Node.js not found")
            return False
    except FileNotFoundError:
        print("[ERROR] Node.js not found")
        return False

def install_gemini_cli():
    """Install Gemini CLI"""
    try:
        print("[INSTALLING] Gemini CLI...")
        result = subprocess.run(['npm', 'install', '-g', '@google/gemini-cli'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[OK] Gemini CLI installed successfully")
            return True
        else:
            print(f"[ERROR] Installation failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("[ERROR] npm not found. Please install Node.js first.")
        return False

def test_gemini_cli():
    """Test if Gemini CLI is working"""
    try:
        print("[TESTING] Gemini CLI...")
        result = subprocess.run(['npx', '@google/gemini-cli', '--version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("[OK] Gemini CLI is working")
            return True
        else:
            print(f"[ERROR] Gemini CLI test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[WARNING] Gemini CLI test timed out")
        return False
    except FileNotFoundError:
        print("[ERROR] npx not found")
        return False

def setup_authentication():
    """Guide user through authentication setup"""
    print("\n🔐 AUTHENTICATION SETUP")
    print("=" * 40)
    print("To use Gemini CLI, you need to set up authentication.")
    print("Choose one of the following options:")
    print()
    print("OPTION 1: Google Account (Recommended)")
    print("  - Free access to Gemini 2.5 Pro")
    print("  - Run: npx @google/gemini-cli auth login")
    print()
    print("OPTION 2: API Key")
    print("  - Get API key from: https://ai.google.dev/")
    print("  - Set environment variable: GEMINI_API_KEY=your_key_here")
    print()
    print("OPTION 3: Vertex AI (Enterprise)")
    print("  - For Google Cloud users")
    print("  - Set: GOOGLE_GENAI_USE_VERTEXAI=true")
    print()
    
    choice = input("Which option would you like to use? (1/2/3): ").strip()
    
    if choice == "1":
        print("\nRun this command to authenticate:")
        print("npx @google/gemini-cli auth login")
        print("\nThis will open a web browser for Google OAuth.")
        
    elif choice == "2":
        api_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
        if api_key:
            # Add to environment
            env_file = Path(".env")
            with open(env_file, "a") as f:
                f.write(f"\nGEMINI_API_KEY={api_key}\n")
            print("[OK] API key saved to .env file")
        else:
            print("[WARNING] You can set GEMINI_API_KEY environment variable later")
            
    elif choice == "3":
        print("\nFor Vertex AI setup:")
        print("1. Enable Vertex AI API in Google Cloud Console")
        print("2. Set up Application Default Credentials")
        print("3. Set GOOGLE_GENAI_USE_VERTEXAI=true")
        
    else:
        print("[WARNING] No authentication method selected")

def create_test_script():
    """Create a test script for Gemini integration"""
    test_script = '''#!/usr/bin/env python3
"""
Test Gemini CLI integration with ForexSwing AI
"""

import sys
import os
sys.path.append('core')

from integrations.gemini_data_interpreter import GeminiDataInterpreter
from integrations.enhanced_strategy import EnhancedForexStrategy
import pandas as pd
import numpy as np

def test_integration():
    print("GEMINI CLI INTEGRATION TEST")
    print("=" * 40)
    
    # Test 1: Basic Gemini availability
    interpreter = GeminiDataInterpreter()
    print(f"Gemini CLI Available: {interpreter.gemini_available}")
    
    if not interpreter.gemini_available:
        print("❌ Gemini CLI not available")
        print("Run: python scripts/setup_gemini.py")
        return False
    
    # Test 2: Enhanced strategy
    print("\\nTesting Enhanced Strategy...")
    strategy = EnhancedForexStrategy()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 1.1000,
        'volume': np.random.randint(1000, 5000, 100),
        'rsi': np.random.uniform(30, 70, 100),
        'macd': np.random.randn(100) * 0.0001,
        'macd_signal': np.random.randn(100) * 0.0001,
    })
    
    # Test recommendation
    rec = strategy.get_trading_recommendation(sample_data, "EUR/USD")
    
    print(f"✅ Trading recommendation generated:")
    print(f"   Action: {rec['action']}")
    print(f"   Confidence: {rec['confidence']:.1%}")
    print(f"   Risk: {rec['risk_level']}")
    
    print("\\n🎯 Integration test complete!")
    return True

if __name__ == "__main__":
    test_integration()
'''
    
    with open("test_gemini_integration.py", "w") as f:
        f.write(test_script)
    
    print("[OK] Test script created: test_gemini_integration.py")

def main():
    """Main setup function"""
    print("GEMINI CLI SETUP FOR FOREXSWING AI")
    print("=" * 50)
    print("This script will set up Gemini CLI integration for enhanced trading analysis.")
    print()
    
    # Step 1: Check Node.js
    if not check_nodejs():
        print("\n[ERROR] Setup failed: Node.js 20+ is required")
        print("Install Node.js from: https://nodejs.org/")
        return
    
    # Step 2: Install Gemini CLI
    if not install_gemini_cli():
        print("\n[ERROR] Setup failed: Could not install Gemini CLI")
        return
    
    # Step 3: Test installation
    if not test_gemini_cli():
        print("\n[WARNING] Gemini CLI installed but test failed")
        print("You may need to configure authentication")
    
    # Step 4: Authentication setup
    setup_authentication()
    
    # Step 5: Create test script
    create_test_script()
    
    print("\n[COMPLETE] SETUP FINISHED!")
    print("=" * 30)
    print("Next steps:")
    print("1. Complete authentication (see instructions above)")
    print("2. Run: python test_gemini_integration.py")
    print("3. Integrate with your trading strategy")
    print()
    print("Your ForexSwing AI can now use Gemini for:")
    print("  - Advanced market interpretation")
    print("  - Signal validation and enhancement") 
    print("  - Anomaly detection")
    print("  - Real-time trading commentary")

if __name__ == "__main__":
    main()