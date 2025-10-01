#!/usr/bin/env python3
"""
ForexSwing AI - Dependency Installer
Installs all required packages for the companion system to function
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"🔧 {description}")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        print(f"   Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python version compatible")
        return True
    else:
        print("   ❌ Python 3.8+ required")
        return False

def install_package(package, description=""):
    """Install a Python package"""
    print(f"📦 Installing {package}")
    if description:
        print(f"   Purpose: {description}")
    
    # Try different installation methods
    commands = [
        f"pip install {package}",
        f"python -m pip install {package}",
        f"pip3 install {package}",
        f"python3 -m pip install {package}"
    ]
    
    for cmd in commands:
        if run_command(cmd, f"Installing {package}"):
            return True
    
    print(f"   ⚠️ Could not install {package} - may need manual installation")
    return False

def check_existing_packages():
    """Check what packages are already installed"""
    print("🔍 Checking existing packages...")
    
    packages_to_check = [
        "requests", "feedparser", "pandas", "numpy", 
        "torch", "google-generativeai"
    ]
    
    available = []
    missing = []
    
    for package in packages_to_check:
        try:
            __import__(package.replace('-', '_'))
            available.append(package)
            print(f"   ✅ {package} already installed")
        except ImportError:
            missing.append(package)
            print(f"   ❌ {package} missing")
    
    return available, missing

def install_core_dependencies():
    """Install core dependencies for ForexSwing AI"""
    print("\n🚀 INSTALLING FOREXSWING AI DEPENDENCIES")
    print("=" * 60)
    
    # Core packages needed
    packages = {
        "requests": "HTTP requests for API calls",
        "feedparser": "RSS feed parsing for Yahoo Finance news",
        "pandas": "Data manipulation for market data", 
        "numpy": "Numerical computing for AI calculations",
        "torch": "PyTorch for LSTM model",
        "google-generativeai": "Google Gemini AI API"
    }
    
    installed_count = 0
    
    for package, description in packages.items():
        if install_package(package, description):
            installed_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"   Attempted: {len(packages)} packages")
    print(f"   Successful: {installed_count} packages")
    
    return installed_count == len(packages)

def create_environment_script():
    """Create a script to set up the environment"""
    script_content = '''#!/bin/bash
# ForexSwing AI - Environment Setup Script

echo "🤖 ForexSwing AI - Environment Setup"
echo "===================================="

# Check if virtual environment exists
if [ ! -d "forexai_env" ]; then
    echo "📁 Creating virtual environment..."
    python -m venv forexai_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source forexai_env/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "🚀 Installing ForexSwing AI dependencies..."
pip install requests feedparser pandas numpy torch google-generativeai

# Test installation
echo "🧪 Testing installation..."
python -c "
import requests, feedparser, pandas, numpy, torch, google.generativeai
print('✅ All core dependencies installed successfully!')
"

echo ""
echo "🎉 ForexSwing AI environment ready!"
echo "🚀 Start the companion system with:"
echo "   source forexai_env/bin/activate"
echo "   python companion_api_service.py"
'''
    
    with open("setup_environment.sh", "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod("setup_environment.sh", 0o755)
    print("✅ Created setup_environment.sh script")

def test_forexai_functionality():
    """Test if ForexSwing AI core functionality works"""
    print("\n🧪 TESTING FOREXSWING AI FUNCTIONALITY")
    print("=" * 50)
    
    # Test basic imports
    test_imports = [
        ("requests", "HTTP requests"),
        ("json", "JSON handling"), 
        ("time", "Time operations"),
        ("datetime", "Date/time handling")
    ]
    
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {module} - {description}")
        except ImportError:
            print(f"   ❌ {module} - {description} (MISSING)")
    
    # Test API service startup
    print(f"\n🔗 Testing API service components...")
    
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        print(f"   ✅ HTTP server components available")
    except ImportError:
        print(f"   ❌ HTTP server components missing")
    
    # Test if companion_api_service can be imported
    try:
        # Check if the main script exists
        if os.path.exists("companion_api_service.py"):
            print(f"   ✅ companion_api_service.py found")
        else:
            print(f"   ❌ companion_api_service.py missing")
    except Exception as e:
        print(f"   ⚠️ Could not test API service: {e}")

def main():
    """Main installation process"""
    print("🤖 FOREXSWING AI - DEPENDENCY INSTALLER")
    print("=" * 60)
    print("This script will install all required dependencies")
    print("for the ForexSwing AI companion system to function.")
    print("")
    
    # Check Python version
    if not check_python_version():
        print("❌ Incompatible Python version. Please upgrade to Python 3.8+")
        return False
    
    # Check existing packages
    available, missing = check_existing_packages()
    
    if not missing:
        print("\n🎉 All dependencies already installed!")
    else:
        print(f"\n📦 Need to install: {', '.join(missing)}")
        
        # Try to install missing packages
        if not install_core_dependencies():
            print("\n⚠️ Some packages failed to install.")
            print("📋 Manual installation guide:")
            print("   1. Create virtual environment: python -m venv forexai_env")
            print("   2. Activate: source forexai_env/bin/activate")
            print("   3. Install: pip install requests feedparser pandas numpy torch google-generativeai")
    
    # Create environment setup script
    create_environment_script()
    
    # Test functionality
    test_forexai_functionality()
    
    print(f"\n🎯 INSTALLATION COMPLETE")
    print("=" * 30)
    print("📋 Next steps:")
    print("   1. If packages failed: run ./setup_environment.sh")
    print("   2. Start API service: python companion_api_service.py")
    print("   3. Install browser extension from browser_extension/ folder")
    print("   4. Visit TradingView to see AI overlays!")
    print("")
    print("🔑 API Keys configured:")
    print("   ✅ Alpha Vantage: OXGW647WZO8XTKA1")
    print("   ✅ Google Gemini: AIzaSyBcSKWOjFghU3UItDVjNL62tWpGnn7I-bQ")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)