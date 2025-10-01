#!/usr/bin/env python3
"""
ForexSwing AI 2025 - Application Launcher
Simple launcher script for the all-in-one trading application
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install missing requirements"""
    print("Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements_app.txt"
    
    if requirements_file.exists():
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
    else:
        # Fallback to essential packages
        cmd = [sys.executable, "-m", "pip", "install", 
               "streamlit", "plotly", "pandas", "numpy", "torch"]
    
    try:
        subprocess.run(cmd, check=True)
        print("Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def launch_app():
    """Launch the Streamlit application"""
    app_file = Path(__file__).parent / "forexswing_app.py"
    
    if not app_file.exists():
        print("Application file not found!")
        return False
    
    print("Launching ForexSwing AI 2025...")
    print("Opening web interface...")
    print("Access URL: http://localhost:8502")
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_file),
        "--server.port", "8502",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ]
    
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
        return True
    except Exception as e:
        print(f"Error launching application: {e}")
        return False

def main():
    """Main launcher function"""
    print("=" * 60)
    print("ForexSwing AI 2025 - All-in-One Trading Application")
    print("Professional AI Trading System")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("ForexBot.py").exists():
        print("Error: Please run this script from the ForexSwing-AI directory")
        print("Current directory should contain ForexBot.py")
        return False
    
    # Check requirements
    missing = check_requirements()
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        response = input("Install missing packages? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            if not install_requirements():
                print("Failed to install requirements. Please install manually:")
                print("pip install streamlit plotly pandas numpy torch")
                return False
        else:
            print("Cannot launch without required packages")
            return False
    
    # Launch the application
    print("All requirements satisfied")
    return launch_app()

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)