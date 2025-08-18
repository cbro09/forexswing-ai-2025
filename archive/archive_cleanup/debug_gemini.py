#!/usr/bin/env python3
"""
Debug Gemini CLI detection
"""

import subprocess
import sys

def test_gemini_detection():
    print("Testing Gemini CLI detection...")
    
    try:
        print("Running: npx @google/gemini-cli --version")
        result = subprocess.run(['npx.cmd', '@google/gemini-cli', '--version'], 
                              capture_output=True, text=True, timeout=15, shell=True)
        
        print(f"Return code: {result.returncode}")
        print(f"Stdout: '{result.stdout.strip()}'")
        print(f"Stderr: '{result.stderr.strip()}'")
        
        if result.returncode == 0:
            print("[OK] Gemini CLI is available!")
            return True
        else:
            print("[ERROR] Gemini CLI test failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("[ERROR] Command timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return False

if __name__ == "__main__":
    test_gemini_detection()