#!/usr/bin/env python3
"""
Simple monitoring dashboard for ForexSwing AI
Shows system status, recent predictions, and model performance
"""

import requests
import time
import os
from datetime import datetime
from typing import Dict, List
import json

class ForexBotMonitor:
    """Simple terminal-based monitoring dashboard"""

    def __init__(self, api_url: str = "http://localhost:8082"):
        self.api_url = api_url
        self.pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD']

    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_system_status(self) -> Dict:
        """Get system status from API"""
        try:
            response = requests.get(f"{self.api_url}/api/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def get_pair_analysis(self, pair: str) -> Dict:
        """Get analysis for a specific pair"""
        try:
            response = requests.get(
                f"{self.api_url}/api/analyze",
                params={'pair': pair},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def format_analysis(self, analysis: Dict) -> str:
        """Format analysis for display"""
        if 'error' in analysis:
            return f"❌ Error: {analysis['error']}"

        action = analysis.get('action', 'N/A')
        confidence = analysis.get('confidence', 0) * 100
        risk = analysis.get('risk_level', 'N/A')

        # Color code actions
        action_display = action
        if action == 'BUY':
            action_display = f"\033[92m{action}\033[0m"  # Green
        elif action == 'SELL':
            action_display = f"\033[91m{action}\033[0m"  # Red
        else:
            action_display = f"\033[93m{action}\033[0m"  # Yellow

        components = analysis.get('components', {})

        output = f"""
  Action: {action_display}
  Confidence: {confidence:.1f}%
  Risk: {risk}

  Components:
    LSTM:   {components.get('lstm', 'N/A')}
    Gemini: {components.get('gemini', 'N/A')}
    News:   {components.get('news', 'N/A')}

  Quality: {analysis.get('data_quality', 'N/A')}
  Time: {analysis.get('processing_time', 'N/A')}
"""
        return output

    def display_dashboard(self):
        """Display the monitoring dashboard"""
        self.clear_screen()

        print("="*70)
        print(" "*20 + "ForexSwing AI 2025 - Monitor")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # System status
        status = self.get_system_status()
        print("SYSTEM STATUS:")
        if 'error' in status:
            print(f"  ❌ Cannot connect: {status['error']}")
        else:
            print(f"  ✅ Status: {status.get('status', 'unknown')}")
            print(f"  📊 Pairs: {status.get('supported_pairs', 0)}")
            print(f"  💾 Cache: {status.get('cache_entries', 0)} entries")
            print(f"  🔑 API: {status.get('alpha_vantage_key', 'missing')}")

        print()
        print("-"*70)

        # Currency pair analyses
        for pair in self.pairs:
            print(f"\n{pair}:")
            print("-"*70)

            analysis = self.get_pair_analysis(pair)
            print(self.format_analysis(analysis))

        print()
        print("="*70)
        print("Press Ctrl+C to exit | Refreshes every 60 seconds")
        print("="*70)

    def run(self, refresh_interval: int = 60):
        """Run the monitoring dashboard"""
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    import sys

    api_url = "http://localhost:8082"
    if len(sys.argv) > 1:
        api_url = sys.argv[1]

    print(f"Connecting to ForexSwing AI at {api_url}...")
    print("Starting monitoring dashboard...")
    time.sleep(2)

    monitor = ForexBotMonitor(api_url)
    monitor.run()
