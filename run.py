#!/usr/bin/env python3
"""
SensorScope Analytics Launcher
Simple script to launch the Streamlit application
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit application"""
    print("🚀 Starting SensorScope Analytics...")
    print("=" * 50)
    
    # Check if app.py exists
    if not Path("app.py").exists():
        print("❌ app.py not found in current directory")
        print("💡 Make sure you're in the project root directory")
        sys.exit(1)
    
    # Check if virtual environment is activated
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️ Virtual environment not detected")
        print("💡 Consider activating your virtual environment first:")
        if os.name == 'nt':  # Windows
            print("   venv\\Scripts\\activate")
        else:  # macOS/Linux
            print("   source venv/bin/activate")
        print()
    
    # Check if .env file exists
    if Path(".env").exists():
        print("✅ Environment configuration found")
    else:
        print("⚠️ .env file not found")
        print("💡 Create .env with your database configuration")
        print()
    
    print("🌐 Launching Streamlit application...")
    print("📍 URL: http://localhost:8501")
    print("💡 Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError:
        print("\n❌ Failed to start Streamlit")
        print("💡 Make sure Streamlit is installed: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✅ Application stopped")
        sys.exit(0)

if __name__ == "__main__":
    main() 