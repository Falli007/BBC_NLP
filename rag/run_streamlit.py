#!/usr/bin/env python3
"""
Launcher script for the BBC News RAG Streamlit app.
This script handles the import path and launches the Streamlit app.
"""

import sys
import os
import subprocess

# Add the parent directory to the path so we can import the RAG system
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def main():
    """Launch the Streamlit app."""
    print("🚀 Starting BBC News RAG System...")
    print("📰 Opening Streamlit interface...")
    
    # Change to the rag directory
    os.chdir(current_dir)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

if __name__ == "__main__":
    main()
