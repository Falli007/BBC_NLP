#!/usr/bin/env python3
"""
Quick script to check OpenAI API key status and test connection.
"""

import os
from openai import OpenAI

def check_api_key():
    """Check if OpenAI API key is set and test connection."""
    print("Checking OpenAI API Key Status...")
    print("=" * 50)
    
    # Check if the environment variable exists
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not found!")
        print("\nTo set it:")
        print("PowerShell: $env:OPENAI_API_KEY = 'your-key-here'")
        print("Command Prompt: set OPENAI_API_KEY=your-key-here")
        return False
    
    print(f"SUCCESS: API Key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Test the connection
    try:
        client = OpenAI(api_key=api_key)
        
        # Make a simple test call
        print("\nTesting API connection...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API test successful'"}],
            max_tokens=10
        )
        
        print("SUCCESS: API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"ERROR: API connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    check_api_key()
