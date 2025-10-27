#!/usr/bin/env python3
"""
Generate a new proxy API key for FreeAPI

This script creates a new proxy authentication token that users can use
to access the OpenAI-compatible proxy endpoints.

Usage:
    uv run generate_proxy_key.py
"""

import sys
import os

# Add parent directory to path to import from server
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import proxy_auth

def main():
    """Generate and display a new proxy API key"""
    print("=" * 60)
    print("FreeAPI Proxy Key Generator")
    print("=" * 60)
    print()
    
    # Generate new token
    new_token = proxy_auth.add_token()
    
    print("✅ New proxy API key generated successfully!")
    print()
    print("Your API Key:")
    print(f"  {new_token}")
    print()
    print("Usage in test.py:")
    print(f'  OPENROUTER_API_KEY = "{new_token}"')
    print()
    print("Usage with OpenAI Python client:")
    print("  client = OpenAI(")
    print(f'    base_url="http://localhost:8964/v1",')
    print(f'    api_key="{new_token}"')
    print("  )")
    print()
    print("This key has been saved to: openrouter/proxy_tokens.ini")
    print("=" * 60)

if __name__ == "__main__":
    main()

