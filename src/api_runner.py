"""OpenAI-compatible client factory for API-backed endpoints."""

import os

from openai import OpenAI


def create_client(api_key: str = None) -> OpenAI:
    """Create an OpenAI client.

    Looks for the API key in:
    1. Provided api_key parameter
    2. OPENAI_API_KEY environment variable
    3. config/openai_key.txt file
    """
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        key_file = os.path.join("config", "openai_key.txt")
        if os.path.exists(key_file):
            with open(key_file) as f:
                api_key = f.read().strip()
    if not api_key:
        raise ValueError("No OpenAI API key found. Set OPENAI_API_KEY or create config/openai_key.txt")

    return OpenAI(api_key=api_key)
