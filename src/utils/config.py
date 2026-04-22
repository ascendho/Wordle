import os
from dotenv import load_dotenv
from predibase import Predibase

def get_predibase_client() -> Predibase:
    """Load environment variables and initialize Predibase client."""
    load_dotenv()
    
    api_key = os.environ.get("PREDIBASE_API_KEY")
    if not api_key:
        raise ValueError("PREDIBASE_API_KEY environment variable is not set")
    
    return Predibase(api_token=api_key)
