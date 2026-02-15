from deepgram import Deepgram
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('DEEPGRAM_API_KEY')
print(f"API Key present: {bool(api_key)}")

if not api_key:
    print("API Key is missing!")
else:
    try:
        dg_client = Deepgram(api_key)
        print("Deepgram client initialized successfully")
    except Exception as e:
        print(f"Error initializing Deepgram: {e}")
        import traceback
        traceback.print_exc()
