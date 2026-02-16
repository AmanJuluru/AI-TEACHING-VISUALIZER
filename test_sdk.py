from deepgram import DeepgramClient
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def main():
    try:
        dg_client = DeepgramClient(api_key=os.getenv('DEEPGRAM_API_KEY'))
        print("Client verified.")
        # Minimal test to see if we can create a live connection
        async with dg_client.listen.asynclive.v("1") as connection:
            print("Connection created successfully.")
        print("Connection closed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
