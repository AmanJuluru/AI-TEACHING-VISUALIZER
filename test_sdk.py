from deepgram import Deepgram
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def main():
    try:
        dg_client = Deepgram(os.getenv('DEEPGRAM_API_KEY'))
        print("Client verified.")
        # Minimal test to see if we can even create a live socket
        socket = await dg_client.transcription.live({'punctuate': True})
        print("Socket created successfully.")
        await socket.close()
        print("Socket closed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
