import asyncio
import os
import json
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from deepgram import AsyncDeepgramClient

load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Initialize Deepgram Client
api_key = os.getenv("DEEPGRAM_API_KEY")

@app.get("/", response_class=HTMLResponse)
def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket accepted")
    
    deepgram = AsyncDeepgramClient(api_key=api_key)

    try:
        # Use the Async client pattern for Fern-generated SDK
        # deepgram.listen.v1.connect returns an AsyncContextManager that yields a socket client
        # Options are passed directly to connect
        async with deepgram.listen.v1.connect(
            model="nova-2", 
            smart_format="true",
            language="en-US"
        ) as dg_connection:
            print("Deepgram socket connected")

            async def sender():
                """Receive audio from client and send to Deepgram"""
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        print(f"Received {len(data)} bytes from client")
                        
                        # Sending raw bytes is handled by send_media or _send
                        if hasattr(dg_connection, 'send_media'):
                             await dg_connection.send_media(data)
                        else:
                             await dg_connection._send(data) 
                except Exception as e:
                    print(f"Sender error (client disconnected?): {e}")

            async def receiver():
                """Receive transcripts from Deepgram and send to client"""
                try:
                    # Iterate over messages from Deepgram
                    async for message in dg_connection:
                        print(f"Received message type: {type(message)}")
                        # print(f"Message content: {message}") 
                        
                        # Check if it has 'channel' attribute (ListenV1ResultsEvent)
                        if hasattr(message, 'channel'):
                            alternatives = message.channel.alternatives
                            if alternatives and len(alternatives) > 0:
                                transcript = alternatives[0].transcript
                                if transcript:
                                    print(f"Transcript: {transcript}")
                                    
                                    # Save to file
                                    with open("transcriptions.txt", "a") as f:
                                        f.write(transcript + " ")
                                    
                                    # Send to client
                                    await websocket.send_text(transcript)
                        
                        # Handle other event types if needed (Metadata, etc.)
                        
                except Exception as e:
                    print(f"Receiver error: {e}")

            # Run sender and receiver concurrently
            sender_task = asyncio.create_task(sender())
            receiver_task = asyncio.create_task(receiver())

            # Wait for either to finish (likely sender when WebSocket closes)
            done, pending = await asyncio.wait(
                [sender_task, receiver_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

    except Exception as e:
        print(f"Error in websocket_endpoint: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("WebSocket closed")
        # dg_connection is closed automatically by context manager