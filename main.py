import asyncio
import os
import json
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from deepgram import AsyncDeepgramClient
from transcription_analyzer import TranscriptionAnalyzer, ContentType

load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Initialize Deepgram Client
api_key = os.getenv("DEEPGRAM_API_KEY")

# Initialize Transcription Analyzer
analyzer = TranscriptionAnalyzer()

@app.get("/", response_class=HTMLResponse)
def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/analysis")
async def get_analysis():
    """Get current analysis of transcriptions"""
    try:
        with open("transcriptions.txt", "r", encoding="utf-8") as f:
            text = f.read()
        
        results = await analyzer.segment_text(text)
        
        return {
            "status": "success",
            "summary": {
                "total_filler": len(results.filler),
                "total_administration": len(results.administration),
                "total_concepts": len(results.visual_concept),
                "total_segments": len(results.filler) + len(results.administration) + len(results.visual_concept)
            },
            "categories": results.to_dict()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/stats")
async def get_stats():
    """Get quick statistics"""
    try:
        with open("transcriptions.txt", "r", encoding="utf-8") as f:
            text = f.read()
        
        results = await analyzer.segment_text(text)
        total = len(results.filler) + len(results.administration) + len(results.visual_concept)
        
        return {
            "filler": {
                "count": len(results.filler),
                "percentage": round((len(results.filler) / total * 100), 2) if total > 0 else 0
            },
            "administration": {
                "count": len(results.administration),
                "percentage": round((len(results.administration) / total * 100), 2) if total > 0 else 0
            },
            "visual_concept": {
                "count": len(results.visual_concept),
                "percentage": round((len(results.visual_concept) / total * 100), 2) if total > 0 else 0
            },
            "total": total
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/buffer_status")
async def get_buffer_status():
    """Get current buffer status"""
    try:
        status = analyzer.get_buffer_status()
        concepts_status = analyzer.get_visual_concepts_status()
        return {
            "status": "success",
            "buffer_status": status,
            "visual_concepts": concepts_status
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket accepted")
    
    deepgram = AsyncDeepgramClient(api_key=api_key)

    try:
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
                        
                        if hasattr(dg_connection, 'send_media'):
                             await dg_connection.send_media(data)
                        else:
                             await dg_connection._send(data) 
                except Exception as e:
                    print(f"Sender error (client disconnected?): {e}")

            async def receiver():
                """Receive transcripts from Deepgram, classify, and generate images"""
                try:
                    async for message in dg_connection:
                        print(f"Received message type: {type(message)}")
                        
                        if hasattr(message, 'channel'):
                            alternatives = message.channel.alternatives
                            if alternatives and len(alternatives) > 0:
                                transcript = alternatives[0].transcript
                                if transcript:
                                    print(f"Transcript: {transcript}")
                                    
                                    # Send live transcript to client
                                    await websocket.send_json({
                                        "type": "transcript",
                                        "text": transcript
                                    })
                                    
                                    # Add transcript to buffer for batch classification
                                    try:
                                        batch_results = await analyzer.add_to_buffer(transcript)
                                        
                                        if batch_results:
                                            print(f"Buffer full! Processing {len(batch_results)} classified segments")
                                            
                                            # Process classified segments
                                            for segment_text, content_type in batch_results:
                                                # Save to file
                                                with open("transcriptions.txt", "a") as f:
                                                    f.write(segment_text + " ")
                                                
                                                # Collect visual concepts for image generation
                                                if content_type == ContentType.VISUAL_CONCEPT:
                                                    analyzer.add_visual_concept(segment_text)
                                                    print(f"  Visual concept added: {segment_text[:60]}...")
                                                
                                                # Send classification result to client
                                                await websocket.send_json({
                                                    "type": "classification",
                                                    "text": segment_text,
                                                    "category": content_type.value
                                                })
                                            
                                            # Check if enough visual concepts for image generation
                                            if analyzer.should_generate_image():
                                                concepts_status = analyzer.get_visual_concepts_status()
                                                print(f"Generating image from {concepts_status['count']} visual concepts...")
                                                
                                                # Notify client that image generation is starting
                                                await websocket.send_json({
                                                    "type": "image_generating",
                                                    "concepts": concepts_status['concepts']
                                                })
                                                
                                                try:
                                                    result = await analyzer.generate_image_from_concepts()
                                                    if result:
                                                        print(f"Image generated! Sending to client...")
                                                        await websocket.send_json({
                                                            "type": "generated_image",
                                                            "image_data": result['image_base64'],
                                                            "mime_type": result['mime_type'],
                                                            "concepts_used": result['concepts_used']
                                                        })
                                                except RuntimeError as e:
                                                    print(f"Image generation error: {e}")
                                                    rate_status = analyzer.get_rate_limit_status()
                                                    await websocket.send_json({
                                                        "type": "error",
                                                        "error": str(e),
                                                        "rate_limit": rate_status
                                                    })
                                        else:
                                            # Buffer not full yet
                                            buffer_status = analyzer.get_buffer_status()
                                            concepts_status = analyzer.get_visual_concepts_status()
                                            print(f"Buffering: {buffer_status['buffered_segments']}/{buffer_status['buffer_size']} segments | Concepts: {concepts_status['count']}/{concepts_status['min_required']}")
                                            
                                            await websocket.send_json({
                                                "type": "buffering",
                                                "buffered": buffer_status['buffered_segments'],
                                                "buffer_size": buffer_status['buffer_size'],
                                                "concepts_count": concepts_status['count'],
                                                "concepts_required": concepts_status['min_required']
                                            })
                                    
                                    except RuntimeError as e:
                                        print(f"Classification error: {e}")
                                        rate_status = analyzer.get_rate_limit_status()
                                        await websocket.send_json({
                                            "type": "error",
                                            "error": str(e),
                                            "rate_limit": rate_status
                                        })
                                        continue
                        
                except Exception as e:
                    print(f"Receiver error: {e}")

            # Run sender and receiver concurrently
            sender_task = asyncio.create_task(sender())
            receiver_task = asyncio.create_task(receiver())

            done, pending = await asyncio.wait(
                [sender_task, receiver_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()

    except Exception as e:
        print(f"Error in websocket_endpoint: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("WebSocket closed")