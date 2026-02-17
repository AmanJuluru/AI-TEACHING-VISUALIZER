---
description: Complete end-to-end flow of the AI Teaching Visualizer project
---

# 🎓 AI Teaching Visualizer — Complete Project Flow

## Overview

The AI Teaching Visualizer is a **real-time speech-to-visualization** application.
A teacher speaks into their microphone, and the system:

1. **Transcribes** speech in real time (Deepgram)
2. **Classifies** each segment using AI (Gemini Flash 2.0)
3. **Generates educational diagrams** from visual concepts (Gemini Image Generation)
4. **Displays everything live** in the browser

---

## Architecture Diagram

```
┌──────────────┐     audio/webm      ┌───────────────────┐
│   Browser    │ ──────────────────▶  │   FastAPI Server   │
│  (index.html)│ ◀────── JSON ──────  │     (main.py)      │
│              │    WebSocket /listen  │                    │
└──────┬───────┘                      └────────┬───────────┘
       │                                       │
       │ getUserMedia()                        │ audio bytes
       │ MediaRecorder                         ▼
       │                              ┌─────────────────┐
       │                              │    Deepgram      │
       │                              │   (Nova-2 STT)   │
       │                              └────────┬─────────┘
       │                                       │ transcript text
       │                                       ▼
       │                              ┌──────────────────────────┐
       │                              │  TranscriptionAnalyzer   │
       │                              │ (transcription_analyzer) │
       │                              │                          │
       │                              │  Buffer (5 segments)     │
       │                              │         │                │
       │                              │         ▼                │
       │                              │  Gemini Flash 2.0        │
       │                              │  (classify_batch_async)  │
       │                              │         │                │
       │                              │    ┌────┼─────┐          │
       │                              │    ▼    ▼     ▼          │
       │                              │ FILLER ADMIN CONCEPT     │
       │                              │                │         │
       │                              │                ▼         │
       │                              │  Visual Concepts Buffer  │
       │                              │  (≥3 concepts triggers)  │
       │                              │         │                │
       │                              │         ▼                │
       │                              │  Gemini 2.5 Flash Image  │
       │                              │  (generate_image_from_   │
       │                              │   concepts)              │
       │                              └────────┬─────────────────┘
       │                                       │
       │◀──── classification JSON ─────────────┘
       │◀──── generated_image (base64) ────────┘
       ▼
  Live UI update
```

---

## Step-by-Step Flow

### 1. Application Startup

| File | What happens |
|------|-------------|
| `main.py` | FastAPI app is created, `.env` is loaded |
| `main.py` | `DEEPGRAM_API_KEY` is read from environment |
| `main.py` | `TranscriptionAnalyzer()` is initialized |
| `transcription_analyzer.py` | Gemini SDK is configured with `GOOGLE_API_KEY` |
| `transcription_analyzer.py` | Image generation client (`google-genai`) is initialized |
| `transcription_analyzer.py` | Rate limiter is set up (9 RPM / 240 RPD) |

**Run command:**
```bash
uvicorn main:app --reload
```

---

### 2. Browser Connects

| Step | Details |
|------|---------|
| User opens `http://localhost:8000` | `GET /` serves `index.html` |
| Browser requests microphone access | `navigator.mediaDevices.getUserMedia({ audio: true })` |
| MediaRecorder starts | Encodes audio as `audio/webm`, fires `dataavailable` every **250ms** |
| WebSocket opens | Connects to `ws://localhost:8000/listen` |
| Status bar updates | Shows "🟢 Connected — Listening for speech…" |

---

### 3. Audio Streaming (Client → Server → Deepgram)

```
Browser ──(audio chunks every 250ms)──▶ WebSocket /listen
                                              │
                                      sender() task
                                              │
                                              ▼
                                     Deepgram Live STT
                                     (model: nova-2)
                                     (language: en-US)
                                     (smart_format: true)
```

- The `sender()` coroutine in `main.py` receives raw audio bytes from the browser
  and forwards them to Deepgram's streaming API via `dg_connection.send_media(data)`

---

### 4. Transcription Received (Deepgram → Server)

The `receiver()` coroutine listens for Deepgram responses:

```python
async for message in dg_connection:
    transcript = message.channel.alternatives[0].transcript
```

**Immediately sent to browser:**
```json
{ "type": "transcript", "text": "the mitochondria is the powerhouse of the cell" }
```

The transcript text appears in the **Live Transcript** panel on the left.

---

### 5. Buffering & Batch Classification

Each transcript segment is added to a **5-segment buffer**:

```
Buffer: [ seg1, seg2, seg3, seg4, seg5 ]  ← buffer full!
```

**While buffering** (buffer not full), the server sends:
```json
{ "type": "buffering", "buffered": 3, "buffer_size": 5, "concepts_count": 1 }
```

**When buffer reaches 5**, all segments are sent to **Gemini Flash 2.0** in a single batch:

```
Prompt to Gemini:
  "Classify each segment into FILLER, ADMINISTRATION, or VISUAL_CONCEPT"
  1. um yeah so
  2. open your textbooks to page 42
  3. the mitochondria produces ATP through cellular respiration
  4. uh
  5. this process is called oxidative phosphorylation
```

**Gemini responds:**
```
1. FILLER
2. ADMINISTRATION
3. VISUAL_CONCEPT
4. FILLER
5. VISUAL_CONCEPT
```

---

### 6. Classification Categories

| Category | Emoji | Description | Example |
|----------|-------|-------------|---------|
| **FILLER** | — | Hesitations, discourse markers | "um", "like", "you know" |
| **ADMINISTRATION** | ⚙️ | Logistics, greetings, meta-talk | "open your textbooks", "hello everyone" |
| **VISUAL_CONCEPT** | 💡 | Core educational content | "DNA is a double helix structure" |

Each classified segment is sent to the browser:
```json
{ "type": "classification", "text": "...", "category": "visual_concept" }
```

The transcript line is re-rendered with color coding:
- 💡 Green border for visual concepts
- Italic gray for filler
- ⚙️ Yellow for administration

---

### 7. Visual Concept Accumulation & Image Generation

When a segment is classified as `VISUAL_CONCEPT`, it is added to a separate **concepts buffer**:

```
Visual Concepts: [ concept1, concept2, concept3 ]  ← threshold reached (≥3)!
```

**Trigger:** When `≥ 3` visual concepts accumulate, image generation starts.

**Step-by-step:**

1. Server sends `{ "type": "image_generating", "concepts": [...] }` → browser shows spinner
2. A prompt is built from the concepts:
   ```
   Create a clear, educational diagram for a classroom setting.
   Visually explain:
   - the mitochondria produces ATP through cellular respiration
   - this process is called oxidative phosphorylation
   - the electron transport chain transfers electrons
   ```
3. `Gemini 2.5 Flash Image` model generates an educational diagram
4. Image bytes are base64-encoded and sent to browser:
   ```json
   {
     "type": "generated_image",
     "image_data": "<base64>",
     "mime_type": "image/png",
     "concepts_used": ["...", "...", "..."]
   }
   ```
5. Browser displays the image in the **Generated Visualization** panel
6. Concepts buffer is cleared; accumulation restarts for the next image

---

### 8. Error Handling & Rate Limiting

| Protection | Details |
|------------|---------|
| **Rate Limiter** | 9 requests/minute, 240 requests/day (under Gemini free tier limits) |
| **Exponential Backoff** | On API failure: retries 3 times with 5s → 10s → 20s delays |
| **Buffer Recovery** | If classification fails, segments are put back into the buffer |
| **Concept Recovery** | If image generation fails, concepts are put back into the buffer |
| **Error UI** | Red banner with rate limit info + expandable error log |

---

### 9. REST API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve the frontend HTML |
| `/api/analysis` | GET | Classify full `transcriptions.txt` and return results |
| `/api/stats` | GET | Return category counts and percentages |
| `/api/buffer_status` | GET | Return current buffer and visual concepts status |
| `/listen` | WebSocket | Real-time audio streaming and classification pipeline |

---

### 10. Frontend UI Components

| Component | Location | Function |
|-----------|----------|----------|
| **Status Bar** | Top | Connection status (🟢/🔴) |
| **Error Banner** | Below status | Dismissible API error display |
| **Buffer Indicator** | Below error | Shows buffering progress (e.g., 3/5) |
| **Live Transcript** | Left panel | Scrolling feed of classified segments |
| **Generated Visualization** | Right panel | Latest AI-generated educational image |
| **Stats Bar** | Below panels | Counters: concepts, images, segments, API remaining |
| **Image History** | Below stats | Thumbnails of all previously generated images |
| **Error Log** | Bottom | Expandable log of all errors with timestamps |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python, FastAPI, Uvicorn |
| **Speech-to-Text** | Deepgram Nova-2 (streaming WebSocket) |
| **AI Classification** | Google Gemini Flash 2.0 (`gemini-2.0-flash`) |
| **Image Generation** | Google Gemini Flash 2.5 Image (`gemini-2.5-flash-image`) |
| **Frontend** | Vanilla HTML/CSS/JS, Jinja2 templates |
| **Real-time Comms** | WebSocket (browser ↔ server ↔ Deepgram) |

---

## Environment Variables

```env
DEEPGRAM_API_KEY=your_deepgram_api_key
GOOGLE_API_KEY=your_google_gemini_api_key
```

---

## File Structure

```
AI-TEACHING-VISUALIZER/
├── main.py                     # FastAPI server, WebSocket, routing
├── transcription_analyzer.py   # Gemini classification + image generation
├── templates/
│   └── index.html              # Frontend UI (HTML + CSS + JS)
├── requirements.txt            # Python dependencies
├── .env                        # API keys (DEEPGRAM + GOOGLE)
├── transcriptions.txt          # Persisted transcription text
└── analysis_results.json       # Saved analysis output
```

---

## Data Flow Summary

```
🎤 Microphone
    ↓ audio/webm (250ms chunks)
📡 WebSocket /listen
    ↓ raw bytes
🗣️ Deepgram Nova-2 STT
    ↓ transcript text
📦 Segment Buffer (5 segments)
    ↓ batch of 5
🤖 Gemini Flash 2.0 (classification)
    ↓ FILLER | ADMINISTRATION | VISUAL_CONCEPT
💡 Visual Concepts Buffer (≥3 concepts)
    ↓ prompt with concepts
🎨 Gemini 2.5 Flash Image (generation)
    ↓ base64 image
🖥️ Browser UI (real-time display)
```