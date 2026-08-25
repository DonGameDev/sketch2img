"""
backend/server.py
A minimal FastAPI server with a WebSocket endpoint (scaffold for real-time deltas)
"""
from fastapi import FastAPI, WebSocket
import asyncio
import base64
from starlette.responses import PlainTextResponse

app = FastAPI()

@app.get("/health")
async def health():
    return PlainTextResponse("OK")

@app.websocket("/ws/draw")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # Expect data = { "bbox": [x,y,w,h], "image": <base64 png> }
            # For now echo back a simple acknowledgement. Worker integration will be added in later commits.
            await websocket.send_json({"status": "received", "bbox": data.get("bbox")})
    except Exception as e:
        await websocket.close()
