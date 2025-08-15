import os
import time
import random
import base64
import numpy as np
from typing import Optional
from contextlib import asynccontextmanager
import requests
import tempfile

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ultralytics import YOLO  # type: ignore
import cv2  # type: ignore

# For Vercel deployment - model needs to be hosted externally
MODEL_URL = os.getenv("MODEL_URL", "https://your-model-hosting-service.com/best.pt")
LOCAL_MODEL_PATH = os.getenv("YOLO_WEIGHTS", r"runs\detect\train\weights\best.pt")

model: Optional[YOLO] = None
MOVES = ["rock", "paper", "scissors"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_model()
    yield
    # Shutdown (if needed)


app = FastAPI(title="RPS YOLO API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectRequest(BaseModel):
    # Expect a base64 encoded image frame from client
    image_base64: str


class DetectResponse(BaseModel):
    player_move: Optional[str]
    computer_move: Optional[str]
    round_winner: Optional[str]
    confidence: Optional[float]
    inference_time_ms: float
    bounding_box: Optional[dict] = None  # {"x1": float, "y1": float, "x2": float, "y2": float}


async def download_model():
    """Download model from external URL for Vercel deployment"""
    try:
        print(f"Downloading model from {MODEL_URL}")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
        
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        
        temp_file.close()
        print(f"Model downloaded to {temp_file.name}")
        return temp_file.name
    except Exception as e:
        print(f"Failed to download model: {e}")
        return None


async def load_model():
    global model
    if model is None:
        try:
            # Try local path first (for development)
            if os.path.exists(LOCAL_MODEL_PATH):
                model_path = LOCAL_MODEL_PATH
                print(f"Using local model: {model_path}")
            else:
                # Download from external URL (for Vercel)
                model_path = await download_model()
                if not model_path:
                    raise FileNotFoundError("Could not load model from local or remote source")
            
            model = YOLO(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load YOLO model: {e}")


def decode_image(b64_data: str):
    try:
        # Remove data URL prefix if present
        if ',' in b64_data:
            b64_data = b64_data.split(',')[1]
        
        raw = base64.b64decode(b64_data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        return img
    except Exception as e:
        print(f"Image decode error: {e}")
        return None


def decide_winner(player: str, computer: str):
    if player == computer:
        return "tie"
    wins = {('rock','scissors'), ('scissors','paper'), ('paper','rock')}
    return 'player' if (player, computer) in wins else 'computer'


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    try:
        if model is None:
            await load_model()
        
        img = decode_image(req.image_base64)
        if img is None:
            return DetectResponse(
                player_move=None, 
                computer_move=None, 
                round_winner=None, 
                confidence=None, 
                inference_time_ms=0,
                bounding_box=None
            )
        
        start = time.time()
        results = model.predict(source=img, imgsz=320, verbose=False, conf=0.5)
        elapsed = (time.time() - start) * 1000.0
        
        top_label = None
        top_conf = 0.0
        best_box = None
        
        if results and len(results) > 0:
            r = results[0]
            if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                names = r.names if hasattr(r, 'names') else {}
                for box in r.boxes:
                    try:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        
                        label = names.get(cls_id, str(cls_id)).lower()
                        
                        # Only consider valid moves with sufficient confidence
                        if label in MOVES and conf > top_conf and conf > 0.5:
                            top_conf = conf
                            top_label = label
                            
                            # Get bounding box coordinates
                            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                            best_box = {
                                "x1": xyxy[0],
                                "y1": xyxy[1], 
                                "x2": xyxy[2],
                                "y2": xyxy[3]
                            }
                    except Exception as e:
                        print(f"Error processing detection: {e}")
                        continue
        
        # Generate game result if we detected a valid move
        if top_label:
            computer_move = random.choice(MOVES)
            winner = decide_winner(top_label, computer_move)
        else:
            computer_move = None
            winner = None
        
        return DetectResponse(
            player_move=top_label,
            computer_move=computer_move,
            round_winner=winner,
            confidence=top_conf if top_label else None,
            inference_time_ms=elapsed,
            bounding_box=best_box,
        )
    
    except Exception as e:
        print(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "environment": "vercel" if os.getenv("VERCEL") else "local"
    }


@app.get("/")
async def root():
    return {"message": "Rock Paper Scissors YOLO API is running!"}


# Vercel serverless function handler
handler = app
