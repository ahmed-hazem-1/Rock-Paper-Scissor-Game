import os
import time
import random
import base64
import numpy as np
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ultralytics import YOLO  # type: ignore
import cv2  # type: ignore

MODEL_PATH = os.getenv("YOLO_WEIGHTS", r".\best.pt")

model: Optional[YOLO] = None
MOVES = ["rock", "paper", "scissors"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model()
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


def load_model():
    global model
    if model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
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
            load_model()
        
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
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    }


