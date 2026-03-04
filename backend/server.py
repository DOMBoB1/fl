from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

# pylint: disable=no-member  # OpenCV's cv2 bindings lack static attribute info

from engine import MonitoringEngine

app = FastAPI()
engine = MonitoringEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "invalid image"}

    _frame_out, stats = engine.process_external_frame(img, flip=False)
    return {"stats": stats}