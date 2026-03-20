from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from dataset_store import FrameDatasetRecorder, init_dataset_db

import numpy as np
from PIL import Image
import io
import os

from dataset_store import FrameDatasetRecorder, init_dataset_db
from engine import MonitoringEngine, set_dataset_recorder


app = FastAPI(title="Classroom Monitor API", version="1.0")
engine = MonitoringEngine()

dataset_recorder = FrameDatasetRecorder(save_every_n_frames=1)
init_dataset_db()
set_dataset_recorder(dataset_recorder)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    boxes: int = Form(0),
):
    try:
        data = await image.read()
        if not data:
            return JSONResponse({"error": "empty upload"}, status_code=400)

        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        img = np.array(pil_img)
        img = img[:, :, ::-1].copy()

        res = engine.process_external_frame(img, want_boxes=bool(int(boxes)))

        if isinstance(res, tuple) and len(res) == 2:
            stats = res[1]
        else:
            stats = res

        return {"stats": stats}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/session/start")
def session_start():
    dataset_recorder.start(
    source="webcam",
    notes="Sesiune pornita din frontend"
)
    try:
        engine.start_session()
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/session/stop")
def session_stop():
    dataset_recorder.stop()
    try:
        engine.stop_session()
        return {
            "ok": True,
            "summary": engine.get_session_summary()
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/session/report")
def session_report():
    try:
        if engine.session_active:
            return JSONResponse(
                {"error": "Stop session first"},
                status_code=400
            )

        path = engine.export_session_report_xlsx()
        filename = os.path.basename(path)

        return FileResponse(
            path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=filename
        )

    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)