from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
import traceback

from dataset_store import init_dataset_db
from engine import MonitoringEngine, set_dataset_recorder


app = FastAPI(title="Classroom Monitor API", version="1.0")
engine = MonitoringEngine()

init_dataset_db()
set_dataset_recorder(None)

app.add_middleware(
    CORSMiddleware,
    # Permite orice port de pe localhost / 127.0.0.1 (Vite poate folosi 5173, 5174, etc.)
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


def _looks_like_upload(value) -> bool:
    return (
        value is not None
        and hasattr(value, "read")
        and hasattr(value, "filename")
    )


async def _pick_uploaded_image(request: Request):
    form = await request.form()

    upload = None
    for key in ("image", "file", "frame"):
        value = form.get(key)
        if _looks_like_upload(value):
            upload = value
            break

    boxes_raw = form.get("boxes", 0)
    try:
        boxes = int(boxes_raw)
    except (TypeError, ValueError):
        boxes = 0

    show_face_raw = form.get("show_face", 1)
    try:
        show_face = int(show_face_raw)
    except (TypeError, ValueError):
        show_face = 1

    show_head_raw = form.get("show_head", 1)
    try:
        show_head = int(show_head_raw)
    except (TypeError, ValueError):
        show_head = 1

    return upload, boxes, show_face, show_head


@app.post("/analyze")
async def analyze(request: Request):
    try:
        image, boxes, show_face, show_head = await _pick_uploaded_image(request)

        if image is None:
            return JSONResponse(
                {
                    "error": "missing image input",
                    "detail": "Expected multipart form-data with one of: image, file, frame",
                },
                status_code=400,
            )

        data = await image.read()
        if not data:
            return JSONResponse({"error": "empty upload"}, status_code=400)

        try:
            pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        except UnidentifiedImageError:
            return JSONResponse(
                {"error": "invalid image", "detail": "Uploaded file is not a valid image"},
                status_code=400,
            )

        img = np.array(pil_img)
        if img.size == 0:
            return JSONResponse({"error": "decoded image is empty"}, status_code=400)

        img = img[:, :, ::-1].copy()

        stats = engine.process_external_frame(img, want_boxes=bool(boxes))

        if not isinstance(stats, dict):
            return JSONResponse(
                {"error": "engine returned invalid response"},
                status_code=500,
            )

        if bool(boxes):
            faces_data = stats.get("faces_data", [])
            filtered_faces_data = []

            for item in faces_data:
                row = dict(item)

                if not bool(show_face):
                    row.pop("face_bbox_n", None)
                    if row.get("kind") == "face":
                        row["bbox_n"] = None

                if not bool(show_head):
                    row.pop("head_bbox_n", None)

                filtered_faces_data.append(row)

            stats["faces_data"] = filtered_faces_data
        else:
            stats["faces_data"] = []

        return {"stats": stats}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {
                "error": str(e),
                "type": type(e).__name__,
            },
            status_code=500,
        )


@app.post("/session/start")
def session_start():
    try:
        engine.start_session()
        return {
            "ok": True,
            "session_active": engine.session_active,
            "session_id": engine.session_id,
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/session/stop")
def session_stop():
    try:
        engine.stop_session()
        return {
            "ok": True,
            "summary": engine.get_session_summary(),
            "session_active": engine.session_active,
            "session_id": engine.session_id,
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/session/summary")
def session_summary():
    try:
        return {
            "ok": True,
            "session_active": engine.session_active,
            "session_id": engine.session_id,
            "summary": engine.get_session_summary(),
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/session/report")
def session_report(
    report_type: str = "teacher",
    chart_individual: int = 1,
    chart_group: int = 1,
    chart_alerts: int = 0,
):
    try:
        if engine.session_active:
            return JSONResponse(
                {"error": "Stop session first"},
                status_code=400,
            )

        selected_report_type = str(report_type or "teacher").strip().lower()
        if selected_report_type in {"user", "prof"}:
            selected_report_type = "teacher"
        if selected_report_type not in {"teacher", "developer"}:
            selected_report_type = "teacher"

        report_options = {
            "report_type": selected_report_type,
            "charts": {
                "individual": bool(chart_individual),
                "group": bool(chart_group),
                "alerts": bool(chart_alerts),
            },
        }

        path = engine.export_session_report_xlsx(report_options=report_options)
        filename = os.path.basename(path)

        return FileResponse(
            path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=filename,
        )

    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)