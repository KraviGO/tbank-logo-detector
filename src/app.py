from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
import os

class BoundingBox(BaseModel):
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)

class Detection(BaseModel):
    bbox: BoundingBox = Field(..., description="Результат детекции")

class DetectionResponse(BaseModel):
    detections: List[Detection] = Field(..., description="Список найденных логотипов")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")

app = FastAPI(title="T-Bank Logo Detection API", version="1.0.0")

MODEL_PATH = Path(os.getenv("YOLO_WEIGHTS", "models/yolov8s_640_e50.pt"))
IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
DEFAULT_CONF = float(os.getenv("YOLO_CONF", "0.25"))
DEFAULT_IOU  = float(os.getenv("YOLO_IOU", "0.7"))

def auto_device(prefer: Optional[str] = None) -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    if prefer == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = auto_device(os.getenv("YOLO_DEVICE"))
model: Optional[YOLO] = None

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Не найдены веса модели по пути: {MODEL_PATH.resolve()}")
    model = YOLO(str(MODEL_PATH))
    _ = model.predict(np.zeros((32, 32, 3), dtype=np.uint8), imgsz=32, conf=0.01, device=DEVICE, verbose=False)

ALLOWED_CT = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

def read_image_to_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось декодировать изображение")
    return img

def clip_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(round(x1)), W - 1))
    y1 = max(0, min(int(round(y1)), H - 1))
    x2 = max(0, min(int(round(x2)), W - 1))
    y2 = max(0, min(int(round(y2)), H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "weights": str(MODEL_PATH), "imgsz": IMGSZ}

@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    if file.content_type not in ALLOWED_CT:
        raise HTTPException(status_code=400, detail=f"Неподдерживаемый Content-Type: {file.content_type}")

    try:
        data = await file.read()
        img = read_image_to_bgr(data)

        if model is None:
            raise RuntimeError("Модель не загружена")

        res = model.predict(
            img,
            imgsz=IMGSZ,
            conf=DEFAULT_CONF,
            iou=DEFAULT_IOU,
            device=DEVICE,
            verbose=False
        )
        r = res[0]

        H, W = r.orig_shape
        detections: List[Detection] = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            for (x1, y1, x2, y2) in xyxy:
                cx1, cy1, cx2, cy2 = clip_box(x1, y1, x2, y2, W, H)
                detections.append(Detection(bbox=BoundingBox(x_min=cx1, y_min=cy1, x_max=cx2, y_max=cy2)))

        return DetectionResponse(detections=detections)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="internal_error", detail=str(e)).model_dump()
        )