import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any

app = FastAPI()

# 헬스 체크용 엔드포인트
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Color Extractor"}