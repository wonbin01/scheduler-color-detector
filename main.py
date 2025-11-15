import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any
import json

class Cell(BaseModel):
    cell_id: int
    normalized_vertices : List[List[float]]

class DocumentData(BaseModel):
    image_width: int
    image_height: int
    cells: List[Cell]

app = FastAPI()

@app.post("/extract/colorInfo")
async def extract_color(
    image_file: UploadFile = File(...), 
    document_data_json: str = Form(...) 
):
    try:
        json_dict=json.loads(document_data_json)
        data: DocumentData=DocumentData.parse_obj(json_dict)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="유효한 JSON 형식이 아닙니다.")
    except Exception as e:
        raise HTTPException(status_code=400,detail=f"document data validation 실패: {e}")
    
    file_bytes=await image_file.read()
    try:
        np_array=np.frombuffer(file_bytes,np.uint8)
        image=cv2.imdecode(np_array,cv2.IMREAD_COLOR) #3채널 컬러 이미지로 로드
        if image is None:
            raise ValueError("이미지 파일을 디코딩하는 데 실패했습니다.")
        actual_height,actual_width=image.shape[:2]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 처리 오류 발생: {e}")
    results=[]
    
    for cell in data.cells:
        norm_vertices = np.array(cell.normalized_vertices, dtype=np.float32)
        
        abs_vertices = np.round(
            norm_vertices * np.array([actual_width, actual_height])
        ).astype(np.int32)
        
        min_x = np.clip(np.min(abs_vertices[:, 0]), 0, actual_width)
        max_x = np.clip(np.max(abs_vertices[:, 0]), 0, actual_width)
        min_y = np.clip(np.min(abs_vertices[:, 1]), 0, actual_height)
        max_y = np.clip(np.max(abs_vertices[:, 1]), 0, actual_height)

        cropped_area = image[min_y:max_y, min_x:max_x]
        
        avg_bgr = (0, 0, 0)
        if cropped_area.size > 0:
            avg_color_tuple = cv2.mean(cropped_area)
            avg_bgr = (int(avg_color_tuple[0]), int(avg_color_tuple[1]), int(avg_color_tuple[2]))
            
        
        results.append({
            "cell_id": cell.cell_id,
            "average_color_bgr": avg_bgr
        })
        
    color_info=classify_color(results)
    print(color_info)
    return color_info

def classify_color(results):
    classified=[]
    for cell in results:
        id=cell["cell_id"]
        b,g,r=cell["average_color_bgr"]
        if(max(b,g,r) - min(b,g,r) < 10):
            category="매점"
        elif b>r+10:
            category="웰컴"
        elif r>b+10:
            category="엔젤"
        else:
            category="기타"
        classified.append({
            "cell_id" : id,
            "position" : category
        })
    return classified       

# 헬스 체크용 엔드포인트
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Color Extractor"}