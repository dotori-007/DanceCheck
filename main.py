import os
import json
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_FILE_PATH = os.path.join(BASE_DIR, "logs.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# 동시 파일 접근 제어를 위한 asyncio.Lock
file_lock = asyncio.Lock()

def init_logs_file():
    """logs.json 파일이 없거나 비어있는 경우 빈 배열 []로 초기화"""
    if not os.path.exists(LOGS_FILE_PATH):
        with open(LOGS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
    else:
        # 파일이 손상되었거나 빈 파일인 경우 안전 복구
        try:
            with open(LOGS_FILE_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("Empty file")
                json.loads(content)
        except Exception:
            with open(LOGS_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 기동 시 초기화
    init_logs_file()
    yield

app = FastAPI(
    title="DanceCheck Server",
    description="초경량 모바일 댄스 판정 및 로그 저장 서버",
    version="1.0.0",
    lifespan=lifespan
)

# 모바일 로컬 테스트 및 브라우저 호환을 위한 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 모델 정의
class ChecklistStatus(BaseModel):
    is_body_detected: bool = Field(..., description="전신/주요 관절 인식 여부")
    is_moving: bool = Field(..., description="기준치 이상의 유의미한 움직임 감지")
    is_rhythmic: bool = Field(..., description="리듬감 및 주기적 바운스/패턴 감지")
    is_dynamic: bool = Field(..., description="사지 및 가동 범위(Range of Motion) 충족")
    is_sustained: bool = Field(..., description="연속성 및 지속 시간 충족")

class MetricDetails(BaseModel):
    dance_score: float = Field(..., ge=0, le=100, description="최종 댄스 판정 점수 (0~100)")
    energy_level: float = Field(..., description="평균 운동 에너지 지표")
    bpm_estimate: Optional[float] = Field(None, description="추정된 리듬 BPM")
    periodicity_score: float = Field(..., description="주기성 일치도 (0~1)")
    analysis_duration_sec: float = Field(..., description="분석 수행 시간(초)")
    fps: Optional[float] = Field(None, description="평균 클라이언트 FPS")

class DanceLogEntry(BaseModel):
    session_id: Optional[str] = Field(None, description="세션 고유 ID")
    timestamp: Optional[str] = Field(None, description="측정 일시 (ISO-8601 또는 현지 시간)")
    is_dancing: bool = Field(..., description="최종 춤 판정 결과 (True / False)")
    confidence: float = Field(..., ge=0, le=100, description="판정 신뢰도 (%)")
    checklist: ChecklistStatus = Field(..., description="체크리스트 상세 결과")
    metrics: MetricDetails = Field(..., description="측정 세부 수치")
    notes: Optional[str] = Field(None, description="추가 코멘트 또는 판정 사유")
    user_agent: Optional[str] = Field(None, description="클라이언트 환경 (iOS/Android/Browser)")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "storage": "logs.json (in-file)"
    }

@app.get("/api/logs", response_model=List[Dict[str, Any]])
async def get_logs(limit: int = 50):
    """저장된 춤 판정 로그 목록을 최신순으로 조회"""
    async with file_lock:
        if not os.path.exists(LOGS_FILE_PATH):
            return []
        try:
            with open(LOGS_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    return []
                # 최신순 정렬
                return list(reversed(data))[:limit]
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"로그 파일 읽기 오류: {str(e)}"
            )

@app.post("/api/logs", status_code=status.HTTP_201_CREATED)
async def create_log(log: DanceLogEntry):
    """새로운 춤 판정 결과를 logs.json에 비동기 락과 함께 추가"""
    entry = log.model_dump()
    if not entry.get("timestamp"):
        entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async with file_lock:
        try:
            records = []
            if os.path.exists(LOGS_FILE_PATH):
                with open(LOGS_FILE_PATH, "r", encoding="utf-8") as f:
                    try:
                        records = json.load(f)
                        if not isinstance(records, list):
                            records = []
                    except json.JSONDecodeError:
                        records = []

            records.append(entry)

            with open(LOGS_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "message": "로그가 성공적으로 기록되었습니다.",
                "total_records": len(records),
                "saved_entry": entry
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"로그 저장 중 오류 발생: {str(e)}"
            )

@app.delete("/api/logs")
async def clear_logs():
    """저장된 모든 로그 초기화 (테스트 및 리셋용)"""
    async with file_lock:
        try:
            with open(LOGS_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return {"success": True, "message": "모든 로그가 초기화되었습니다."}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"로그 초기화 오류: {str(e)}"
            )

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    """모바일 웹 앱 메인 페이지"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "DanceCheck Server is running. static/index.html not found."}

@app.get("/{file_name:path}")
async def serve_static_fallback(file_name: str):
    """루트 경로 정적 파일 요청 및 SPA fallback 대응"""
    target_path = os.path.join(STATIC_DIR, file_name)
    if os.path.isfile(target_path):
        return FileResponse(target_path)
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="File Not Found")

if __name__ == "__main__":
    import uvicorn
    # Render 및 클라우드 배포 환경의 PORT 환경 변수 지원 (기본 8000)
    port = int(os.environ.get("PORT", 8000))
    # CPU 150%, 1GB RAM 환경에 맞춘 경량 단일 워커 구동
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)
