# ⚡ DanceCheck

> **모바일 웹(iOS Safari / Android Chrome) 실시간 댄스 분석 및 객관적 판정 Hi-Fi 프로토타입 웹 애플리케이션**

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](#)
[![MediaPipe](https://img.shields.io/badge/Vision_AI-MediaPipe_Pose_Lite-00f0ff)](#)
[![Vanilla JS](https://img.shields.io/badge/Frontend-Buildless_Vanilla_JS-F7DF1E?logo=javascript&logoColor=black)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 프로젝트 개요
스마트폰 카메라를 통해 사용자의 전신 움직임을 실시간 분석하고, 사전에 정의된 객관적인 5대 체크리스트 기준에 따라 **'춤 여부(Dancing vs Non-Dancing)'**를 실시간 및 세션 단위로 판정하여 서버의 `logs.json`에 기록하는 초경량 웹 앱입니다.

---

## 🏗️ 핵심 아키텍처 및 제약 사항

1. **100% 클라이언트 연산 (Client-Side AI):**
   - 무거운 AI 추론을 서버에서 하지 않고, 모바일 브라우저의 WebGL/GPU 가속을 활용해 `@mediapipe/tasks-vision`의 `pose_landmarker_lite` 모델로 100% 처리합니다.
2. **초경량 백엔드 (Server Spec: RAM 1GB / CPU 150% 최적화):**
   - 별도 RDBMS/NoSQL 없이 Python 표준 `json` 모듈과 `asyncio.Lock`을 사용해 파일 동시성 접근을 제어합니다.
   - FastAPI 기반 초경량 정적 파일 서빙 및 비동기 REST API (`/api/logs`)를 제공합니다.
3. **빌드리스(Buildless) 프론트엔드:**
   - 별도의 번들러(Node.js, Webpack, Vite 등) 없이 Vanilla JS (ES Modules), HTML5, CSS3만으로 작성되어 즉시 배포 및 구동 가능합니다.

---

## 📋 객관적 5대 춤 판정 체크리스트 (`poseEngine.js`)

| 항목 | 판정 기준 | 세부 분석 알고리즘 |
| :--- | :--- | :--- |
| **1. 관절 안정 인식** | `is_body_detected` | 어깨(11, 12) 및 골반(23, 24) 등 핵심 코어 관절 신뢰도 0.5 이상 |
| **2. 운동 에너지 역치** | `is_moving` | 8대 주요 관절(어깨, 손목, 엉덩이, 발목)의 프레임 간 변위 속도를 계산하여 정지/단순 떨림 필터링 |
| **3. 리듬 주기성 & 바운스** | `is_rhythmic` | 골반/손목의 시계열(최근 3초) 상하 수직 진동 및 피크 간격 분산도를 측정하여 규칙적 비트/바운스 검출 및 **추정 BPM(60~200)** 산출 |
| **4. 사지 가동 범위** | `is_dynamic` | 단순 손가락 까딱임이 아닌 팔/다리의 공간 확장 궤적(Range of Motion) 확인 |
| **5. 동작 지속성** | `is_sustained` | 일회성 돌발 동작(물건 줍기 등)과 구분하기 위해 1.5초 이상 연속 리듬 유지 확인 |

---

## 📂 디렉토리 구조
```
DanceCheck/
├── main.py              # FastAPI 서버 (정적 파일 서빙 + logs.json CRUD API)
├── logs.json            # 판정 기록 저장 파일 (서버 실행 시 자동 초기화)
├── requirements.txt     # fastapi, uvicorn
├── .gitignore
├── README.md
└── static/
    ├── index.html       # 모바일 퍼스트 반응형 UI
    ├── style.css        # Cyberpunk Neon 모바일 다크 테마 (Safe Area 대응)
    ├── app.js           # 카메라 제어, 7초 챌린지 모드, UI 실시간 갱신
    └── poseEngine.js    # MediaPipe Pose Landmarker Lite 및 춤 판정 알고리즘
```

---

## 🚀 빠른 시작 (Getting Started)

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
python main.py
# 또는
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. 접속
- **PC 브라우저:** `http://localhost:8000`
- **모바일 브라우저 (iOS Safari / Android Chrome):** 같은 Wi-Fi 망에서 `http://[호스트_IP]:8000` 접속
