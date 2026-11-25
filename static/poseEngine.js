/**
 * DanceCheck - poseEngine.js
 * MediaPipe Pose Landmarker Lite 기반 실시간 브라우저 AI 및 춤 판정 엔진
 * 100% 클라이언트 연산 (모바일 브라우저 최적화)
 */

import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

export class PoseEngine {
  constructor() {
    this.poseLandmarker = null;
    this.drawingUtils = null;
    this.isReady = false;
    this.initError = null;

    // 시계열 분석을 위한 링 버퍼 (약 90프레임 = 30fps 기준 3초)
    this.bufferSize = 90;
    this.history = []; // { timestamp, landmarks, energy, hipY, leftWristY, rightWristY }

    // 체크리스트 상태
    this.checklist = {
      is_body_detected: false,
      is_moving: false,
      is_rhythmic: false,
      is_dynamic: false,
      is_sustained: false
    };

    // 실시간 메트릭
    this.metrics = {
      dance_score: 0,
      energy_level: 0,
      bpm_estimate: null,
      periodicity_score: 0,
      analysis_duration_sec: 0,
      fps: 0
    };

    // FPS 계산
    this.lastFrameTime = performance.now();
    this.frameCount = 0;
    this.fpsTimer = performance.now();
    this.currentFps = 0;

    // 연속 댄스 유지 시간 추적
    this.sustainedDanceFrames = 0;
  }

  /**
   * MediaPipe Vision 태스크 리졸버 및 PoseLandmarker 초기화
   */
  async initialize(canvasElement) {
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );

      // 모바일 친화적인 lite 모델과 GPU 델리게이트 사용
      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      if (canvasElement) {
        const ctx = canvasElement.getContext("2d");
        this.drawingUtils = new DrawingUtils(ctx);
      }

      this.isReady = true;
      return true;
    } catch (err) {
      console.warn("GPU 모드 초기화 실패, CPU fallback 시도 중...", err);
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "CPU"
          },
          runningMode: "VIDEO",
          numPoses: 1
        });
        if (canvasElement) {
          const ctx = canvasElement.getContext("2d");
          this.drawingUtils = new DrawingUtils(ctx);
        }
        this.isReady = true;
        return true;
      } catch (fallbackErr) {
        console.error("PoseLandmarker 초기화 최종 실패:", fallbackErr);
        this.initError = fallbackErr.message || "MediaPipe 모델을 로드할 수 없습니다.";
        throw fallbackErr;
      }
    }
  }

  /**
   * 단일 비디오 프레임 추론 및 춤 알고리즘 실행
   * @param {HTMLVideoElement} videoElement
   * @param {number} timestamp
   * @returns {Object} { landmarks, checklist, metrics, isDancing }
   */
  processFrame(videoElement, timestamp) {
    if (!this.isReady || !this.poseLandmarker) {
      return null;
    }

    // FPS 계산
    const now = performance.now();
    this.frameCount++;
    if (now - this.fpsTimer >= 1000) {
      this.currentFps = Math.round((this.frameCount * 1000) / (now - this.fpsTimer));
      this.frameCount = 0;
      this.fpsTimer = now;
    }

    // MediaPipe 포즈 감지
    const results = this.poseLandmarker.detectForVideo(videoElement, timestamp);
    const hasPose = results && results.landmarks && results.landmarks.length > 0;

    let currentLandmarks = null;
    if (hasPose) {
      currentLandmarks = results.landmarks[0];
    }

    // 춤 판정 및 체크리스트 계산
    this.evaluateDanceMotion(currentLandmarks, timestamp);

    const isDancing = this.metrics.dance_score >= 60 &&
      this.checklist.is_body_detected &&
      this.checklist.is_moving &&
      (this.checklist.is_rhythmic || this.checklist.is_dynamic);

    return {
      landmarks: currentLandmarks,
      checklist: { ...this.checklist },
      metrics: {
        ...this.metrics,
        fps: this.currentFps
      },
      isDancing: Boolean(isDancing)
    };
  }

  /**
   * 포즈 랜드마크 렌더링
   */
  drawLandmarks(canvasCtx, landmarks, width, height) {
    if (!landmarks || !canvasCtx) return;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, width, height);

    // 관절 연결선 정의
    const connections = PoseLandmarker.POSE_CONNECTIONS;

    // 네온 스타일 스켈레톤 선 그리기
    canvasCtx.lineWidth = 3;
    canvasCtx.strokeStyle = "rgba(0, 240, 255, 0.75)";
    canvasCtx.shadowColor = "#00f0ff";
    canvasCtx.shadowBlur = 8;

    for (const conn of connections) {
      const p1 = landmarks[conn.start];
      const p2 = landmarks[conn.end];
      if (p1 && p2 && (p1.visibility || 1) > 0.4 && (p2.visibility || 1) > 0.4) {
        canvasCtx.beginPath();
        canvasCtx.moveTo(p1.x * width, p1.y * height);
        canvasCtx.lineTo(p2.x * width, p2.y * height);
        canvasCtx.stroke();
      }
    }

    // 관절 포인트 그리기
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      if ((lm.visibility || 1) > 0.4) {
        const x = lm.x * width;
        const y = lm.y * height;

        canvasCtx.beginPath();
        // 손목/발목/골반 등 주요 포인트 강조
        if ([15, 16, 27, 28].includes(i)) {
          // 손/발: 핑크 네온
          canvasCtx.fillStyle = "#ff007f";
          canvasCtx.shadowColor = "#ff007f";
          canvasCtx.arc(x, y, 6, 0, 2 * Math.PI);
        } else if ([11, 12, 23, 24].includes(i)) {
          // 어깨/골반: 옐로우 네온
          canvasCtx.fillStyle = "#ffe600";
          canvasCtx.shadowColor = "#ffe600";
          canvasCtx.arc(x, y, 5, 0, 2 * Math.PI);
        } else {
          canvasCtx.fillStyle = "#00ffff";
          canvasCtx.shadowColor = "#00ffff";
          canvasCtx.arc(x, y, 4, 0, 2 * Math.PI);
        }
        canvasCtx.fill();
      }
    }

    canvasCtx.restore();
  }

  /**
   * 랜드마크 시계열 데이터를 분석하여 체크리스트 및 댄스 점수 산출
   */
  evaluateDanceMotion(landmarks, timestamp) {
    if (!landmarks) {
      this.checklist.is_body_detected = false;
      this.checklist.is_moving = false;
      this.checklist.is_rhythmic = false;
      this.checklist.is_dynamic = false;
      this.checklist.is_sustained = false;
      this.metrics.dance_score = Math.max(0, this.metrics.dance_score - 4);
      this.metrics.energy_level = 0;
      this.sustainedDanceFrames = Math.max(0, this.sustainedDanceFrames - 2);
      return;
    }

    // 1. 관절 인식 여부 (어깨 11,12 & 엉덩이 23,24 필수 확인)
    const ls = landmarks[11], rs = landmarks[12];
    const lh = landmarks[23], rh = landmarks[24];
    const lw = landmarks[15], rw = landmarks[16]; // 손목
    const lk = landmarks[25], rk = landmarks[26]; // 무릎
    const la = landmarks[27], ra = landmarks[28]; // 발목

    const coreVisibility = ((ls?.visibility || 0) + (rs?.visibility || 0) +
                            (lh?.visibility || 0) + (rh?.visibility || 0)) / 4;
    this.checklist.is_body_detected = coreVisibility >= 0.5;

    // 골반 중심 Y 좌표 (수직 바운스 감지용)
    const hipY = (lh && rh) ? (lh.y + rh.y) / 2 : 0.5;
    const shoulderY = (ls && rs) ? (ls.y + rs.y) / 2 : 0.3;

    // 2. 운동 에너지 (프레임 간 주요 관절 변위 계산)
    let frameEnergy = 0;
    if (this.history.length > 0) {
      const prev = this.history[this.history.length - 1];
      const prevLm = prev.landmarks;

      // 주요 8대 관절 (어깨, 손목, 엉덩이, 발목) 변위 계산
      const keyIndices = [11, 12, 15, 16, 23, 24, 27, 28];
      let sumDist = 0;
      let validPoints = 0;

      for (const idx of keyIndices) {
        const currPoint = landmarks[idx];
        const prevPoint = prevLm ? prevLm[idx] : null;
        if (currPoint && prevPoint) {
          const dx = currPoint.x - prevPoint.x;
          const dy = currPoint.y - prevPoint.y;
          sumDist += Math.sqrt(dx * dx + dy * dy);
          validPoints++;
        }
      }

      if (validPoints > 0) {
        // 정규화된 속도 단위
        frameEnergy = (sumDist / validPoints) * 100;
      }
    }

    // 버퍼에 현재 프레임 기록
    this.history.push({
      timestamp,
      landmarks,
      energy: frameEnergy,
      hipY,
      shoulderY,
      lwY: lw?.y || 0.5,
      rwY: rw?.y || 0.5
    });

    if (this.history.length > this.bufferSize) {
      this.history.shift();
    }

    // 최근 버퍼 기반 분석
    const recentFrames = this.history.slice(-30); // 최근 1초
    const avgEnergy = recentFrames.reduce((acc, f) => acc + f.energy, 0) / (recentFrames.length || 1);
    this.metrics.energy_level = parseFloat(avgEnergy.toFixed(2));

    // 기준 2: 움직임 감지 (정적 자세/노이즈 이상의 에너지: 0.45 이상)
    this.checklist.is_moving = avgEnergy >= 0.45;

    // 기준 3: 리듬감 및 주기성 분석 (골반 바운스 및 손 움직임의 반복성)
    const rhythmResult = this.analyzePeriodicity(this.history);
    this.checklist.is_rhythmic = rhythmResult.isRhythmic;
    this.metrics.periodicity_score = parseFloat(rhythmResult.score.toFixed(2));
    this.metrics.bpm_estimate = rhythmResult.bpm;

    // 기준 4: 가동 범위 (Dynamic Range of Motion)
    // 손목과 어깨, 발목의 공간 분산도 분석
    const dynamicResult = this.analyzeRangeOfMotion(this.history);
    this.checklist.is_dynamic = dynamicResult.isDynamic;

    // 기준 5: 연속성 및 지속 시간 (Sustained motion)
    if (this.checklist.is_moving && (this.checklist.is_rhythmic || this.checklist.is_dynamic)) {
      this.sustainedDanceFrames = Math.min(120, this.sustainedDanceFrames + 1);
    } else {
      this.sustainedDanceFrames = Math.max(0, this.sustainedDanceFrames - 1.5);
    }
    // 약 1.5초(45프레임) 이상 지속되었는가
    this.checklist.is_sustained = this.sustainedDanceFrames >= 40;

    // 종합 댄스 점수 산출 (가중 평균)
    let score = 0;
    if (this.checklist.is_body_detected) score += 15;
    // 에너지 점수 (최대 25점)
    score += Math.min(25, (avgEnergy / 2.0) * 25);
    // 리듬 주기성 점수 (최대 35점)
    score += rhythmResult.score * 35;
    // 동적 가동 범위 점수 (최대 15점)
    score += dynamicResult.score * 15;
    // 지속성 점수 (최대 10점)
    score += Math.min(10, (this.sustainedDanceFrames / 40) * 10);

    // 스무딩 적용
    this.metrics.dance_score = Math.round(
      this.metrics.dance_score * 0.7 + Math.min(100, Math.max(0, score)) * 0.3
    );
  }

  /**
   * 시계열 데이터에서 주기적 리듬(바운스, 스윙) 검출 및 BPM 추정
   */
  analyzePeriodicity(history) {
    if (history.length < 30) {
      return { isRhythmic: false, score: 0, bpm: null };
    }

    // 신호: 골반 y축 변위 + 양손 y축 변위 합산 신호
    const signal = history.map(h => (h.hipY * 2.0) + (h.lwY * 0.5) + (h.rwY * 0.5));
    const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
    const centered = signal.map(v => v - mean);

    // 단순화된 로컬 피크(극소/극대) 검출
    const peaks = [];
    for (let i = 2; i < centered.length - 2; i++) {
      if (centered[i] > centered[i - 1] && centered[i] > centered[i - 2] &&
          centered[i] > centered[i + 1] && centered[i] > centered[i + 2] &&
          centered[i] > 0.015) { // 의미 있는 진폭
        peaks.push(i);
      }
    }

    if (peaks.length < 3) {
      return { isRhythmic: false, score: 0.1, bpm: null };
    }

    // 피크 간격(프레임 수) 계산
    const intervals = [];
    for (let i = 1; i < peaks.length; i++) {
      intervals.push(peaks[i] - peaks[i - 1]);
    }

    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;

    // 분산 계산 (일정한 간격일수록 분산이 작음 = 규칙적인 리듬)
    const variance = intervals.reduce((acc, val) => acc + Math.pow(val - avgInterval, 2), 0) / intervals.length;
    const stdDev = Math.sqrt(variance);

    // 30fps 기준 유효 댄스 바운스 간격 (7프레임 ~ 35프레임 -> 50 ~ 250 BPM)
    const isValidDanceInterval = avgInterval >= 7 && avgInterval <= 40;
    const regularity = Math.max(0, 1 - (stdDev / (avgInterval || 1)));

    let bpm = null;
    if (avgInterval > 0) {
      // 30fps 기준: (30 / avgInterval) * 60
      const approxFps = this.currentFps > 10 ? this.currentFps : 30;
      bpm = Math.round((approxFps / avgInterval) * 60);
    }

    const isRhythmic = isValidDanceInterval && regularity >= 0.55 && peaks.length >= 3;
    const score = isRhythmic ? Math.min(1.0, regularity * 0.9 + 0.1) : (regularity * 0.4);

    return {
      isRhythmic,
      score: Math.min(1.0, Math.max(0, score)),
      bpm: isRhythmic ? bpm : null
    };
  }

  /**
   * 상하좌우 관절 가동 범위(Range of Motion) 계산
   */
  analyzeRangeOfMotion(history) {
    if (history.length < 15) {
      return { isDynamic: false, score: 0 };
    }

    const last = history[history.length - 1].landmarks;
    if (!last) return { isDynamic: false, score: 0 };

    // 양 손목과 어깨 간의 상대적 거리 (사지 확장도)
    const ls = last[11], rs = last[12];
    const lw = last[15], rw = last[16];

    let armSpan = 0;
    if (lw && rw) {
      const dx = lw.x - rw.x;
      const dy = lw.y - rw.y;
      armSpan = Math.sqrt(dx * dx + dy * dy);
    }

    // 최근 프레임 손목 Y 궤적 변동폭
    const lwYs = history.map(h => h.lwY).filter(Boolean);
    const rwYs = history.map(h => h.rwY).filter(Boolean);
    const rangeLw = Math.max(...lwYs) - Math.min(...lwYs);
    const rangeRw = Math.max(...rwYs) - Math.min(...rwYs);
    const maxHandSweep = Math.max(rangeLw, rangeRw);

    // 단순 정지 상태가 아닌 손/팔의 유의미한 가동 (0.15 이상)
    const isDynamic = maxHandSweep >= 0.12 || armSpan >= 0.45;
    const score = Math.min(1.0, Math.max(0, (maxHandSweep / 0.3) * 0.7 + (armSpan / 0.6) * 0.3));

    return { isDynamic, score };
  }

  /**
   * 현재 상태 초기화
   */
  reset() {
    this.history = [];
    this.sustainedDanceFrames = 0;
    this.metrics.dance_score = 0;
    this.metrics.energy_level = 0;
    this.metrics.bpm_estimate = null;
    this.metrics.periodicity_score = 0;
    this.checklist = {
      is_body_detected: false,
      is_moving: false,
      is_rhythmic: false,
      is_dynamic: false,
      is_sustained: false
    };
  }
}
