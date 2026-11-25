/**
 * DanceCheck - poseEngine.js
 * MediaPipe Pose Landmarker Lite 기반 실시간 브라우저 AI 및 객관적 춤 판정 엔진
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

    // MediaPipe timestamp 관리 (단조 증가 필수 보장)
    this.lastTimestamp = -1;

    // 시계열 분석을 위한 링 버퍼 (약 90프레임 = 30fps 기준 3초)
    this.bufferSize = 90;
    this.history = []; // { timestamp, landmarks, energy, hipY, torsoHeight }

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

    // 비디오 해상도가 아직 준비되지 않았으면 안전하게 스킵
    if (!videoElement || videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
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

    // MediaPipe는 timestamp가 반드시 엄격하게 단조 증가(strictly increasing)해야 함
    let safeTimestamp = timestamp;
    if (safeTimestamp <= this.lastTimestamp) {
      safeTimestamp = this.lastTimestamp + 1;
    }
    this.lastTimestamp = safeTimestamp;

    let results = null;
    try {
      results = this.poseLandmarker.detectForVideo(videoElement, safeTimestamp);
    } catch (e) {
      console.warn("detectForVideo warning:", e);
      return null;
    }

    const hasPose = results && results.landmarks && results.landmarks.length > 0;
    let currentLandmarks = null;
    if (hasPose) {
      currentLandmarks = results.landmarks[0];
    }

    // 춤 판정 및 체크리스트 계산
    this.evaluateDanceMotion(currentLandmarks, safeTimestamp);

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
    // 랜드마크가 전혀 없으면 즉시 모든 상태 초기화 및 0점
    if (!landmarks) {
      this.resetEvaluationState();
      return;
    }

    // 주요 랜드마크 추출
    // 얼굴: 0(코)
    // 어깨: 11(좌), 12(우)
    // 손목: 15(좌), 16(우)
    // 골반: 23(좌), 24(우)
    // 무릎: 25(좌), 26(우)
    // 발목: 27(좌), 28(우)
    const nose = landmarks[0];
    const ls = landmarks[11], rs = landmarks[12];
    const lw = landmarks[15], rw = landmarks[16];
    const lh = landmarks[23], rh = landmarks[24];
    const lk = landmarks[25], rk = landmarks[26];
    const la = landmarks[27], ra = landmarks[28];

    // =========================================================================
    // [핵심 필터 1] 얼굴 클로즈업 배제 및 전신/상체 관절 필수 감지
    // =========================================================================
    const shoulderVis = ((ls?.visibility || 0) + (rs?.visibility || 0)) / 2;
    const hipVis = ((lh?.visibility || 0) + (rh?.visibility || 0)) / 2;

    // 양 어깨와 양 골반의 좌표
    const shoulderCenterY = (ls && rs) ? (ls.y + rs.y) / 2 : 0;
    const shoulderCenterX = (ls && rs) ? (ls.x + rs.x) / 2 : 0;
    const hipCenterY = (lh && rh) ? (lh.y + rh.y) / 2 : 0;
    const hipCenterX = (lh && rh) ? (lh.x + rh.x) / 2 : 0;

    // 몸통 길이(Torso Height): 어깨 중심 ~ 골반 중심
    const torsoHeight = hipCenterY - shoulderCenterY;

    // 어깨 너비
    const shoulderWidth = (ls && rs) ? Math.abs(ls.x - rs.x) : 0;

    // 얼굴 클로즈업 판별 조건:
    // 1) 골반 가시성이 0.45 미만이거나, 골반이 화면 하단 밖(> 0.96)에 잘린 경우
    // 2) 몸통 길이(torsoHeight)가 0.12 미만으로 너무 작거나 역전된 경우
    // 3) 어깨 너비가 화면 가로의 65% 이상을 차지할 정도로 카메라에 바짝 다가온 경우
    // 4) 코의 Y좌표와 어깨 Y좌표의 거리가 몸통 길이보다 긴 경우 (얼굴만 꽉 찬 화면)
    const isCloseUp = hipVis < 0.45 ||
                      hipCenterY > 0.96 ||
                      torsoHeight < 0.12 ||
                      shoulderWidth > 0.65;

    // 관절 인식 여부 (어깨와 골반이 화면 내에 안정적으로 잡혀야 함)
    const isCoreVisible = shoulderVis >= 0.6 && hipVis >= 0.5 && !isCloseUp;
    this.checklist.is_body_detected = isCoreVisible;

    // 몸이 제대로 감지되지 않았거나(얼굴만 비춤) 클로즈업이면 춤 판정 즉시 불인정(0점)
    if (!this.checklist.is_body_detected) {
      this.resetEvaluationState();
      return;
    }

    // =========================================================================
    // [핵심 필터 2] 스케일 불변 운동 에너지 (몸통 크기 정규화 & 얼굴 움직임 배제)
    // =========================================================================
    let frameEnergy = 0;
    if (this.history.length > 0) {
      const prev = this.history[this.history.length - 1];
      const prevLm = prev.landmarks;

      // 머리/얼굴(0~10)은 전면 제외하고, 오직 사지 및 몸통 관절만 추적!
      // 손목(15,16)에 높은 가중치, 골반(23,24)/발목(27,28)/팔꿈치(13,14) 포함
      const keyJoints = [
        { idx: 15, weight: 1.5 }, // 좌 손목
        { idx: 16, weight: 1.5 }, // 우 손목
        { idx: 13, weight: 1.0 }, // 좌 팔꿈치
        { idx: 14, weight: 1.0 }, // 우 팔꿈치
        { idx: 23, weight: 1.2 }, // 좌 골반
        { idx: 24, weight: 1.2 }, // 우 골반
        { idx: 25, weight: 1.0 }, // 좌 무릎
        { idx: 26, weight: 1.0 }, // 우 무릎
        { idx: 27, weight: 1.3 }, // 좌 발목
        { idx: 28, weight: 1.3 }  // 우 발목
      ];

      let weightedDistSum = 0;
      let totalWeight = 0;

      // 카메라 거리와 무관하도록 모든 변위를 torsoHeight로 정규화
      const safeTorsoScale = Math.max(0.12, torsoHeight);

      for (const item of keyJoints) {
        const currPoint = landmarks[item.idx];
        const prevPoint = prevLm ? prevLm[item.idx] : null;

        if (currPoint && prevPoint && (currPoint.visibility || 0) > 0.35) {
          const dx = (currPoint.x - prevPoint.x) / safeTorsoScale;
          const dy = (currPoint.y - prevPoint.y) / safeTorsoScale;
          const dist = Math.sqrt(dx * dx + dy * dy);
          weightedDistSum += dist * item.weight;
          totalWeight += item.weight;
        }
      }

      if (totalWeight > 0) {
        // 몸통 대비 이동 비율 (1.0 = 몸통 길이만큼 1프레임에 이동)
        frameEnergy = (weightedDistSum / totalWeight) * 10;
      }
    }

    // 시계열 버퍼에 기록
    this.history.push({
      timestamp,
      landmarks,
      energy: frameEnergy,
      hipY: hipCenterY,
      torsoHeight,
      lwY: lw?.y || 0.5,
      rwY: rw?.y || 0.5
    });

    if (this.history.length > this.bufferSize) {
      this.history.shift();
    }

    // 최근 프레임 평균 에너지 계산 (최근 1초)
    const recentFrames = this.history.slice(-30);
    const avgEnergy = recentFrames.reduce((acc, f) => acc + f.energy, 0) / (recentFrames.length || 1);
    this.metrics.energy_level = parseFloat(avgEnergy.toFixed(2));

    // 기준 2: 유의미한 움직임 감지
    // 정적 자세/미세 숨쉬기(0.2 미만) 필터링 -> 0.35 이상일 때만 움직임 인정
    this.checklist.is_moving = avgEnergy >= 0.35;

    // =========================================================================
    // [기준 3] 리듬 주기성 & 수직 바운스 분석 (얼굴 끄덕임 제외, 골반 바운스 기반)
    // =========================================================================
    const rhythmResult = this.analyzePeriodicity(this.history, torsoHeight);
    this.checklist.is_rhythmic = rhythmResult.isRhythmic;
    this.metrics.periodicity_score = parseFloat(rhythmResult.score.toFixed(2));
    this.metrics.bpm_estimate = rhythmResult.bpm;

    // =========================================================================
    // [기준 4] 사지 가동 범위 (Range of Motion)
    // =========================================================================
    const dynamicResult = this.analyzeRangeOfMotion(this.history, torsoHeight);
    this.checklist.is_dynamic = dynamicResult.isDynamic;

    // =========================================================================
    // [기준 5] 연속성 및 지속 시간 (Sustained Motion)
    // =========================================================================
    if (this.checklist.is_moving && (this.checklist.is_rhythmic || this.checklist.is_dynamic)) {
      this.sustainedDanceFrames = Math.min(100, this.sustainedDanceFrames + 1.2);
    } else {
      this.sustainedDanceFrames = Math.max(0, this.sustainedDanceFrames - 2.0);
    }
    this.checklist.is_sustained = this.sustainedDanceFrames >= 35; // 약 1.2초 이상 지속

    // =========================================================================
    // [종합 댄스 점수 산출]
    // =========================================================================
    let score = 0;
    // 1. 관절 안정 감지 (10점)
    score += 10;
    // 2. 운동 에너지 (최대 25점: avgEnergy 0.35 ~ 2.0 범위)
    const energyScore = Math.min(25, Math.max(0, ((avgEnergy - 0.25) / 1.5) * 25));
    score += energyScore;
    // 3. 리듬 주기성 (최대 35점)
    score += rhythmResult.score * 35;
    // 4. 가동 범위 (최대 18점)
    score += dynamicResult.score * 18;
    // 5. 지속성 (최대 12점)
    score += Math.min(12, (this.sustainedDanceFrames / 35) * 12);

    // 움직이지 않거나 리듬/가동범위가 전혀 없으면 점수를 대폭 억제
    if (!this.checklist.is_moving) {
      score = Math.min(15, score * 0.3);
    }

    // 스무딩 적용
    this.metrics.dance_score = Math.round(
      this.metrics.dance_score * 0.65 + Math.min(100, Math.max(0, score)) * 0.35
    );
  }

  /**
   * 시계열 데이터에서 골반/손목의 주기적 바운스 검출 및 BPM 추정
   */
  analyzePeriodicity(history, torsoHeight) {
    if (history.length < 30) {
      return { isRhythmic: false, score: 0, bpm: null };
    }

    const safeTorsoScale = Math.max(0.12, torsoHeight || 0.2);

    // 골반의 Y 변화(수직 바운스)를 torsoHeight로 정규화한 신호
    const signal = history.map(h => (h.hipY / safeTorsoScale));
    const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
    const centered = signal.map(v => v - mean);

    // 피크 검출: 단순 떨림이 아닌 몸통 길이의 최소 3.5% 이상의 바운스
    const minAmplitude = 0.035;
    const peaks = [];

    for (let i = 2; i < centered.length - 2; i++) {
      if (centered[i] > centered[i - 1] && centered[i] > centered[i - 2] &&
          centered[i] > centered[i + 1] && centered[i] > centered[i + 2] &&
          centered[i] > minAmplitude) {
        peaks.push(i);
      }
    }

    if (peaks.length < 3) {
      return { isRhythmic: false, score: 0, bpm: null };
    }

    // 피크 간격 계산
    const intervals = [];
    for (let i = 1; i < peaks.length; i++) {
      intervals.push(peaks[i] - peaks[i - 1]);
    }

    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const variance = intervals.reduce((acc, val) => acc + Math.pow(val - avgInterval, 2), 0) / intervals.length;
    const stdDev = Math.sqrt(variance);

    // 댄스 유효 인터벌: 8프레임 ~ 35프레임 (약 55 ~ 220 BPM)
    const isValidInterval = avgInterval >= 8 && avgInterval <= 35;
    // 규칙성 (표준편차가 작을수록 박자가 일정)
    const regularity = Math.max(0, 1 - (stdDev / (avgInterval || 1)));

    let bpm = null;
    if (avgInterval > 0) {
      const approxFps = this.currentFps > 10 ? this.currentFps : 30;
      bpm = Math.round((approxFps / avgInterval) * 60);
    }

    const isRhythmic = isValidInterval && regularity >= 0.55 && peaks.length >= 3;
    const score = isRhythmic ? Math.min(1.0, regularity * 0.9 + 0.1) : (regularity * 0.3);

    return {
      isRhythmic,
      score: Math.min(1.0, Math.max(0, score)),
      bpm: isRhythmic ? bpm : null
    };
  }

  /**
   * 상하좌우 관절 가동 범위 (Range of Motion) 계산
   */
  analyzeRangeOfMotion(history, torsoHeight) {
    if (history.length < 15) {
      return { isDynamic: false, score: 0 };
    }

    const safeTorsoScale = Math.max(0.12, torsoHeight || 0.2);
    const last = history[history.length - 1].landmarks;
    if (!last) return { isDynamic: false, score: 0 };

    // 양 손목 거리
    const lw = last[15], rw = last[16];
    let armSpan = 0;
    if (lw && rw) {
      const dx = (lw.x - rw.x) / safeTorsoScale;
      const dy = (lw.y - rw.y) / safeTorsoScale;
      armSpan = Math.sqrt(dx * dx + dy * dy);
    }

    // 최근 15프레임 손목 수직 궤적 진폭
    const lwYs = history.map(h => h.lwY).filter(Boolean);
    const rwYs = history.map(h => h.rwY).filter(Boolean);
    const rangeLw = (Math.max(...lwYs) - Math.min(...lwYs)) / safeTorsoScale;
    const rangeRw = (Math.max(...rwYs) - Math.min(...rwYs)) / safeTorsoScale;
    const maxSweep = Math.max(rangeLw, rangeRw);

    // 몸통 대비 충분한 팔 궤적(0.4 이상) 또는 팔 벌림(1.2 이상)
    const isDynamic = maxSweep >= 0.35 || armSpan >= 1.2;
    const score = Math.min(1.0, Math.max(0, (maxSweep / 0.8) * 0.6 + (armSpan / 2.0) * 0.4));

    return { isDynamic, score };
  }

  /**
   * 감지 실패 또는 클로즈업 시 상태 안전 초기화
   */
  resetEvaluationState() {
    this.checklist.is_body_detected = false;
    this.checklist.is_moving = false;
    this.checklist.is_rhythmic = false;
    this.checklist.is_dynamic = false;
    this.checklist.is_sustained = false;
    // 점수를 빠르게 0으로 감소
    this.metrics.dance_score = Math.max(0, this.metrics.dance_score - 8);
    this.metrics.energy_level = 0;
    this.metrics.bpm_estimate = null;
    this.metrics.periodicity_score = 0;
    this.sustainedDanceFrames = 0;
  }

  /**
   * 카메라 전환 및 재시작 시 엔진 완전 초기화
   */
  reset() {
    this.history = [];
    this.lastTimestamp = -1;
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
