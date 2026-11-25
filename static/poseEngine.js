/**
 * DanceCheck - poseEngine.js
 * MediaPipe Pose Landmarker Lite 기반 실시간 브라우저 AI 및 객관적 춤 판정 엔진
 * 100% 클라이언트 연산 (모바일 브라우저 최적화)
 * - 다인원(Multi-person) 감지 지원 (최대 4인)
 * - 얼굴 포인트 제거 & 단일 머리(Head) 포인트 통합 렌더링/측정
 */

import {
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// 다인원 스켈레톤 색상 테마 팔레트 (Cyberpunk Neon)
const PERSON_PALETTES = [
  { stroke: "rgba(0, 240, 255, 0.8)", glow: "#00f0ff", head: "#00ffff", joint: "#ff007f", core: "#ffe600" },
  { stroke: "rgba(57, 255, 20, 0.8)", glow: "#39ff14", head: "#39ff14", joint: "#ffe600", core: "#00f0ff" },
  { stroke: "rgba(255, 0, 127, 0.8)", glow: "#ff007f", head: "#ff007f", joint: "#00f0ff", core: "#ffe600" },
  { stroke: "rgba(255, 170, 0, 0.8)", glow: "#ffaa00", head: "#ffaa00", joint: "#39ff14", core: "#ff007f" }
];

export class PoseEngine {
  constructor() {
    this.poseLandmarker = null;
    this.isReady = false;
    this.initError = null;

    // MediaPipe timestamp 관리 (단조 증가 필수 보장)
    this.lastTimestamp = -1;

    // 시계열 분석을 위한 링 버퍼 (약 90프레임 = 30fps 기준 3초)
    this.bufferSize = 90;
    this.history = []; // { timestamp, landmarks, energy, hipY, torsoHeight, headY }

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
      fps: 0,
      person_count: 0
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
   * 다인원 감지 (numPoses: 4) 지원
   */
  async initialize() {
    const modelAssetPath = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );

      // 다인원(numPoses: 4) 감지 및 GPU 델리게이트
      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 4,
        minPoseDetectionConfidence: 0.45,
        minPosePresenceConfidence: 0.45,
        minTrackingConfidence: 0.45
      });

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
            modelAssetPath,
            delegate: "CPU"
          },
          runningMode: "VIDEO",
          numPoses: 4
        });
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
   */
  processFrame(videoElement, timestamp) {
    if (!this.isReady || !this.poseLandmarker) {
      return null;
    }

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

    // MediaPipe timestamp 단조 증가 보장
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

    const allPoses = results?.landmarks || [];
    const personCount = allPoses.length;
    this.metrics.person_count = personCount;

    // 감지된 인원 전체 춤 판정 및 분석
    this.evaluateMultiPersonDance(allPoses, safeTimestamp);

    const isDancing = this.metrics.dance_score >= 60 &&
      this.checklist.is_body_detected &&
      this.checklist.is_moving &&
      (this.checklist.is_rhythmic || this.checklist.is_dynamic);

    return {
      allLandmarks: allPoses,
      landmarks: allPoses[0] || null, // 대표 랜드마크
      personCount,
      checklist: { ...this.checklist },
      metrics: {
        ...this.metrics,
        fps: this.currentFps
      },
      isDancing: Boolean(isDancing)
    };
  }

  /**
   * 얼굴 세부 포인트(0~10)를 제거하고 머리(Head) 단일 중심 좌표 계산
   */
  getHeadPoint(landmarks) {
    if (!landmarks) return null;

    const nose = landmarks[0];
    const earL = landmarks[7], earR = landmarks[8];
    const ls = landmarks[11], rs = landmarks[12];

    const neckX = (ls && rs) ? (ls.x + rs.x) / 2 : (nose?.x || 0.5);
    const neckY = (ls && rs) ? (ls.y + rs.y) / 2 : 0.3;

    let headX = neckX;
    let headY = neckY - 0.12; // 기본 오프셋
    let visibility = 0.5;

    // 양 귀가 감지되면 귀의 중간점을 머리 중심으로 사용
    if (earL && earR && (earL.visibility || 0) > 0.3 && (earR.visibility || 0) > 0.3) {
      headX = (earL.x + earR.x) / 2;
      headY = (earL.y + earR.y) / 2;
      visibility = (earL.visibility + earR.visibility) / 2;
    } else if (nose && (nose.visibility || 0) > 0.3) {
      headX = nose.x;
      // 코에서 위쪽으로 살짝 올려서 정수리/머리 중심점 형성
      const headOffset = Math.max(0.04, Math.abs(neckY - nose.y) * 0.4);
      headY = nose.y - headOffset;
      visibility = nose.visibility || 0.5;
    }

    return { x: headX, y: headY, visibility, neckX, neckY };
  }

  /**
   * 다인원 랜드마크 렌더링 (얼굴 디테일 제외, 단일 머리 포인트 & 깔끔한 스켈레톤)
   */
  drawLandmarks(canvasCtx, allLandmarks, width, height) {
    if (!canvasCtx) return;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, width, height);

    if (!allLandmarks || allLandmarks.length === 0) {
      canvasCtx.restore();
      return;
    }

    // 신체 관절 연결선 (얼굴 내부 0~10 제외, 순수 몸통/사지)
    const bodyConnections = [
      // 상체 및 팔
      [11, 12], // 어깨 사이
      [11, 13], [13, 15], // 왼팔
      [12, 14], [14, 16], // 오른팔
      // 몸통
      [11, 23], [12, 24], [23, 24], // 어깨-골반 및 골반 사이
      // 하체 및 다리
      [23, 25], [25, 27], // 왼다리
      [24, 26], [26, 28]  // 오른다리
    ];

    // 감지된 각 인원별 렌더링
    allLandmarks.forEach((landmarks, pIdx) => {
      const palette = PERSON_PALETTES[pIdx % PERSON_PALETTES.length];
      const head = this.getHeadPoint(landmarks);

      // 1. 머리 -> 목(어깨 중심) 척추 연결선 그리기
      if (head && head.visibility > 0.35) {
        canvasCtx.lineWidth = 3.5;
        canvasCtx.strokeStyle = palette.stroke;
        canvasCtx.shadowColor = palette.glow;
        canvasCtx.shadowBlur = 8;

        canvasCtx.beginPath();
        canvasCtx.moveTo(head.x * width, head.y * height);
        canvasCtx.lineTo(head.neckX * width, head.neckY * height);
        canvasCtx.stroke();
      }

      // 2. 몸통 및 사지 스켈레톤 선 그리기
      canvasCtx.lineWidth = 3;
      canvasCtx.strokeStyle = palette.stroke;
      canvasCtx.shadowColor = palette.glow;
      canvasCtx.shadowBlur = 6;

      for (const [startIdx, endIdx] of bodyConnections) {
        const p1 = landmarks[startIdx];
        const p2 = landmarks[endIdx];
        if (p1 && p2 && (p1.visibility || 0) > 0.35 && (p2.visibility || 0) > 0.35) {
          canvasCtx.beginPath();
          canvasCtx.moveTo(p1.x * width, p1.y * height);
          canvasCtx.lineTo(p2.x * width, p2.y * height);
          canvasCtx.stroke();
        }
      }

      // 3. 신체 관절 포인트 그리기 (얼굴 0~10 제외)
      for (let i = 11; i < landmarks.length; i++) {
        // 주요 춤 관절만 포인트 렌더링 (어깨, 팔꿈치, 손목, 골반, 무릎, 발목)
        if (![11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].includes(i)) continue;

        const lm = landmarks[i];
        if ((lm.visibility || 0) > 0.35) {
          const x = lm.x * width;
          const y = lm.y * height;

          canvasCtx.beginPath();
          if ([15, 16, 27, 28].includes(i)) {
            // 손목 / 발목: 강조 포인트
            canvasCtx.fillStyle = palette.joint;
            canvasCtx.shadowColor = palette.joint;
            canvasCtx.shadowBlur = 8;
            canvasCtx.arc(x, y, 6, 0, 2 * Math.PI);
          } else if ([11, 12, 23, 24].includes(i)) {
            // 어깨 / 골반: 코어 포인트
            canvasCtx.fillStyle = palette.core;
            canvasCtx.shadowColor = palette.core;
            canvasCtx.shadowBlur = 6;
            canvasCtx.arc(x, y, 5, 0, 2 * Math.PI);
          } else {
            canvasCtx.fillStyle = palette.head;
            canvasCtx.shadowColor = palette.head;
            canvasCtx.shadowBlur = 4;
            canvasCtx.arc(x, y, 4, 0, 2 * Math.PI);
          }
          canvasCtx.fill();
        }
      }

      // 4. 단일 머리(Head) 포인트 원형 렌더링 (얼굴 디테일 대체)
      if (head && head.visibility > 0.35) {
        const hx = head.x * width;
        const hy = head.y * height;

        // 머리 원 (단일 헤드 노드)
        canvasCtx.beginPath();
        canvasCtx.fillStyle = palette.head;
        canvasCtx.shadowColor = palette.glow;
        canvasCtx.shadowBlur = 10;
        canvasCtx.arc(hx, hy, 10, 0, 2 * Math.PI);
        canvasCtx.fill();

        // 머리 내부 코어 닷
        canvasCtx.beginPath();
        canvasCtx.fillStyle = "#ffffff";
        canvasCtx.arc(hx, hy, 4, 0, 2 * Math.PI);
        canvasCtx.fill();

        // 다인원일 때 인원 번호 배지 표시 (P1, P2 ...)
        if (allLandmarks.length > 1) {
          canvasCtx.font = "bold 10px sans-serif";
          canvasCtx.fillStyle = "#ffffff";
          canvasCtx.textAlign = "center";
          canvasCtx.fillText(`P${pIdx + 1}`, hx, hy - 14);
        }
      }
    });

    canvasCtx.restore();
  }

  /**
   * 다인원 포즈 종합 분석 및 춤 판정
   */
  evaluateMultiPersonDance(allPoses, timestamp) {
    if (!allPoses || allPoses.length === 0) {
      this.resetEvaluationState();
      return;
    }

    // 각 인원별 평가 실행
    const evaluatedPersons = allPoses.map(pose => this.evaluateSinglePerson(pose));

    // 유효한 신체(어깨+골반 감지)를 가진 인원 필터링
    const validPersons = evaluatedPersons.filter(p => p.isBodyDetected);

    if (validPersons.length === 0) {
      this.resetEvaluationState();
      return;
    }

    // 가장 활발히 춤을 추고 있는 메인 댄서 선정
    validPersons.sort((a, b) => b.danceScore - a.danceScore);
    const primary = validPersons[0];

    // 그룹 시너지: 2명 이상이 함께 춤출 경우 활성도 평균 보정
    let groupAvgScore = primary.danceScore;
    if (validPersons.length > 1) {
      const avg = validPersons.reduce((sum, p) => sum + p.danceScore, 0) / validPersons.length;
      groupAvgScore = Math.round(primary.danceScore * 0.75 + avg * 0.25);
    }

    // 메인 댄서의 시계열 버퍼 기록
    this.history.push({
      timestamp,
      energy: primary.energy,
      hipY: primary.hipY,
      headY: primary.headY,
      torsoHeight: primary.torsoHeight,
      lwY: primary.lwY,
      rwY: primary.rwY
    });

    if (this.history.length > this.bufferSize) {
      this.history.shift();
    }

    // 체크리스트 반영
    this.checklist.is_body_detected = true;
    this.checklist.is_moving = primary.isMoving;

    // 리듬 주기성 (골반 바운스 및 머리 끄덕임 배제 분석)
    const rhythmResult = this.analyzePeriodicity(this.history, primary.torsoHeight);
    this.checklist.is_rhythmic = rhythmResult.isRhythmic;
    this.metrics.periodicity_score = parseFloat(rhythmResult.score.toFixed(2));
    this.metrics.bpm_estimate = rhythmResult.bpm;

    // 가동 범위
    this.checklist.is_dynamic = primary.isDynamic;

    // 지속 시간
    if (this.checklist.is_moving && (this.checklist.is_rhythmic || this.checklist.is_dynamic)) {
      this.sustainedDanceFrames = Math.min(100, this.sustainedDanceFrames + 1.2);
    } else {
      this.sustainedDanceFrames = Math.max(0, this.sustainedDanceFrames - 2.0);
    }
    this.checklist.is_sustained = this.sustainedDanceFrames >= 35;

    // 최종 댄스 점수
    this.metrics.dance_score = Math.round(
      this.metrics.dance_score * 0.65 + groupAvgScore * 0.35
    );
    this.metrics.energy_level = primary.energy;
  }

  /**
   * 단일 인원의 랜드마크 분석 및 점수 산출
   */
  evaluateSinglePerson(landmarks) {
    const head = this.getHeadPoint(landmarks);
    const ls = landmarks[11], rs = landmarks[12];
    const lh = landmarks[23], rh = landmarks[24];
    const lw = landmarks[15], rw = landmarks[16];

    const shoulderVis = ((ls?.visibility || 0) + (rs?.visibility || 0)) / 2;
    const hipVis = ((lh?.visibility || 0) + (rh?.visibility || 0)) / 2;

    const shoulderCenterY = (ls && rs) ? (ls.y + rs.y) / 2 : 0;
    const hipCenterY = (lh && rh) ? (lh.y + rh.y) / 2 : 0;
    const torsoHeight = hipCenterY - shoulderCenterY;
    const shoulderWidth = (ls && rs) ? Math.abs(ls.x - rs.x) : 0;

    // 얼굴/상체 클로즈업 필터링
    const isCloseUp = hipVis < 0.4 ||
                      hipCenterY > 0.98 ||
                      torsoHeight < 0.10 ||
                      shoulderWidth > 0.70;

    const isBodyDetected = shoulderVis >= 0.55 && hipVis >= 0.45 && !isCloseUp;

    if (!isBodyDetected) {
      return {
        isBodyDetected: false,
        danceScore: 0,
        energy: 0,
        isMoving: false,
        isDynamic: false,
        hipY: hipCenterY,
        headY: head?.y || 0.2,
        torsoHeight: Math.max(0.12, torsoHeight),
        lwY: lw?.y || 0.5,
        rwY: rw?.y || 0.5
      };
    }

    const safeTorsoScale = Math.max(0.12, torsoHeight);

    // 사지 관절 에너지 계산 (얼굴 제외, 손목/발목/골반 중심)
    let frameEnergy = 0;
    if (this.history.length > 0) {
      const prev = this.history[this.history.length - 1];
      // 이전 손목/골반 변위 비교
      const dyLw = Math.abs((lw?.y || 0.5) - prev.lwY) / safeTorsoScale;
      const dyRw = Math.abs((rw?.y || 0.5) - prev.rwY) / safeTorsoScale;
      const dyHip = Math.abs(hipCenterY - prev.hipY) / safeTorsoScale;
      frameEnergy = (dyLw * 1.5 + dyRw * 1.5 + dyHip * 2.0) * 8;
    }

    const isMoving = frameEnergy >= 0.35;

    // 사지 가동 범위
    let armSpan = 0;
    if (lw && rw) {
      const dx = (lw.x - rw.x) / safeTorsoScale;
      const dy = (lw.y - rw.y) / safeTorsoScale;
      armSpan = Math.sqrt(dx * dx + dy * dy);
    }
    const isDynamic = armSpan >= 1.1;

    // 개인 댄스 점수
    let score = 15;
    score += Math.min(30, (frameEnergy / 2.0) * 30);
    score += isDynamic ? 25 : 10;
    if (!isMoving) score = Math.min(15, score * 0.3);

    return {
      isBodyDetected: true,
      danceScore: Math.min(100, Math.round(score)),
      energy: parseFloat(frameEnergy.toFixed(2)),
      isMoving,
      isDynamic,
      hipY: hipCenterY,
      headY: head?.y || 0.2,
      torsoHeight: safeTorsoScale,
      lwY: lw?.y || 0.5,
      rwY: rw?.y || 0.5
    };
  }

  /**
   * 시계열 데이터에서 골반 바운스 기반 주기성 검출
   */
  analyzePeriodicity(history, torsoHeight) {
    if (history.length < 30) {
      return { isRhythmic: false, score: 0, bpm: null };
    }

    const safeTorsoScale = Math.max(0.12, torsoHeight || 0.2);
    const signal = history.map(h => (h.hipY / safeTorsoScale));
    const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
    const centered = signal.map(v => v - mean);

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

    const intervals = [];
    for (let i = 1; i < peaks.length; i++) {
      intervals.push(peaks[i] - peaks[i - 1]);
    }

    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const variance = intervals.reduce((acc, val) => acc + Math.pow(val - avgInterval, 2), 0) / intervals.length;
    const stdDev = Math.sqrt(variance);

    const isValidInterval = avgInterval >= 8 && avgInterval <= 35;
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
   * 초기화
   */
  resetEvaluationState() {
    this.checklist.is_body_detected = false;
    this.checklist.is_moving = false;
    this.checklist.is_rhythmic = false;
    this.checklist.is_dynamic = false;
    this.checklist.is_sustained = false;
    this.metrics.dance_score = Math.max(0, this.metrics.dance_score - 8);
    this.metrics.energy_level = 0;
    this.metrics.bpm_estimate = null;
    this.metrics.periodicity_score = 0;
    this.sustainedDanceFrames = 0;
  }

  reset() {
    this.history = [];
    this.lastTimestamp = -1;
    this.sustainedDanceFrames = 0;
    this.metrics.dance_score = 0;
    this.metrics.energy_level = 0;
    this.metrics.bpm_estimate = null;
    this.metrics.periodicity_score = 0;
    this.metrics.person_count = 0;
    this.checklist = {
      is_body_detected: false,
      is_moving: false,
      is_rhythmic: false,
      is_dynamic: false,
      is_sustained: false
    };
  }
}
