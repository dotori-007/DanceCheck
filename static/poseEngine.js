/**
 * DanceCheck - poseEngine.js
 * MediaPipe Pose Landmarker Lite 기반 실시간 브라우저 AI 및 객관적 춤 판정 엔진
 * 100% 클라이언트 연산 (모바일 가로/세로 모드 완벽 대응)
 * 
 * 주요 개선:
 * 1. [1인 다중 중복 인식 차단(NMS)] 근접 중복 포즈 필터링으로 1명이 여러 명으로 뜨는 고스트 현상 제거
 * 2. [가로모드 완벽 대응] Video Aspect Ratio 등방성 거리 계산 및 화면 회전 시 안정적 추적
 * 3. [최근 3초 롤링 윈도우 평균 판정] 순간적 멈춤/오차에 흔들리지 않고 3초간의 체크리스트 평균 충족률 및 평균 댄스 점수로 춤 여부 확정
 * 4. [FPS 최적화] Canvas shadowBlur 배제 및 듀얼 스트로크 초고속 렌더링 유지
 */

import {
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// 다인원 스켈레톤 네온 테마 팔레트 (Cyberpunk Neon)
const PERSON_PALETTES = [
  { stroke: "#00f0ff", glowLine: "rgba(0, 240, 255, 0.35)", head: "#00ffff", joint: "#ff007f", core: "#ffe600" },
  { stroke: "#39ff14", glowLine: "rgba(57, 255, 20, 0.35)", head: "#39ff14", joint: "#ffe600", core: "#00f0ff" },
  { stroke: "#ff007f", glowLine: "rgba(255, 0, 127, 0.35)", head: "#ff007f", joint: "#00f0ff", core: "#ffe600" },
  { stroke: "#ffaa00", glowLine: "rgba(255, 170, 0, 0.35)", head: "#ffaa00", joint: "#39ff14", core: "#ff007f" }
];

/**
 * 인원별 독립 시계열 움직임 추적 클래스
 */
class PersonTrack {
  constructor(id) {
    this.id = id;
    this.lastSeen = performance.now();
    this.lostFrames = 0;

    // 시계열 링 버퍼 (약 60프레임 = 2초)
    this.history = [];
    this.maxHistory = 60;

    // 관절 이전 위치
    this.prevJoints = null;
    this.torsoHeight = 0.2;

    // 에너지 및 지속 시간
    this.instantEnergy = 0;
    this.smoothEnergy = 0;
    this.danceHoldFrames = 0;
    this.sustainedDanceFrames = 0;

    // 평가 상태
    this.isBodyDetected = false;
    this.isMoving = false;
    this.isRhythmic = false;
    this.isDynamic = false;
    this.isSustained = false;
    this.isDancing = false;
    this.danceScore = 0;
    this.bpm = null;
  }

  update(landmarks, timestamp, head, aspectRatio = 1.0) {
    this.lastSeen = performance.now();
    this.lostFrames = 0;

    const ls = landmarks[11], rs = landmarks[12];
    const lh = landmarks[23], rh = landmarks[24];
    const lw = landmarks[15], rw = landmarks[16];
    const le = landmarks[13], re = landmarks[14];

    const shoulderVis = ((ls?.visibility || 0) + (rs?.visibility || 0)) / 2;
    const hipVis = ((lh?.visibility || 0) + (rh?.visibility || 0)) / 2;

    const shoulderCenterY = (ls && rs) ? (ls.y + rs.y) / 2 : 0;
    const hipCenterY = (lh && rh) ? (lh.y + rh.y) / 2 : 0;
    const torsoH = hipCenterY - shoulderCenterY;
    const shoulderWidth = (ls && rs) ? Math.abs((ls.x - rs.x) * aspectRatio) : 0;

    // 전신 코어 검출 유효성 검사 (가로/세로 모드 비율 정규화 적용)
    // 가로 모드에서도 어깨와 골반이 보이면 정상 인식되도록 임계치 최적화
    const isCloseUp = hipVis < 0.35 ||
                      hipCenterY > 0.99 ||
                      torsoH < 0.08 ||
                      shoulderWidth > 0.85;

    this.isBodyDetected = shoulderVis >= 0.45 && hipVis >= 0.35 && !isCloseUp;

    if (!this.isBodyDetected) {
      this.instantEnergy = 0;
      this.smoothEnergy *= 0.85;
      this.danceHoldFrames = Math.max(0, this.danceHoldFrames - 2);
      this.danceScore = Math.max(0, this.danceScore - 4);
      this.isDancing = false;
      this.isMoving = false;
      return;
    }

    this.torsoHeight = Math.max(0.10, torsoH);
    const scale = this.torsoHeight;

    // 사지 및 코어 2D 등방성 변위 에너지 계산 (aspectRatio 적용)
    // [핵심] 카메라 센서 노이즈/손떨림 데드밴드(Deadband Filter) 적용
    // 정지 상태 시 미세 플리커(< threshold)를 완전 0으로 클램핑하여 오인식 원천 차단
    let currentEnergy = 0;
    let wristVelocity = 0;
    if (this.prevJoints) {
      const deadband = (dist, thresh) => (dist < thresh ? 0 : dist - thresh * 0.7);

      const rawLw = Math.hypot((lw?.x - this.prevJoints.lwX) * aspectRatio, (lw?.y - this.prevJoints.lwY)) / scale;
      const rawRw = Math.hypot((rw?.x - this.prevJoints.rwX) * aspectRatio, (rw?.y - this.prevJoints.rwY)) / scale;
      const rawLe = Math.hypot((le?.x - this.prevJoints.leX) * aspectRatio, (le?.y - this.prevJoints.leY)) / scale;
      const rawRe = Math.hypot((re?.x - this.prevJoints.reX) * aspectRatio, (re?.y - this.prevJoints.reY)) / scale;
      const rawHip = Math.hypot(((lh?.x + rh?.x)/2 - this.prevJoints.hipX) * aspectRatio, hipCenterY - this.prevJoints.hipY) / scale;
      const rawHead = head ? Math.hypot((head.x - this.prevJoints.headX) * aspectRatio, head.y - this.prevJoints.headY) / scale : 0;

      // 관절별 미세 노이즈 데드밴드 (가만히 서 있을 때의 센서 지터 완전 제거)
      const dLw = deadband(rawLw, 0.024);
      const dRw = deadband(rawRw, 0.024);
      const dLe = deadband(rawLe, 0.018);
      const dRe = deadband(rawRe, 0.018);
      const dHip = deadband(rawHip, 0.015);
      const dHead = deadband(rawHead, 0.014);

      wristVelocity = (dLw + dRw) / 2;
      currentEnergy = (dLw * 1.5 + dRw * 1.5 + dLe * 0.9 + dRe * 0.9 + dHip * 2.2 + dHead * 0.8) * 5.0;
    }

    this.instantEnergy = parseFloat(currentEnergy.toFixed(2));

    this.prevJoints = {
      lwX: lw?.x || 0, lwY: lw?.y || 0,
      rwX: rw?.x || 0, rwY: rw?.y || 0,
      leX: le?.x || 0, leY: le?.y || 0,
      reX: re?.x || 0, reY: re?.y || 0,
      hipX: (lh && rh) ? (lh.x + rh.x) / 2 : 0,
      hipY: hipCenterY,
      headX: head?.x || 0,
      headY: head?.y || 0
    };

    // 시계열 링 버퍼
    this.history.push({
      timestamp,
      hipY: hipCenterY,
      lwY: lw?.y || 0,
      rwY: rw?.y || 0,
      energy: this.instantEnergy
    });
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }

    // 에너지 지수 평활 (상승 즉시 반영, 하강 부드러운 감쇄)
    if (this.instantEnergy > this.smoothEnergy) {
      this.smoothEnergy = this.smoothEnergy * 0.15 + this.instantEnergy * 0.85;
    } else {
      this.smoothEnergy = this.smoothEnergy * 0.90 + this.instantEnergy * 0.10;
    }

    // 움직임 판정 (노이즈 데드밴드 통과 후 유효 움직임 여부)
    this.isMoving = this.smoothEnergy >= 0.45 || this.instantEnergy >= 0.55;

    // 사지 가동 범위 (Dynamic Range)
    // [핵심] 가만히 서 있는 정적 자세(Static Pose) 제외!
    // 실제로 팔이나 스텝을 활발히 뻗고 움직이는 동작 중에만 인정
    let armSpan = 0;
    if (lw && rw) {
      armSpan = Math.hypot((lw.x - rw.x) * aspectRatio / scale, (lw.y - rw.y) / scale);
    }
    const handsHigh = (lw && ls && lw.y < ls.y) || (rw && rs && rw.y < rs.y);
    const hasLimbMotion = wristVelocity >= 0.035 || this.instantEnergy >= 0.80;

    this.isDynamic = this.isMoving && (
      (handsHigh && hasLimbMotion) ||
      (armSpan >= 1.25 && hasLimbMotion) ||
      (this.instantEnergy >= 1.6)
    );

    // 리듬 및 비트 주기성 판정 (움직임 게이팅 적용)
    const rhythm = this.evaluateRhythm();
    this.isRhythmic = rhythm.isRhythmic;
    this.bpm = rhythm.bpm;

    // 춤 상태 유지 래치: 실제 댄스 동작일 때만 활성화
    const isDanceMotion = this.isMoving && (
      this.instantEnergy >= 0.85 ||
      this.smoothEnergy >= 0.70 ||
      (this.isRhythmic && this.smoothEnergy >= 0.50) ||
      (this.isDynamic && this.smoothEnergy >= 0.55)
    );

    if (isDanceMotion) {
      this.danceHoldFrames = 40;
      this.sustainedDanceFrames = Math.min(120, this.sustainedDanceFrames + 1.5);
    } else if (this.danceHoldFrames > 0) {
      this.danceHoldFrames--;
      this.sustainedDanceFrames = Math.max(0, this.sustainedDanceFrames - 0.7);
    } else {
      this.sustainedDanceFrames = Math.max(0, this.sustainedDanceFrames - 2.0);
    }

    this.isSustained = this.sustainedDanceFrames >= 30;

    // 댄스 점수 산출 (3-Tier 구조: 정지 0~25점 / 단순동작 26~49점 / 댄스 50~100점)
    let score = 0;
    if (this.isMoving) {
      score += 15; // 유효 움직임 기본 점수
      score += Math.min(45, (this.smoothEnergy / 1.8) * 45);
      if (this.isDynamic) score += 15;
      if (this.isRhythmic) score += 15;
      if (this.isSustained) score += 10;
    } else {
      // 정지 상태: 0~15점 미만으로 즉시 감쇄
      score = Math.min(15, this.smoothEnergy * 25);
    }

    this.danceScore = Math.min(100, Math.round(score));
    this.isDancing = this.isBodyDetected && (
      (this.danceHoldFrames > 0 && this.danceScore >= 50 && this.isMoving) ||
      (this.danceScore >= 55 && (this.isRhythmic || (this.isDynamic && this.isSustained)))
    );
  }

  evaluateRhythm() {
    // 움직임이 없거나 에너지가 미흡한 정지 상태에서는 리듬 분석 즉시 차단 (가짜 비트 방지)
    if (!this.isMoving || this.smoothEnergy < 0.40 || this.history.length < 24) {
      return { isRhythmic: false, bpm: null };
    }

    const scale = this.torsoHeight || 0.2;
    const signal = this.history.map(h => (h.hipY + (h.lwY + h.rwY) * 0.35) / scale);
    const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
    const centered = signal.map(v => v - mean);

    // 진폭(Peak-to-Peak) 검사: 미세 숨쉬기/손떨림 노이즈 파형 차단
    const maxVal = Math.max(...centered);
    const minVal = Math.min(...centered);
    if ((maxVal - minVal) < 0.07) {
      return { isRhythmic: false, bpm: null };
    }

    const peaks = [];
    const minThreshold = 0.035;

    for (let i = 2; i < centered.length - 2; i++) {
      if (centered[i] > centered[i - 1] && centered[i] > centered[i - 2] &&
          centered[i] > centered[i + 1] && centered[i] > centered[i + 2] &&
          centered[i] > minThreshold) {
        peaks.push(i);
      }
    }

    if (peaks.length < 3) {
      return { isRhythmic: false, bpm: null };
    }

    const intervals = [];
    for (let i = 1; i < peaks.length; i++) {
      intervals.push(peaks[i] - peaks[i - 1]);
    }

    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const variance = intervals.reduce((acc, v) => acc + Math.pow(v - avgInterval, 2), 0) / intervals.length;
    const stdDev = Math.sqrt(variance);

    const isIntervalValid = avgInterval >= 7 && avgInterval <= 36;
    const regularity = Math.max(0, 1 - (stdDev / (avgInterval || 1)));

    let bpm = null;
    if (avgInterval > 0) {
      bpm = Math.round((30 / avgInterval) * 60);
      if (bpm < 50 || bpm > 240) bpm = null;
    }

    const isRhythmic = isIntervalValid && regularity >= 0.50 && peaks.length >= 3;
    return { isRhythmic, bpm: isRhythmic ? bpm : null };
  }
}

export class PoseEngine {
  constructor() {
    this.poseLandmarker = null;
    this.isReady = false;
    this.initError = null;

    // MediaPipe timestamp 관리
    this.lastTimestamp = -1;

    // 인원별 독립 트랙 관리 (trackId -> PersonTrack)
    this.tracks = new Map();
    this.nextTrackId = 1;

    // [핵심] 최근 3초(3000ms) 롤링 윈도우 기록 버퍼
    this.windowDurationMs = 3000;
    this.rollingWindow = []; // { timestamp, body, moving, rhythm, dynamic, sustained, score, energy, isDancing }

    // 3초 롤링 윈도우 평균 메트릭
    this.windowMetrics = {
      is_dancing: false,
      verdict_text: "인물 감지 대기 중",
      avg_score: 0,
      avg_energy: 0,
      active_count: 0, // 평균 충족 항목 수 (0~5)
      rates: {
        body: 0,     // 0 ~ 100%
        moving: 0,   // 0 ~ 100%
        rhythm: 0,   // 0 ~ 100%
        dynamic: 0,  // 0 ~ 100%
        sustained: 0 // 0 ~ 100%
      }
    };

    // 순간 체크리스트 상태
    this.checklist = {
      is_body_detected: false,
      is_moving: false,
      is_rhythmic: false,
      is_dynamic: false,
      is_sustained: false
    };

    // 실시간 종합 메트릭
    this.metrics = {
      dance_score: 0,
      energy_level: 0,
      bpm_estimate: null,
      periodicity_score: 0,
      analysis_duration_sec: 0,
      fps: 0,
      person_count: 0,
      dancing_person_count: 0
    };

    // FPS 계산
    this.frameCount = 0;
    this.fpsTimer = performance.now();
    this.currentFps = 0;
  }

  /**
   * MediaPipe Vision 태스크 리졸버 및 PoseLandmarker 초기화
   * confidence를 0.55로 상향하여 불필요한 유령 박스 생성 억제
   */
  async initialize() {
    const modelAssetPath = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );

      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 4,
        minPoseDetectionConfidence: 0.55,
        minPosePresenceConfidence: 0.55,
        minTrackingConfidence: 0.55
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
          numPoses: 4,
          minPoseDetectionConfidence: 0.55,
          minPosePresenceConfidence: 0.55,
          minTrackingConfidence: 0.55
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
   * [핵심] 한 사람이 여러 명으로 중복 인식되는 현상(NMS) 필터링
   * 동일인의 어깨/골반 중심점이 근접한 경우 더 신뢰도 높은 1개 포즈만 유지
   */
  filterDuplicatePoses(allPoses, aspectRatio = 1.0) {
    if (!allPoses || allPoses.length <= 1) return allPoses || [];

    const filtered = [];
    const duplicateThreshold = 0.18; // 동일 인물 중심점 판별 거리

    for (const pose of allPoses) {
      const ls = pose[11], rs = pose[12], lh = pose[23], rh = pose[24];
      if (!ls || !rs) continue;

      const neckX = (ls.x + rs.x) / 2;
      const neckY = (ls.y + rs.y) / 2;
      const hipX = (lh && rh) ? (lh.x + rh.x) / 2 : neckX;
      const hipY = (lh && rh) ? (lh.y + rh.y) / 2 : neckY + 0.3;
      const cx = (neckX + hipX) / 2;
      const cy = (neckY + hipY) / 2;

      // 이미 채택된 포즈들과 비교
      let isDuplicate = false;
      for (const accepted of filtered) {
        const a_ls = accepted[11], a_rs = accepted[12], a_lh = accepted[23], a_rh = accepted[24];
        const a_neckX = (a_ls.x + a_rs.x) / 2;
        const a_neckY = (a_ls.y + a_rs.y) / 2;
        const a_hipX = (a_lh && a_rh) ? (a_lh.x + a_rh.x) / 2 : a_neckX;
        const a_hipY = (a_lh && a_rh) ? (a_lh.y + a_rh.y) / 2 : a_neckY + 0.3;
        const a_cx = (a_neckX + a_hipX) / 2;
        const a_cy = (a_neckY + a_hipY) / 2;

        const dist = Math.hypot((cx - a_cx) * aspectRatio, cy - a_cy);
        const neckDist = Math.hypot((neckX - a_neckX) * aspectRatio, neckY - a_neckY);

        if (dist < duplicateThreshold || neckDist < 0.14) {
          isDuplicate = true;
          break;
        }
      }

      if (!isDuplicate) {
        filtered.push(pose);
      }
    }

    return filtered;
  }

  /**
   * 단일 비디오 프레임 추론 및 3초 롤링 윈도우 종합 평가
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

    // 가로/세로 비율 계산
    const aspectRatio = (videoElement.videoWidth || 640) / (videoElement.videoHeight || 480);

    // 1. 중복 인식 포즈 제거 (1인 다중 인식 차단)
    const rawPoses = results?.landmarks || [];
    const distinctPoses = this.filterDuplicatePoses(rawPoses, aspectRatio);
    this.metrics.person_count = distinctPoses.length;

    // 2. 인원별 독립 트래킹 및 순간 평가
    const instantEval = this.updateMultiPersonTracking(distinctPoses, safeTimestamp, aspectRatio);

    // 3. [핵심] 최근 3초 롤링 윈도우 기록 및 평균 계산
    this.update3SecondRollingWindow(safeTimestamp, instantEval);

    return {
      allLandmarks: distinctPoses,
      landmarks: distinctPoses[0] || null,
      personCount: distinctPoses.length,
      dancingCount: instantEval.dancingCount,
      checklist: { ...this.checklist },
      windowMetrics: { ...this.windowMetrics },
      metrics: {
        ...this.metrics,
        dance_score: this.windowMetrics.avg_score, // 3초 평균 점수 반영
        fps: this.currentFps
      },
      isDancing: this.windowMetrics.is_dancing
    };
  }

  /**
   * [핵심] 최근 3초 롤링 윈도우 평균 계산 및 화면 판정 결정
   */
  update3SecondRollingWindow(timestamp, instantEval) {
    // 현재 프레임 상태 버퍼에 추가
    this.rollingWindow.push({
      timestamp,
      body: this.checklist.is_body_detected,
      moving: this.checklist.is_moving,
      rhythm: this.checklist.is_rhythmic,
      dynamic: this.checklist.is_dynamic,
      sustained: this.checklist.is_sustained,
      score: this.metrics.dance_score,
      energy: this.metrics.energy_level,
      isDancing: instantEval.isDancing
    });

    // 3초(3000ms) 이전 과거 데이터 제거
    const cutoff = timestamp - this.windowDurationMs;
    while (this.rollingWindow.length > 0 && this.rollingWindow[0].timestamp < cutoff) {
      this.rollingWindow.shift();
    }

    const total = this.rollingWindow.length;
    if (total === 0) {
      this.windowMetrics = {
        is_dancing: false,
        verdict_text: "인물 감지 대기 중",
        avg_score: 0,
        avg_energy: 0,
        active_count: 0,
        rates: { body: 0, moving: 0, rhythm: 0, dynamic: 0, sustained: 0 }
      };
      return;
    }

    // 3초간 각 항목별 충족 비율(%) 계산
    const bodyCount = this.rollingWindow.filter(w => w.body).length;
    const movingCount = this.rollingWindow.filter(w => w.moving).length;
    const rhythmCount = this.rollingWindow.filter(w => w.rhythm).length;
    const dynamicCount = this.rollingWindow.filter(w => w.dynamic).length;
    const sustainedCount = this.rollingWindow.filter(w => w.sustained).length;

    const rates = {
      body: Math.round((bodyCount / total) * 100),
      moving: Math.round((movingCount / total) * 100),
      rhythm: Math.round((rhythmCount / total) * 100),
      dynamic: Math.round((dynamicCount / total) * 100),
      sustained: Math.round((sustainedCount / total) * 100)
    };

    // 3초 평균 점수 및 평균 에너지
    const avgScore = Math.round(
      this.rollingWindow.reduce((acc, w) => acc + w.score, 0) / total
    );
    const avgEnergy = parseFloat(
      (this.rollingWindow.reduce((acc, w) => acc + w.energy, 0) / total).toFixed(2)
    );

    // 3초 평균 기준 충족 항목 수 (충족률 45~50% 이상인 항목 수)
    let activeCount = 0;
    if (rates.body >= 50) activeCount++;
    if (rates.moving >= 50) activeCount++;
    if (rates.rhythm >= 40) activeCount++;
    if (rates.dynamic >= 40) activeCount++;
    if (rates.sustained >= 45) activeCount++;

    // 3초 평균 기준 종합 춤 여부 판정
    // [핵심] 가만히 서 있는 사람의 오인식 차단:
    // 1. 유효 움직임 비율 50% 이상 유지
    // 2. 3초 평균 에너지 0.65 이상
    // 3. 3초 평균 댄스 점수 52 이상
    // 4. 리듬/동적 가동/지속성 중 명확한 댄스 시그널 동반
    const hasDanceSignal = rates.rhythm >= 35 || rates.dynamic >= 35 || rates.sustained >= 40;
    const isDancingWindow = rates.body >= 45 &&
      rates.moving >= 50 &&
      avgEnergy >= 0.65 &&
      avgScore >= 52 &&
      hasDanceSignal;

    let verdictText = "🧍 정지 상태 (Idle)";
    if (rates.body < 40) {
      verdictText = "전신 인식 필요 (뒤로 물러서주세요)";
    } else if (isDancingWindow) {
      if (this.metrics.dancing_person_count >= 2) {
        verdictText = `👥 ${this.metrics.dancing_person_count}명 동시 댄스 중! 🔥`;
      } else {
        verdictText = `🔥 춤 감지 중! (Dancing)`;
      }
    } else if (rates.moving >= 30 || avgEnergy >= 0.40) {
      verdictText = "🚶 단순 행동/움직임 중";
    }

    this.windowMetrics = {
      is_dancing: isDancingWindow,
      verdict_text: verdictText,
      avg_score: avgScore,
      avg_energy: avgEnergy,
      active_count: activeCount,
      rates
    };
  }

  /**
   * 다인원 트래킹 매칭 및 프레임 순간 평가
   */
  updateMultiPersonTracking(distinctPoses, timestamp, aspectRatio) {
    if (!distinctPoses || distinctPoses.length === 0) {
      this.resetEvaluationState();
      return { isDancing: false, dancingCount: 0 };
    }

    const currentPosesData = distinctPoses.map((landmarks, index) => {
      const head = this.getHeadPoint(landmarks);
      const ls = landmarks[11], rs = landmarks[12];
      const lh = landmarks[23], rh = landmarks[24];
      const cx = ((ls?.x || 0.5) + (rs?.x || 0.5) + (lh?.x || 0.5) + (rh?.x || 0.5)) / 4;
      const cy = ((ls?.y || 0.5) + (rs?.y || 0.5) + (lh?.y || 0.5) + (rh?.y || 0.5)) / 4;
      return { index, landmarks, head, cx, cy, matchedTrack: null };
    });

    const activeTracks = Array.from(this.tracks.values());
    const matchedTrackIds = new Set();

    for (const poseData of currentPosesData) {
      let closestTrack = null;
      let minDistance = 0.35;

      for (const track of activeTracks) {
        if (matchedTrackIds.has(track.id)) continue;
        if (!track.prevJoints) continue;

        const dist = Math.hypot((poseData.cx - track.prevJoints.hipX) * aspectRatio, poseData.cy - track.prevJoints.hipY);
        if (dist < minDistance) {
          minDistance = dist;
          closestTrack = track;
        }
      }

      if (closestTrack) {
        matchedTrackIds.add(closestTrack.id);
        closestTrack.update(poseData.landmarks, timestamp, poseData.head, aspectRatio);
        poseData.matchedTrack = closestTrack;
      } else {
        const newTrack = new PersonTrack(this.nextTrackId++);
        newTrack.update(poseData.landmarks, timestamp, poseData.head, aspectRatio);
        this.tracks.set(newTrack.id, newTrack);
        matchedTrackIds.add(newTrack.id);
        poseData.matchedTrack = newTrack;
      }
    }

    const now = performance.now();
    for (const [id, track] of this.tracks.entries()) {
      if (!matchedTrackIds.has(id)) {
        track.lostFrames++;
        if (now - track.lastSeen > 1200 || track.lostFrames > 35) {
          this.tracks.delete(id);
        }
      }
    }

    const validTracks = currentPosesData
      .map(p => p.matchedTrack)
      .filter(t => t && t.isBodyDetected);

    if (validTracks.length === 0) {
      this.resetEvaluationState();
      return { isDancing: false, dancingCount: 0 };
    }

    validTracks.sort((a, b) => b.danceScore - a.danceScore);
    const primary = validTracks[0];

    const dancingTracks = validTracks.filter(t => t.isDancing);
    const dancingCount = dancingTracks.length;
    this.metrics.dancing_person_count = dancingCount;

    let combinedScore = primary.danceScore;
    if (dancingCount >= 2) {
      const synergyBonus = Math.min(15, (dancingCount - 1) * 6);
      combinedScore = Math.min(100, combinedScore + synergyBonus);
    }

    // 순간 체크리스트
    this.checklist.is_body_detected = true;
    this.checklist.is_moving = validTracks.some(t => t.isMoving);
    this.checklist.is_rhythmic = validTracks.some(t => t.isRhythmic);
    this.checklist.is_dynamic = validTracks.some(t => t.isDynamic);
    this.checklist.is_sustained = validTracks.some(t => t.isSustained);

    this.metrics.energy_level = parseFloat(primary.smoothEnergy.toFixed(2));
    this.metrics.bpm_estimate = primary.bpm || (dancingTracks.find(t => t.bpm)?.bpm) || null;
    this.metrics.periodicity_score = this.checklist.is_rhythmic ? 0.85 : 0.2;
    this.metrics.dance_score = combinedScore;

    const isDancing = dancingCount > 0 || (combinedScore >= 50 && this.checklist.is_moving);
    return { isDancing, dancingCount };
  }

  /**
   * 얼굴 세부 포인트(0~10) 제거 및 단일 머리 중심점 계산
   */
  getHeadPoint(landmarks) {
    if (!landmarks) return null;

    const nose = landmarks[0];
    const earL = landmarks[7], earR = landmarks[8];
    const ls = landmarks[11], rs = landmarks[12];

    const neckX = (ls && rs) ? (ls.x + rs.x) / 2 : (nose?.x || 0.5);
    const neckY = (ls && rs) ? (ls.y + rs.y) / 2 : 0.3;

    let headX = neckX;
    let headY = neckY - 0.12;
    let visibility = 0.5;

    if (earL && earR && (earL.visibility || 0) > 0.3 && (earR.visibility || 0) > 0.3) {
      headX = (earL.x + earR.x) / 2;
      headY = (earL.y + earR.y) / 2;
      visibility = (earL.visibility + earR.visibility) / 2;
    } else if (nose && (nose.visibility || 0) > 0.3) {
      headX = nose.x;
      const headOffset = Math.max(0.04, Math.abs(neckY - nose.y) * 0.4);
      headY = nose.y - headOffset;
      visibility = nose.visibility || 0.5;
    }

    return { x: headX, y: headY, visibility, neckX, neckY };
  }

  /**
   * 스켈레톤 초고속 렌더링 (가로/세로 비율 자동 정합)
   * clear가 true일 때만 clearRect 실행 (비디오 합성 녹화 시 비디오 유지)
   */
  drawLandmarks(canvasCtx, allLandmarks, width, height, clear = true) {
    if (!canvasCtx) return;

    canvasCtx.save();
    if (clear) {
      canvasCtx.clearRect(0, 0, width, height);
    }

    if (!allLandmarks || allLandmarks.length === 0) {
      canvasCtx.restore();
      return;
    }

    const bodyConnections = [
      [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
      [11, 23], [12, 24], [23, 24],
      [23, 25], [25, 27], [24, 26], [26, 28]
    ];

    allLandmarks.forEach((landmarks, pIdx) => {
      const palette = PERSON_PALETTES[pIdx % PERSON_PALETTES.length];
      const head = this.getHeadPoint(landmarks);

      // 1. 머리 -> 목 척추 라인
      if (head && head.visibility > 0.35) {
        const hx = head.x * width, hy = head.y * height;
        const nx = head.neckX * width, ny = head.neckY * height;

        canvasCtx.lineWidth = 5;
        canvasCtx.strokeStyle = palette.glowLine;
        canvasCtx.beginPath();
        canvasCtx.moveTo(hx, hy);
        canvasCtx.lineTo(nx, ny);
        canvasCtx.stroke();

        canvasCtx.lineWidth = 2.5;
        canvasCtx.strokeStyle = palette.stroke;
        canvasCtx.beginPath();
        canvasCtx.moveTo(hx, hy);
        canvasCtx.lineTo(nx, ny);
        canvasCtx.stroke();
      }

      // 2. 스켈레톤 뼈대
      canvasCtx.lineWidth = 5;
      canvasCtx.strokeStyle = palette.glowLine;
      canvasCtx.beginPath();
      for (const [startIdx, endIdx] of bodyConnections) {
        const p1 = landmarks[startIdx], p2 = landmarks[endIdx];
        if (p1 && p2 && (p1.visibility || 0) > 0.35 && (p2.visibility || 0) > 0.35) {
          canvasCtx.moveTo(p1.x * width, p1.y * height);
          canvasCtx.lineTo(p2.x * width, p2.y * height);
        }
      }
      canvasCtx.stroke();

      canvasCtx.lineWidth = 2.5;
      canvasCtx.strokeStyle = palette.stroke;
      canvasCtx.beginPath();
      for (const [startIdx, endIdx] of bodyConnections) {
        const p1 = landmarks[startIdx], p2 = landmarks[endIdx];
        if (p1 && p2 && (p1.visibility || 0) > 0.35 && (p2.visibility || 0) > 0.35) {
          canvasCtx.moveTo(p1.x * width, p1.y * height);
          canvasCtx.lineTo(p2.x * width, p2.y * height);
        }
      }
      canvasCtx.stroke();

      // 3. 주요 관절
      for (let i = 11; i < landmarks.length; i++) {
        if (![11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].includes(i)) continue;

        const lm = landmarks[i];
        if ((lm.visibility || 0) > 0.35) {
          const x = lm.x * width, y = lm.y * height;
          canvasCtx.beginPath();
          if ([15, 16, 27, 28].includes(i)) {
            canvasCtx.fillStyle = palette.joint;
            canvasCtx.arc(x, y, 5, 0, 2 * Math.PI);
          } else if ([11, 12, 23, 24].includes(i)) {
            canvasCtx.fillStyle = palette.core;
            canvasCtx.arc(x, y, 4.5, 0, 2 * Math.PI);
          } else {
            canvasCtx.fillStyle = palette.head;
            canvasCtx.arc(x, y, 3.5, 0, 2 * Math.PI);
          }
          canvasCtx.fill();
        }
      }

      // 4. 머리 단일 노드 렌더링
      if (head && head.visibility > 0.35) {
        const hx = head.x * width, hy = head.y * height;

        canvasCtx.beginPath();
        canvasCtx.fillStyle = palette.glowLine;
        canvasCtx.arc(hx, hy, 12, 0, 2 * Math.PI);
        canvasCtx.fill();

        canvasCtx.beginPath();
        canvasCtx.fillStyle = palette.head;
        canvasCtx.arc(hx, hy, 8, 0, 2 * Math.PI);
        canvasCtx.fill();

        canvasCtx.beginPath();
        canvasCtx.fillStyle = "#ffffff";
        canvasCtx.arc(hx, hy, 3, 0, 2 * Math.PI);
        canvasCtx.fill();

        if (allLandmarks.length > 1) {
          canvasCtx.font = "bold 11px Pretendard, sans-serif";
          canvasCtx.fillStyle = "#ffffff";
          canvasCtx.textAlign = "center";
          canvasCtx.fillText(`P${pIdx + 1}`, hx, hy - 15);
        }
      }
    });

    canvasCtx.restore();
  }

  resetEvaluationState() {
    this.checklist.is_body_detected = false;
    this.checklist.is_moving = false;
    this.checklist.is_rhythmic = false;
    this.checklist.is_dynamic = false;
    this.checklist.is_sustained = false;
    this.metrics.dance_score = 0;
    this.metrics.energy_level = 0;
    this.metrics.bpm_estimate = null;
    this.metrics.periodicity_score = 0;
    this.metrics.dancing_person_count = 0;
  }

  reset() {
    this.tracks.clear();
    this.rollingWindow = [];
    this.nextTrackId = 1;
    this.lastTimestamp = -1;
    this.resetEvaluationState();
    this.windowMetrics = {
      is_dancing: false,
      verdict_text: "인물 감지 대기 중",
      avg_score: 0,
      avg_energy: 0,
      active_count: 0,
      rates: { body: 0, moving: 0, rhythm: 0, dynamic: 0, sustained: 0 }
    };
  }
}
