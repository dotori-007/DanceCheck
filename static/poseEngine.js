/**
 * DanceCheck - poseEngine.js
 * MediaPipe Pose Landmarker Lite 기반 실시간 브라우저 AI 및 객관적 춤 판정 엔진
 * 100% 클라이언트 연산 (모바일 브라우저 최적화)
 * 
 * 주요 최적화:
 * 1. [FPS 극대화] CPU/GPU를 잠식하는 canvas shadowBlur 완전 제거 및 경량 듀얼 스트로크 네온 렌더링
 * 2. [다인원 독립 추적] Centroid 기반 PersonTracker로 인원별 독립 시계열(History & Energy) 버퍼 분리
 * 3. [시계열 슬라이딩 윈도우 & 댄스 래치(Latch)] 한 순간의 속도 0(정지/비트 멈춤)에도 춤 판정이 꺼지지 않도록 1.5초 유예 버퍼(Decay/Hold) 적용
 * 4. [그룹 시너지 보너스] 옆 사람이 가만히 있어도 춤추는 사람의 점수가 깎이지 않고, 함께 출 때 시너지 가산점 부여
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
 * 인원별 독립 움직임 및 댄스 상태 추적 객체
 */
class PersonTrack {
  constructor(id) {
    this.id = id;
    this.lastSeen = performance.now();
    this.lostFrames = 0;

    // 시계열 링 버퍼 (약 60프레임 = 2초)
    this.history = []; // { timestamp, hipY, lwY, rwY, energy }
    this.maxHistory = 60;

    // 관절 이전 위치
    this.prevJoints = null;
    this.torsoHeight = 0.2;

    // 에너지 및 지속 시간
    this.instantEnergy = 0;
    this.smoothEnergy = 0;
    this.danceHoldFrames = 0; // 춤 상태 유지 카운터 (비트 일시정지에도 유지)
    this.sustainedDanceFrames = 0; // 누적 댄스 지속 프레임

    // 평가 결과
    this.isBodyDetected = false;
    this.isMoving = false;
    this.isRhythmic = false;
    this.isDynamic = false;
    this.isSustained = false;
    this.isDancing = false;
    this.danceScore = 0;
    this.bpm = null;
  }

  update(landmarks, timestamp, head) {
    this.lastSeen = performance.now();
    this.lostFrames = 0;

    const ls = landmarks[11], rs = landmarks[12];
    const lh = landmarks[23], rh = landmarks[24];
    const lw = landmarks[15], rw = landmarks[16];
    const le = landmarks[13], re = landmarks[14];
    const la = landmarks[27], ra = landmarks[28];

    const shoulderVis = ((ls?.visibility || 0) + (rs?.visibility || 0)) / 2;
    const hipVis = ((lh?.visibility || 0) + (rh?.visibility || 0)) / 2;

    const shoulderCenterY = (ls && rs) ? (ls.y + rs.y) / 2 : 0;
    const hipCenterY = (lh && rh) ? (lh.y + rh.y) / 2 : 0;
    const torsoH = hipCenterY - shoulderCenterY;
    const shoulderWidth = (ls && rs) ? Math.abs(ls.x - rs.x) : 0;

    // 얼굴/상체 클로즈업 필터링 (골반과 어깨가 모두 유효해야 함)
    const isCloseUp = hipVis < 0.4 ||
                      hipCenterY > 0.98 ||
                      torsoH < 0.10 ||
                      shoulderWidth > 0.72;

    this.isBodyDetected = shoulderVis >= 0.50 && hipVis >= 0.40 && !isCloseUp;

    if (!this.isBodyDetected) {
      this.instantEnergy = 0;
      this.smoothEnergy *= 0.85;
      this.danceHoldFrames = Math.max(0, this.danceHoldFrames - 2);
      this.danceScore = Math.max(0, this.danceScore - 5);
      this.isDancing = false;
      this.isMoving = false;
      return;
    }

    this.torsoHeight = Math.max(0.12, torsoH);
    const scale = this.torsoHeight;

    // 사지 및 코어 2D 변위 기반 순수 운동 에너지 계산
    let currentEnergy = 0;
    if (this.prevJoints) {
      const dLw = Math.hypot((lw?.x || 0) - this.prevJoints.lwX, (lw?.y || 0) - this.prevJoints.lwY) / scale;
      const dRw = Math.hypot((rw?.x || 0) - this.prevJoints.rwX, (rw?.y || 0) - this.prevJoints.rwY) / scale;
      const dLe = Math.hypot((le?.x || 0) - this.prevJoints.leX, (le?.y || 0) - this.prevJoints.leY) / scale;
      const dRe = Math.hypot((re?.x || 0) - this.prevJoints.reX, (re?.y || 0) - this.prevJoints.reY) / scale;
      const dHip = Math.hypot(((lh?.x + rh?.x)/2 || 0) - this.prevJoints.hipX, hipCenterY - this.prevJoints.hipY) / scale;
      const dHead = head ? Math.hypot(head.x - this.prevJoints.headX, head.y - this.prevJoints.headY) / scale : 0;

      // 손목, 팔꿈치, 골반, 머리 가중합 (급격한 클로즈업 떨림 방지)
      currentEnergy = (dLw * 1.6 + dRw * 1.6 + dLe * 1.0 + dRe * 1.0 + dHip * 2.2 + dHead * 0.8) * 5.0;
    }

    this.instantEnergy = parseFloat(currentEnergy.toFixed(2));

    // 이전 좌표 업데이트
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

    // 시계열 버퍼 기록
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

    // 1. [핵심] 시계열 에너지 지수 평활 (Leaky Integrator & Peak Decay)
    // 춤동작 중 순간 정지(Freeze, Hit)나 방향 전환 시 에너지가 0으로 급락하는 것을 방지
    if (this.instantEnergy > this.smoothEnergy) {
      // 상승 시 빠른 반응 (85% 즉각 반영)
      this.smoothEnergy = this.smoothEnergy * 0.15 + this.instantEnergy * 0.85;
    } else {
      // 하강 시 부드러운 감쇄 (초당 약 40% 감쇄 -> 1초간 여운 유지)
      this.smoothEnergy = this.smoothEnergy * 0.94 + this.instantEnergy * 0.06;
    }

    // 2. 가동 범위 (Dynamic Range): 손목의 이동 궤적 및 팔 벌림 폭
    let armSpan = 0;
    if (lw && rw) {
      armSpan = Math.hypot((lw.x - rw.x) / scale, (lw.y - rw.y) / scale);
    }
    // 손이 어깨보다 높이 올라가거나 넓게 움직이는지
    const handsHigh = (lw && ls && lw.y < ls.y) || (rw && rs && rw.y < rs.y);
    this.isDynamic = armSpan >= 1.0 || handsHigh || this.instantEnergy >= 1.2;

    // 3. 움직임 유효 판정 (단순 정지가 아님)
    this.isMoving = this.smoothEnergy >= 0.28 || this.instantEnergy >= 0.35;

    // 4. 리듬 분석 (골반 바운스 & 손목 비트)
    const rhythm = this.evaluateRhythm();
    this.isRhythmic = rhythm.isRhythmic;
    this.bpm = rhythm.bpm;

    // 5. 춤 상태 유지 래치(Hold Latch)
    // 활발한 춤 동작(에너지 0.5 이상 또는 리듬/동적 충족)이 한 번 감지되면
    // 최소 45프레임 (약 1.5초) 동안은 동작이 잠깐 멈춰도 춤추는 중으로 유지
    if (this.isMoving && (this.instantEnergy >= 0.6 || this.smoothEnergy >= 0.55 || this.isRhythmic)) {
      this.danceHoldFrames = 45; // 약 1.5초 보장
      this.sustainedDanceFrames = Math.min(120, this.sustainedDanceFrames + 1.5);
    } else if (this.danceHoldFrames > 0) {
      this.danceHoldFrames--;
      this.sustainedDanceFrames = Math.max(0, this.sustainedDanceFrames - 0.5);
    } else {
      this.sustainedDanceFrames = Math.max(0, this.sustainedDanceFrames - 1.5);
    }

    this.isSustained = this.sustainedDanceFrames >= 30;

    // 6. 점수 산출 (안정적인 시계열 기반)
    let score = 15;
    // 에너지 기여 (최대 45점)
    score += Math.min(45, (this.smoothEnergy / 1.6) * 45);
    // 동적 가동 범위 (최대 20점)
    if (this.isDynamic) score += 20;
    // 리듬 및 주기성 (최대 20점)
    if (this.isRhythmic) score += 20;
    // 지속성 보너스 (최대 15점)
    if (this.isSustained) score += 15;

    // 움직임이 완전 정지된 경우 서서히 감쇄
    if (!this.isMoving && this.danceHoldFrames === 0) {
      score = Math.min(20, score * 0.4);
    }

    this.danceScore = Math.min(100, Math.round(score));

    // 최종 개인 댄스 판정
    this.isDancing = this.isBodyDetected && (
      (this.danceHoldFrames > 0 && this.danceScore >= 45) ||
      (this.danceScore >= 55 && (this.isMoving || this.isRhythmic || this.isDynamic))
    );
  }

  evaluateRhythm() {
    if (this.history.length < 25) {
      return { isRhythmic: false, bpm: null };
    }

    const scale = this.torsoHeight || 0.2;
    // 골반 또는 손목의 주기적 변위 신호
    const signal = this.history.map(h => (h.hipY + (h.lwY + h.rwY) * 0.3) / scale);
    const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
    const centered = signal.map(v => v - mean);

    const peaks = [];
    const minThreshold = 0.025;

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

    // 30fps 기준 interval 8~32 = 56~225 BPM
    const isIntervalValid = avgInterval >= 7 && avgInterval <= 34;
    const regularity = Math.max(0, 1 - (stdDev / (avgInterval || 1)));

    let bpm = null;
    if (avgInterval > 0) {
      bpm = Math.round((30 / avgInterval) * 60);
      if (bpm < 55 || bpm > 230) bpm = null;
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

    // MediaPipe timestamp 관리 (단조 증가 필수 보장)
    this.lastTimestamp = -1;

    // 다인원 트래킹 맵 (trackId -> PersonTrack)
    this.tracks = new Map();
    this.nextTrackId = 1;

    // 통합 체크리스트 상태
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

    // 전체 댄스 상태 유지 래치
    this.globalDanceHoldTimer = 0;
  }

  /**
   * MediaPipe Vision 태스크 리졸버 및 PoseLandmarker 초기화
   * 다인원 감지 (numPoses: 4) 및 안정적인 추적 컨피던스 설정
   */
  async initialize() {
    const modelAssetPath = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );

      // 다인원(numPoses: 4) 감지 및 최적화된 Tracking Confidence
      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 4,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
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
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
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
    this.metrics.person_count = allPoses.length;

    // 다인원 트래킹 매칭 및 시계열 댄스 분석
    const evaluation = this.updateMultiPersonTracking(allPoses, safeTimestamp);

    return {
      allLandmarks: allPoses,
      landmarks: allPoses[0] || null,
      personCount: allPoses.length,
      dancingCount: evaluation.dancingCount,
      checklist: { ...this.checklist },
      metrics: {
        ...this.metrics,
        fps: this.currentFps
      },
      isDancing: evaluation.isDancing
    };
  }

  /**
   * 인원별 Centroid 기반 트래킹 매칭 및 통합 평가
   */
  updateMultiPersonTracking(allPoses, timestamp) {
    if (!allPoses || allPoses.length === 0) {
      this.resetEvaluationState();
      return { isDancing: false, dancingCount: 0 };
    }

    // 현재 프레임 각 포즈의 중심점(어깨+골반) 계산
    const currentPosesData = allPoses.map((landmarks, index) => {
      const head = this.getHeadPoint(landmarks);
      const ls = landmarks[11], rs = landmarks[12];
      const lh = landmarks[23], rh = landmarks[24];
      const cx = ((ls?.x || 0.5) + (rs?.x || 0.5) + (lh?.x || 0.5) + (rh?.x || 0.5)) / 4;
      const cy = ((ls?.y || 0.5) + (rs?.y || 0.5) + (lh?.y || 0.5) + (rh?.y || 0.5)) / 4;
      return { index, landmarks, head, cx, cy, matchedTrack: null };
    });

    // 기존 트랙들과 최소 거리 매칭 (Centroid 거리 < 0.35)
    const activeTracks = Array.from(this.tracks.values());
    const matchedTrackIds = new Set();

    for (const poseData of currentPosesData) {
      let closestTrack = null;
      let minDistance = 0.35; // 허용 반경

      for (const track of activeTracks) {
        if (matchedTrackIds.has(track.id)) continue;
        if (!track.prevJoints) continue;

        const dist = Math.hypot(poseData.cx - track.prevJoints.hipX, poseData.cy - track.prevJoints.hipY);
        if (dist < minDistance) {
          minDistance = dist;
          closestTrack = track;
        }
      }

      if (closestTrack) {
        matchedTrackIds.add(closestTrack.id);
        closestTrack.update(poseData.landmarks, timestamp, poseData.head);
        poseData.matchedTrack = closestTrack;
      } else {
        // 새 트랙 생성
        const newTrack = new PersonTrack(this.nextTrackId++);
        newTrack.update(poseData.landmarks, timestamp, poseData.head);
        this.tracks.set(newTrack.id, newTrack);
        matchedTrackIds.add(newTrack.id);
        poseData.matchedTrack = newTrack;
      }
    }

    // 매칭되지 않은 오래된 트랙 정리 (1초 이상 미검출 시 삭제)
    const now = performance.now();
    for (const [id, track] of this.tracks.entries()) {
      if (!matchedTrackIds.has(id)) {
        track.lostFrames++;
        if (now - track.lastSeen > 1200 || track.lostFrames > 35) {
          this.tracks.delete(id);
        }
      }
    }

    // 유효한 신체(전신/코어)를 가진 활성 트랙 추출
    const validTracks = currentPosesData
      .map(p => p.matchedTrack)
      .filter(t => t && t.isBodyDetected);

    if (validTracks.length === 0) {
      this.resetEvaluationState();
      return { isDancing: false, dancingCount: 0 };
    }

    // 가장 활발히 춤추는 메인 댄서 선정
    validTracks.sort((a, b) => b.danceScore - a.danceScore);
    const primary = validTracks[0];

    // 춤추는 중인 인원 수 카운트
    const dancingTracks = validTracks.filter(t => t.isDancing);
    const dancingCount = dancingTracks.length;
    this.metrics.dancing_person_count = dancingCount;

    // 그룹 시너지 보너스: 2명 이상이 함께 춤출 때 가산점 부여 (옆 사람이 가만히 있어도 감점 안 됨!)
    let combinedScore = primary.danceScore;
    if (dancingCount >= 2) {
      const synergyBonus = Math.min(15, (dancingCount - 1) * 6);
      combinedScore = Math.min(100, combinedScore + synergyBonus);
    }

    // 통합 체크리스트 반영
    this.checklist.is_body_detected = true;
    this.checklist.is_moving = validTracks.some(t => t.isMoving);
    this.checklist.is_rhythmic = validTracks.some(t => t.isRhythmic);
    this.checklist.is_dynamic = validTracks.some(t => t.isDynamic);
    this.checklist.is_sustained = validTracks.some(t => t.isSustained);

    // 메트릭 반영
    this.metrics.energy_level = parseFloat(primary.smoothEnergy.toFixed(2));
    this.metrics.bpm_estimate = primary.bpm || (dancingTracks.find(t => t.bpm)?.bpm) || null;
    this.metrics.periodicity_score = this.checklist.is_rhythmic ? 0.85 : 0.2;

    // 점수 평활
    this.metrics.dance_score = Math.round(
      this.metrics.dance_score * 0.4 + combinedScore * 0.6
    );

    // 전체 춤 판정: 최소 1명 이상이 춤을 추고 있거나, 유예 래치가 살아있을 때
    let isOverallDancing = dancingCount > 0 || (this.metrics.dance_score >= 50 && this.checklist.is_moving);

    if (isOverallDancing) {
      this.globalDanceHoldTimer = 35; // 약 1.2초간 전체 춤 상태 유지
    } else if (this.globalDanceHoldTimer > 0) {
      this.globalDanceHoldTimer--;
      isOverallDancing = true;
    }

    return {
      isDancing: Boolean(isOverallDancing),
      dancingCount
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
   * 다인원 랜드마크 초고속 렌더링 (shadowBlur 완전 제거로 모바일 30+ FPS 보장)
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
      [11, 12], // 어깨 사이
      [11, 13], [13, 15], // 왼팔
      [12, 14], [14, 16], // 오른팔
      [11, 23], [12, 24], [23, 24], // 몸통
      [23, 25], [25, 27], // 왼다리
      [24, 26], [26, 28]  // 오른다리
    ];

    allLandmarks.forEach((landmarks, pIdx) => {
      const palette = PERSON_PALETTES[pIdx % PERSON_PALETTES.length];
      const head = this.getHeadPoint(landmarks);

      // 1. 머리 -> 목 척추 연결선 (2패스 경량 네온)
      if (head && head.visibility > 0.35) {
        const hx = head.x * width;
        const hy = head.y * height;
        const nx = head.neckX * width;
        const ny = head.neckY * height;

        // 글로우 효과를 shadowBlur 대신 반투명 두꺼운 선으로 대체 (CPU/GPU 부하 0)
        canvasCtx.lineWidth = 6;
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

      // 2. 스켈레톤 선 그리기 (2패스 경량 네온 렌더링)
      // 패스 1: 반투명 글로우 외곽선
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

      // 패스 2: 선명한 중심 네온 라인
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

      // 3. 주요 관절 포인트 렌더링 (단일 패스 솔리드 원)
      for (let i = 11; i < landmarks.length; i++) {
        if (![11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].includes(i)) continue;

        const lm = landmarks[i];
        if ((lm.visibility || 0) > 0.35) {
          const x = lm.x * width;
          const y = lm.y * height;

          canvasCtx.beginPath();
          if ([15, 16, 27, 28].includes(i)) {
            // 손목/발목 포인트
            canvasCtx.fillStyle = palette.joint;
            canvasCtx.arc(x, y, 5, 0, 2 * Math.PI);
          } else if ([11, 12, 23, 24].includes(i)) {
            // 코어 어깨/골반 포인트
            canvasCtx.fillStyle = palette.core;
            canvasCtx.arc(x, y, 4.5, 0, 2 * Math.PI);
          } else {
            // 팔꿈치/무릎
            canvasCtx.fillStyle = palette.head;
            canvasCtx.arc(x, y, 3.5, 0, 2 * Math.PI);
          }
          canvasCtx.fill();
        }
      }

      // 4. 단일 머리(Head) 포인트 원형 렌더링 (얼굴 디테일 대체)
      if (head && head.visibility > 0.35) {
        const hx = head.x * width;
        const hy = head.y * height;

        // 머리 외곽 링
        canvasCtx.beginPath();
        canvasCtx.fillStyle = palette.glowLine;
        canvasCtx.arc(hx, hy, 12, 0, 2 * Math.PI);
        canvasCtx.fill();

        // 머리 중심 원
        canvasCtx.beginPath();
        canvasCtx.fillStyle = palette.head;
        canvasCtx.arc(hx, hy, 8, 0, 2 * Math.PI);
        canvasCtx.fill();

        // 머리 내부 화이트 코어 닷
        canvasCtx.beginPath();
        canvasCtx.fillStyle = "#ffffff";
        canvasCtx.arc(hx, hy, 3, 0, 2 * Math.PI);
        canvasCtx.fill();

        // 다인원일 때 인원 번호 배지 표시 (P1, P2 ...)
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

  /**
   * 평가 상태 초기화
   */
  resetEvaluationState() {
    this.checklist.is_body_detected = false;
    this.checklist.is_moving = false;
    this.checklist.is_rhythmic = false;
    this.checklist.is_dynamic = false;
    this.checklist.is_sustained = false;
    this.metrics.dance_score = Math.max(0, this.metrics.dance_score - 5);
    this.metrics.energy_level = Math.max(0, this.metrics.energy_level * 0.8);
    this.metrics.bpm_estimate = null;
    this.metrics.periodicity_score = 0;
    this.metrics.dancing_person_count = 0;
  }

  reset() {
    this.tracks.clear();
    this.nextTrackId = 1;
    this.lastTimestamp = -1;
    this.globalDanceHoldTimer = 0;
    this.metrics.dance_score = 0;
    this.metrics.energy_level = 0;
    this.metrics.bpm_estimate = null;
    this.metrics.periodicity_score = 0;
    this.metrics.person_count = 0;
    this.metrics.dancing_person_count = 0;
    this.checklist = {
      is_body_detected: false,
      is_moving: false,
      is_rhythmic: false,
      is_dynamic: false,
      is_sustained: false
    };
  }
}
