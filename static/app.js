/**
 * DanceCheck - app.js
 * 카메라 제어, UI 실시간 갱신, MP4 영상 녹화 및 내보내기
 * 100% 클라이언트 연산 (모바일 가로/세로 대응)
 */

import { PoseEngine } from "./poseEngine.js";

class DanceCheckApp {
  constructor() {
    this.poseEngine = new PoseEngine();
    this.video = document.getElementById("webcam");
    this.canvas = document.getElementById("output-canvas");
    this.canvasCtx = this.canvas.getContext("2d");

    // UI 요소
    this.startOverlay = document.getElementById("start-overlay");
    this.startBtn = document.getElementById("start-btn");
    this.modelStatusText = document.getElementById("model-status");
    this.fpsBadge = document.getElementById("fps-badge");
    this.cameraFlipBtn = document.getElementById("camera-flip-btn");
    this.liveStatusBadge = document.getElementById("live-status-badge");
    this.liveStatusText = document.getElementById("live-status-text");
    this.danceScoreBar = document.getElementById("dance-score-bar");
    this.danceScoreText = document.getElementById("dance-score-text");

    // 체크리스트 요소
    this.chkItems = {
      body: {
        el: document.getElementById("chk-body"),
        state: document.getElementById("chk-body-state")
      },
      moving: {
        el: document.getElementById("chk-moving"),
        state: document.getElementById("chk-moving-state")
      },
      rhythm: {
        el: document.getElementById("chk-rhythm"),
        state: document.getElementById("chk-rhythm-state")
      },
      dynamic: {
        el: document.getElementById("chk-dynamic"),
        state: document.getElementById("chk-dynamic-state")
      },
      sustained: {
        el: document.getElementById("chk-sustained"),
        state: document.getElementById("chk-sustained-state")
      }
    };
    this.checklistCounter = document.getElementById("checklist-counter");
    this.bpmDisplay = document.getElementById("bpm-display");

    // MP4 녹화 관련 UI
    this.recordBtn = document.getElementById("record-btn");
    this.recordBtnText = document.getElementById("record-btn-text");
    this.recIndicator = document.getElementById("rec-indicator");
    this.recTimer = document.getElementById("rec-timer");
    this.recCameraBadge = document.getElementById("rec-camera-badge");
    this.recCameraTimer = document.getElementById("rec-camera-timer");

    // MP4 내보내기/미리보기 모달
    this.exportModal = document.getElementById("export-modal");
    this.exportVideoPlayer = document.getElementById("export-video-player");
    this.expScore = document.getElementById("exp-score");
    this.expDuration = document.getElementById("exp-duration");
    this.expVerdict = document.getElementById("exp-verdict");
    this.downloadMp4Btn = document.getElementById("download-mp4-btn");
    this.exportCloseBtn = document.getElementById("export-close-btn");
    this.exportCancelBtn = document.getElementById("export-cancel-btn");

    // 토스트
    this.toastContainer = document.getElementById("toast-container");

    // 카메라 및 루프 상태
    this.currentFacingMode = "user"; // 'user' (전면) or 'environment' (후면)
    this.stream = null;
    this.isRunning = false;
    this.isFlipping = false;
    this.loopGeneration = 0;
    this.lastProcessedResult = null;

    // MP4 녹화 상태 관리
    this.isRecording = false;
    this.mediaRecorder = null;
    this.recordedChunks = [];
    this.recordingStartTime = 0;
    this.recordingTimerInterval = null;
    this.recordingScores = [];
    this.lastExportedBlob = null;
    this.lastExportedUrl = null;

    // 비디오 + 스켈레톤 + HUD 합성용 오프스크린 레코딩 캔버스
    this.recCanvas = document.createElement("canvas");
    this.recCtx = this.recCanvas.getContext("2d");

    this.init();
  }

  async init() {
    this.bindEvents();

    try {
      this.modelStatusText.innerHTML = '<span class="spinner"></span> 비전 모델 로드 중...';
      await this.poseEngine.initialize();
      this.modelStatusText.innerHTML = '<span style="color: var(--status-active); font-weight: 600;">● 온디바이스 엔진 준비 완료</span>';
      this.startBtn.disabled = false;
    } catch (err) {
      console.error(err);
      this.modelStatusText.innerHTML = '<span style="color: var(--status-rec);">● 모델 로드 실패: ' + (err.message || '네트워크 확인 필요') + '</span>';
      this.showToast("비전 모델 로드에 실패했습니다. 페이지를 새로고침해주세요.");
    }
  }

  bindEvents() {
    this.startBtn.addEventListener("click", () => this.startCamera());
    this.cameraFlipBtn.addEventListener("click", () => this.toggleCameraFlip());

    // MP4 녹화 토글
    this.recordBtn.addEventListener("click", () => this.toggleRecording());

    // 모달 액션
    this.downloadMp4Btn.addEventListener("click", () => this.downloadExportedVideo());
    this.exportCloseBtn.addEventListener("click", () => this.closeExportModal());
    this.exportCancelBtn.addEventListener("click", () => this.closeExportModal());

    // 모달 배경 클릭 시 닫기
    this.exportModal.addEventListener("click", (e) => {
      if (e.target === this.exportModal) this.closeExportModal();
    });

    // 화면 회전(가로/세로 전환) 감지 대응
    window.addEventListener("orientationchange", () => {
      setTimeout(() => {
        if (this.isRunning) this.startCamera();
      }, 350);
    });
  }

  /**
   * 카메라 권한 획득 및 스트리밍 시작 (iOS Safari / Android Chrome 호환)
   */
  async startCamera() {
    this.loopGeneration++;
    const currentGen = this.loopGeneration;

    if (this.stream) {
      try {
        this.stream.getTracks().forEach(track => track.stop());
      } catch (e) {
        console.warn("트랙 종료 경고:", e);
      }
      this.stream = null;
    }

    this.poseEngine.reset();
    this.canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    const isLandscape = window.innerWidth > window.innerHeight;
    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: this.currentFacingMode },
        width: { ideal: isLandscape ? 640 : 480, max: isLandscape ? 800 : 640 },
        height: { ideal: isLandscape ? 480 : 640, max: isLandscape ? 600 : 800 },
        frameRate: { ideal: 30, max: 30 }
      }
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = this.stream;

      await new Promise((resolve) => {
        if (this.video.readyState >= HTMLMediaElement.HAVE_METADATA) {
          resolve();
        } else {
          this.video.onloadedmetadata = () => resolve();
        }
      });

      await this.video.play();

      this.applyCameraMirroring();
      this.startOverlay.classList.remove("active");
      this.isRunning = true;

      this.requestFrameLoop(currentGen);
      this.showToast(this.currentFacingMode === "user" ? "전면 카메라 활성화" : "후면 카메라 활성화");
    } catch (err) {
      console.error("카메라 접근 오류:", err);
      let errMsg = "카메라에 접근할 수 없습니다. 권한을 확인해주세요.";
      if (err.name === "NotAllowedError") {
        errMsg = "카메라 권한이 거부되었습니다. 브라우저 설정에서 허용해주세요.";
      }
      this.showToast(errMsg);
      alert(errMsg);
    }
  }

  applyCameraMirroring() {
    if (this.currentFacingMode === "environment") {
      this.video.style.transform = "none";
      this.canvas.style.transform = "none";
    } else {
      this.video.style.transform = "scaleX(-1)";
      this.canvas.style.transform = "scaleX(-1)";
    }
  }

  async toggleCameraFlip() {
    if (this.isFlipping) return;
    if (this.isRecording) {
      this.showToast("녹화 중에는 카메라를 전환할 수 없습니다.");
      return;
    }

    this.isFlipping = true;
    this.cameraFlipBtn.disabled = true;

    try {
      this.currentFacingMode = this.currentFacingMode === "user" ? "environment" : "user";
      if (this.isRunning) {
        await this.startCamera();
      }
    } finally {
      this.isFlipping = false;
      this.cameraFlipBtn.disabled = false;
    }
  }

  /**
   * 렌더링 및 추론 루프
   */
  requestFrameLoop(generation) {
    const loop = (timestamp) => {
      if (!this.isRunning || generation !== this.loopGeneration) return;

      if (this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        this.renderFrame(timestamp);
      }

      if ("requestVideoFrameCallback" in this.video) {
        this.video.requestVideoFrameCallback((now) => loop(now));
      } else {
        requestAnimationFrame(loop);
      }
    };

    if ("requestVideoFrameCallback" in this.video) {
      this.video.requestVideoFrameCallback((now) => loop(now));
    } else {
      requestAnimationFrame(loop);
    }
  }

  renderFrame(timestamp) {
    if (this.canvas.width !== this.video.videoWidth || this.canvas.height !== this.video.videoHeight) {
      this.canvas.width = this.video.videoWidth || 480;
      this.canvas.height = this.video.videoHeight || 640;
    }

    // AI 엔진 추론 실행
    const result = this.poseEngine.processFrame(this.video, timestamp);
    if (!result) return;

    this.lastProcessedResult = result;

    // 화면 오버레이 캔버스에 랜드마크 드로잉
    this.poseEngine.drawLandmarks(
      this.canvasCtx,
      result.allLandmarks,
      this.canvas.width,
      this.canvas.height
    );

    // UI 동기화
    this.updateUI(result);

    // MP4 녹화 중인 경우 비디오 + 스켈레톤 + HUD를 합성하여 레코딩 캔버스에 렌더링
    if (this.isRecording && this.mediaRecorder?.state === "recording") {
      this.captureCompositeRecordingFrame(result);
    }
  }

  /**
   * 녹화 캔버스에 카메라 비디오 + AI 스켈레톤 + 분석 워터마크 HUD 합성
   */
  captureCompositeRecordingFrame(result) {
    const rw = this.video.videoWidth || 640;
    const rh = this.video.videoHeight || 480;

    if (this.recCanvas.width !== rw || this.recCanvas.height !== rh) {
      this.recCanvas.width = rw;
      this.recCanvas.height = rh;
    }

    const ctx = this.recCtx;

    // 1. 카메라 비디오 프레임 + 스켈레톤 함께 그리기 (전면 카메라는 둘 다 일관되게 좌우 반전)
    ctx.save();
    if (this.currentFacingMode === "user") {
      ctx.translate(rw, 0);
      ctx.scale(-1, 1);
    }
    // 카메라 원본 비디오 렌더링
    ctx.drawImage(this.video, 0, 0, rw, rh);

    // 스켈레톤 골격 오버레이 (비디오를 지우지 않도록 clear = false 전달)
    this.poseEngine.drawLandmarks(ctx, result.allLandmarks, rw, rh, false);
    ctx.restore();

    // 2. 비디오 상단에 스튜디오 텔레메트리 워터마크 렌더링
    ctx.save();
    const score = result.windowMetrics?.avg_score || result.metrics.dance_score;
    const isDancing = result.windowMetrics?.is_dancing || result.isDancing;
    const stateText = isDancing ? "DANCING" : (result.windowMetrics?.rates.moving >= 35 ? "MOVING" : "IDLE");
    const bpmText = result.metrics.bpm_estimate ? ` · ${result.metrics.bpm_estimate} BPM` : "";

    // 상단 반투명 스튜디오 배너 바
    const bannerW = Math.min(rw - 32, 300);
    ctx.fillStyle = "rgba(10, 11, 16, 0.88)";
    ctx.fillRect(16, 16, bannerW, 38);

    // 정돈된 1px 테두리
    ctx.lineWidth = 1;
    ctx.strokeStyle = isDancing ? "rgba(16, 185, 129, 0.5)" : "rgba(255, 255, 255, 0.12)";
    ctx.strokeRect(16, 16, bannerW, 38);

    // 텍스트 출력
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    ctx.fillStyle = isDancing ? "#10b981" : "#94a3b8";
    ctx.fillText("DANCECHECK · STUDIO", 26, 30);

    ctx.font = "600 12px Pretendard, -apple-system, sans-serif";
    ctx.fillStyle = "#f8fafc";
    ctx.fillText(`[${stateText}] 점수 ${score}%${bpmText}`, 26, 45);

    ctx.restore();

    // 녹화 중 점수 샘플 수집
    this.recordingScores.push(score);
  }

  updateUI(result) {
    const { metrics, landmarks, windowMetrics } = result;
    const wm = windowMetrics || {
      is_dancing: false,
      avg_score: metrics.dance_score,
      avg_energy: metrics.energy_level,
      active_count: 0,
      rates: { body: 0, moving: 0, rhythm: 0, dynamic: 0, sustained: 0 }
    };

    // 1. FPS 배지
    this.fpsBadge.textContent = `FPS: ${metrics.fps || "--"}`;

    // 2. 댄스 게이지 (3초 평균 반영)
    this.danceScoreText.textContent = `${wm.avg_score}%`;
    this.danceScoreBar.style.width = `${wm.avg_score}%`;

    // 3. 실시간 판정 상태 배지 (최근 3초 롤링 윈도우 평균 기준)
    this.liveStatusBadge.className = "live-status-badge";
    const personCount = result.personCount || 0;
    const personPrefix = personCount > 1 ? `👥 ${personCount}명 ` : "";

    if (!landmarks || wm.rates.body < 40) {
      this.liveStatusText.textContent = personCount > 0
        ? `${personPrefix}감지 (골반/전신이 보이도록 물러서주세요)`
        : "인물 감지 대기 중 (전신을 비춰주세요)";
    } else if (wm.is_dancing) {
      this.liveStatusBadge.classList.add("dancing");
      if (result.dancingCount >= 2) {
        this.liveStatusText.textContent = `👥 ${result.dancingCount}명 동시 댄스 중! 🔥 (평균 ${wm.avg_score}점)`;
      } else {
        this.liveStatusText.textContent = `${personPrefix}🔥 춤추는 중! (3초 평균: ${wm.avg_score}점)`;
      }
    } else if (wm.rates.moving >= 30 || wm.avg_energy >= 0.40) {
      this.liveStatusBadge.classList.add("moving");
      this.liveStatusText.textContent = `${personPrefix}🚶 단순 행동 중 (움직임 ${wm.rates.moving}%)`;
    } else {
      this.liveStatusText.textContent = `${personPrefix}🧍 정지/휴식 중 (Idle)`;
    }

    // 4. 체크리스트 HUD 업데이트 (최근 3초 평균 충족률 반영)
    const bodyActive = wm.rates.body >= 50;
    this.toggleCheckItem(this.chkItems.body, bodyActive, `${wm.rates.body}% (인식됨)`, `${wm.rates.body}% (미인식)`);

    const movingActive = wm.rates.moving >= 50;
    this.toggleCheckItem(this.chkItems.moving, movingActive, `${wm.rates.moving}% (${wm.avg_energy} E)`, `${wm.rates.moving}% (정지)`);

    const rhythmActive = wm.rates.rhythm >= 40;
    const rhythmLabel = metrics.bpm_estimate ? `${wm.rates.rhythm}% (${metrics.bpm_estimate} BPM)` : `${wm.rates.rhythm}% 비트`;
    this.toggleCheckItem(this.chkItems.rhythm, rhythmActive, rhythmLabel, `${wm.rates.rhythm}% 무작위`);
    if (metrics.bpm_estimate && rhythmActive) {
      this.bpmDisplay.textContent = `비트 동기화 (${metrics.bpm_estimate} BPM, 3초 평균 ${wm.rates.rhythm}%)`;
    } else {
      this.bpmDisplay.textContent = `주기적 비트/진동 패턴 (3초 평균 ${wm.rates.rhythm}%)`;
    }

    const dynamicActive = wm.rates.dynamic >= 40;
    this.toggleCheckItem(this.chkItems.dynamic, dynamicActive, `${wm.rates.dynamic}% 충족`, `${wm.rates.dynamic}% 제한적`);

    const sustainedActive = wm.rates.sustained >= 45;
    this.toggleCheckItem(this.chkItems.sustained, sustainedActive, `${wm.rates.sustained}% 유지됨`, `${wm.rates.sustained}% 부족`);

    this.checklistCounter.textContent = `3초 평균: ${wm.active_count} / 5 충족`;
  }

  toggleCheckItem(item, isActive, activeText, inactiveText) {
    if (isActive) {
      item.el.classList.add("active");
      item.state.textContent = activeText;
    } else {
      item.el.classList.remove("active");
      item.state.textContent = inactiveText;
    }
  }

  /**
   * =========================================================================
   * MP4 비디오 녹화 & 내보내기 엔진
   * =========================================================================
   */

  getSupportedMimeType() {
    const types = [
      "video/mp4;codecs=avc1",
      "video/mp4",
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm"
    ];
    for (const t of types) {
      if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(t)) {
        return t;
      }
    }
    return "";
  }

  toggleRecording() {
    if (this.isRecording) {
      this.stopRecording();
    } else {
      this.startRecording();
    }
  }

  startRecording() {
    if (!this.isRunning) {
      this.showToast("먼저 카메라를 시작해주세요.");
      return;
    }

    const rw = this.video.videoWidth || 640;
    const rh = this.video.videoHeight || 480;
    this.recCanvas.width = rw;
    this.recCanvas.height = rh;

    // 30 FPS 캔버스 캡처 스트림 생성
    const stream = this.recCanvas.captureStream(30);
    const mimeType = this.getSupportedMimeType();

    try {
      const options = mimeType ? { mimeType, videoBitsPerSecond: 2500000 } : {};
      this.mediaRecorder = new MediaRecorder(stream, options);
    } catch (e) {
      console.warn("MIME 타입 지정 실패, 기본 레코더 사용:", e);
      this.mediaRecorder = new MediaRecorder(stream);
    }

    this.recordedChunks = [];
    this.recordingScores = [];

    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        this.recordedChunks.push(event.data);
      }
    };

    this.mediaRecorder.onstop = () => {
      this.finishRecordingAndExport();
    };

    this.mediaRecorder.start(250); // 250ms 단위 슬라이스
    this.isRecording = true;
    this.recordingStartTime = Date.now();

    // UI 상태 갱신
    this.recordBtn.classList.add("btn-recording");
    this.recordBtnText.textContent = "녹화 중지 및 MP4 내보내기";
    this.recIndicator.style.display = "inline-flex";
    this.recCameraBadge.style.display = "flex";

    this.updateRecordingTimerDisplay(0);
    this.recordingTimerInterval = setInterval(() => {
      const elapsedSec = Math.floor((Date.now() - this.recordingStartTime) / 1000);
      this.updateRecordingTimerDisplay(elapsedSec);
    }, 500);

    this.showToast("영상 녹화가 시작되었습니다. (30 FPS)");
  }

  updateRecordingTimerDisplay(totalSec) {
    const mins = String(Math.floor(totalSec / 60)).padStart(2, "0");
    const secs = String(totalSec % 60).padStart(2, "0");
    const timeStr = `${mins}:${secs}`;
    this.recTimer.textContent = timeStr;
    this.recCameraTimer.textContent = timeStr;
  }

  stopRecording() {
    if (!this.isRecording || !this.mediaRecorder) return;

    this.recordBtnText.textContent = "MP4 영상 인코딩 중...";
    this.recordBtn.disabled = true;

    clearInterval(this.recordingTimerInterval);
    this.recordingTimerInterval = null;

    if (this.mediaRecorder.state !== "inactive") {
      this.mediaRecorder.stop();
    }

    this.isRecording = false;
    this.recordBtn.classList.remove("btn-recording");
    this.recIndicator.style.display = "none";
    this.recCameraBadge.style.display = "none";
  }

  finishRecordingAndExport() {
    this.recordBtn.disabled = false;
    this.recordBtnText.textContent = "댄스 영상 녹화 시작 (MP4)";

    if (this.recordedChunks.length === 0) {
      this.showToast("녹화된 영상 데이터가 없습니다.");
      return;
    }

    const durationSec = Math.max(1, Math.round((Date.now() - this.recordingStartTime) / 1000));
    const avgScore = this.recordingScores.length > 0
      ? Math.round(this.recordingScores.reduce((a, b) => a + b, 0) / this.recordingScores.length)
      : (this.lastProcessedResult?.windowMetrics?.avg_score || 0);

    const isDancingVerdict = avgScore >= 48;
    const mimeType = this.mediaRecorder.mimeType || "video/mp4";

    // Blob 생성 및 URL 발급
    if (this.lastExportedUrl) {
      URL.revokeObjectURL(this.lastExportedUrl);
    }
    this.lastExportedBlob = new Blob(this.recordedChunks, { type: mimeType });
    this.lastExportedUrl = URL.createObjectURL(this.lastExportedBlob);

    // 모달 데이터 채우기
    this.exportVideoPlayer.src = this.lastExportedUrl;
    this.expScore.textContent = `${avgScore}점`;
    this.expDuration.textContent = `${durationSec}초`;
    this.expVerdict.textContent = isDancingVerdict ? "🔥 춤 인정 (Dancing)" : "🚶 일반 움직임";
    this.expVerdict.style.color = isDancingVerdict ? "var(--neon-pink)" : "var(--neon-cyan)";

    // 모달 오픈
    this.exportModal.classList.add("open");
    this.showToast("🎬 MP4 영상 생성이 완료되었습니다!");
  }

  downloadExportedVideo() {
    if (!this.lastExportedUrl) {
      this.showToast("다운로드할 영상이 없습니다.");
      return;
    }

    const dateStr = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "");
    const ext = this.lastExportedBlob?.type?.includes("webm") ? "webm" : "mp4";
    const filename = `DanceCheck_${dateStr}.${ext}`;

    const a = document.createElement("a");
    a.href = this.lastExportedUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    this.showToast(`⬇️ ${filename} 다운로드를 시작했습니다!`);
  }

  closeExportModal() {
    this.exportModal.classList.remove("open");
    if (this.exportVideoPlayer) {
      this.exportVideoPlayer.pause();
    }
  }

  showToast(message) {
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = message;
    this.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(-10px)";
      setTimeout(() => toast.remove(), 300);
    }, 2800);
  }
}

// 앱 실행
document.addEventListener("DOMContentLoaded", () => {
  new DanceCheckApp();
});
