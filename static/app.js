/**
 * DanceCheck - app.js
 * 카메라 제어, UI 실시간 갱신, 챌린지 모드, 서버 REST API 통신
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
    this.historyOpenBtn = document.getElementById("history-open-btn");
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

    // 액션 버튼
    this.challengeBtn = document.getElementById("challenge-btn");
    this.saveLogBtn = document.getElementById("save-log-btn");

    // 히스토리 모달
    this.historyModal = document.getElementById("history-modal");
    this.historyCloseBtn = document.getElementById("history-close-btn");
    this.historyList = document.getElementById("history-list");
    this.historyCount = document.getElementById("history-count");
    this.historyRefreshBtn = document.getElementById("history-refresh-btn");
    this.historyClearBtn = document.getElementById("history-clear-btn");

    // 챌린지 결과 모달
    this.resultModal = document.getElementById("result-modal");
    this.resCloseBtn = document.getElementById("res-close-btn");
    this.resSaveBtn = document.getElementById("res-save-btn");
    this.resIcon = document.getElementById("res-icon");
    this.resTitle = document.getElementById("res-title");
    this.resVerdict = document.getElementById("res-verdict");
    this.resScore = document.getElementById("res-score");
    this.resBpm = document.getElementById("res-bpm");
    this.resEnergy = document.getElementById("res-energy");

    // 토스트
    this.toastContainer = document.getElementById("toast-container");

    // 상태 변수
    this.currentFacingMode = "user"; // 'user' (전면) or 'environment' (후면)
    this.stream = null;
    this.isRunning = false;
    this.isFlipping = false;
    this.loopGeneration = 0;
    this.lastProcessedResult = null;

    // 챌린지 모드 상태
    this.isChallengeActive = false;
    this.challengeFrames = [];
    this.challengePendingResult = null;

    this.init();
  }

  async init() {
    this.bindEvents();

    try {
      this.modelStatusText.innerHTML = '<span class="spinner"></span> MediaPipe Pose AI 로드 중...';
      await this.poseEngine.initialize(this.canvas);
      this.modelStatusText.innerHTML = '✅ AI 모델 준비 완료 (Client 100%)';
      this.startBtn.disabled = false;
    } catch (err) {
      console.error(err);
      this.modelStatusText.innerHTML = '❌ 모델 로드 실패: ' + (err.message || '네트워크 확인 필요');
      this.showToast("MediaPipe 모델 로드에 실패했습니다. 페이지를 새로고침해주세요.");
    }
  }

  bindEvents() {
    this.startBtn.addEventListener("click", () => this.startCamera());
    this.cameraFlipBtn.addEventListener("click", () => this.toggleCameraFlip());

    this.historyOpenBtn.addEventListener("click", () => this.openHistoryModal());
    this.historyCloseBtn.addEventListener("click", () => this.closeHistoryModal());
    this.historyRefreshBtn.addEventListener("click", () => this.fetchLogs());
    this.historyClearBtn.addEventListener("click", () => this.clearLogs());

    this.saveLogBtn.addEventListener("click", () => this.saveCurrentSnapshot());
    this.challengeBtn.addEventListener("click", () => this.startChallengeSession());

    this.resCloseBtn.addEventListener("click", () => this.closeResultModal());
    this.resSaveBtn.addEventListener("click", () => this.saveChallengeResult());

    // 화면 회전(가로/세로 전환) 감지 대응
    window.addEventListener("orientationchange", () => {
      setTimeout(() => {
        if (this.isRunning) this.startCamera();
      }, 350);
    });

    // 배경 클릭 시 모달 닫기
    this.historyModal.addEventListener("click", (e) => {
      if (e.target === this.historyModal) this.closeHistoryModal();
    });
    this.resultModal.addEventListener("click", (e) => {
      if (e.target === this.resultModal) this.closeResultModal();
    });
  }

  /**
   * 카메라 권한 획득 및 스트리밍 시작 (iOS Safari / Android Chrome 호환)
   */
  async startCamera() {
    // 이전 루프 무효화
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

    // 엔진 버퍼 및 캔버스 초기화
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

      // 비디오 메타데이터 로드 대기 (해상도 0x0 상태에서 AI 추론 시도 방지)
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

      // 이전 세대 루프가 겹치지 않도록 현재 세대로만 단일 루프 실행
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
   * 렌더링 및 추론 루프 (중복 실행 방지 generation 적용)
   */
  requestFrameLoop(generation) {
    const loop = (timestamp) => {
      // 카메라 전환 등으로 세대가 바뀌었거나 정지된 경우 루프 자연 종료
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
    // 캔버스 크기를 비디오 실제 해상도와 동기화
    if (this.canvas.width !== this.video.videoWidth || this.canvas.height !== this.video.videoHeight) {
      this.canvas.width = this.video.videoWidth || 480;
      this.canvas.height = this.video.videoHeight || 640;
    }

    // AI 엔진 추론 실행
    const result = this.poseEngine.processFrame(this.video, timestamp);
    if (!result) return;

    this.lastProcessedResult = result;

    // 스켈레톤 랜드마크 드로잉 (다인원 지원)
    this.poseEngine.drawLandmarks(
      this.canvasCtx,
      result.allLandmarks,
      this.canvas.width,
      this.canvas.height
    );

    // UI 동기화
    this.updateUI(result);

    // 챌린지 모드 동작 중이면 프레임 기록
    if (this.isChallengeActive) {
      this.challengeFrames.push(result);
    }
  }

  updateUI(result) {
    const { checklist, metrics, isDancing, landmarks, windowMetrics } = result;
    const wm = windowMetrics || {
      is_dancing: isDancing,
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
    } else if (wm.rates.moving >= 35) {
      this.liveStatusBadge.classList.add("moving");
      this.liveStatusText.textContent = `${personPrefix}🚶 단순 행동 중 (움직임 ${wm.rates.moving}%)`;
    } else {
      this.liveStatusText.textContent = `${personPrefix}🧍 정지/휴식 중 (Idle)`;
    }

    // 4. 체크리스트 HUD 업데이트 (최근 3초 평균 충족률 반영)
    // Body (전신 관절)
    const bodyActive = wm.rates.body >= 50;
    this.toggleCheckItem(this.chkItems.body, bodyActive, `${wm.rates.body}% (인식됨)`, `${wm.rates.body}% (미인식)`);

    // Moving (운동 에너지)
    const movingActive = wm.rates.moving >= 45;
    this.toggleCheckItem(this.chkItems.moving, movingActive, `${wm.rates.moving}% (${wm.avg_energy} E)`, `${wm.rates.moving}% (정지)`);

    // Rhythm (리듬 주기성)
    const rhythmActive = wm.rates.rhythm >= 40;
    const rhythmLabel = metrics.bpm_estimate ? `${wm.rates.rhythm}% (${metrics.bpm_estimate} BPM)` : `${wm.rates.rhythm}% 비트`;
    this.toggleCheckItem(this.chkItems.rhythm, rhythmActive, rhythmLabel, `${wm.rates.rhythm}% 무작위`);
    if (metrics.bpm_estimate && rhythmActive) {
      this.bpmDisplay.textContent = `비트 동기화 (${metrics.bpm_estimate} BPM, 3초 평균 ${wm.rates.rhythm}%)`;
    } else {
      this.bpmDisplay.textContent = `주기적 비트/진동 패턴 (3초 평균 ${wm.rates.rhythm}%)`;
    }

    // Dynamic (가동 범위)
    const dynamicActive = wm.rates.dynamic >= 40;
    this.toggleCheckItem(this.chkItems.dynamic, dynamicActive, `${wm.rates.dynamic}% 충족`, `${wm.rates.dynamic}% 제한적`);

    // Sustained (지속 시간)
    const sustainedActive = wm.rates.sustained >= 45;
    this.toggleCheckItem(this.chkItems.sustained, sustainedActive, `${wm.rates.sustained}% 유지됨`, `${wm.rates.sustained}% 부족`);

    // 총 충족 카운터 (3초 평균 기준)
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
   * 7초 댄스 챌린지 모드 시작
   */
  startChallengeSession() {
    if (!this.isRunning) {
      this.showToast("먼저 카메라를 시작해주세요.");
      return;
    }
    if (this.isChallengeActive) return;

    this.isChallengeActive = true;
    this.challengeFrames = [];
    this.challengeBtn.disabled = true;

    let timeLeft = 7;
    this.challengeBtn.textContent = `⏳ 분석 중... ${timeLeft}s`;
    this.showToast("🎵 7초 댄스 챌린지 시작! 신나게 춤을 춰보세요!");

    const timer = setInterval(() => {
      timeLeft--;
      if (timeLeft > 0) {
        this.challengeBtn.textContent = `⏳ 분석 중... ${timeLeft}s`;
      } else {
        clearInterval(timer);
        this.finishChallengeSession();
      }
    }, 1000);
  }

  finishChallengeSession() {
    this.isChallengeActive = false;
    this.challengeBtn.disabled = false;
    this.challengeBtn.textContent = "⏱️ 7초 댄스 판정 챌린지";

    if (this.challengeFrames.length === 0) {
      this.showToast("프레임이 감지되지 않았습니다.");
      return;
    }

    // 7초간 수집된 프레임 통계 계산
    const totalFrames = this.challengeFrames.length;
    const danceFrames = this.challengeFrames.filter(f => f.isDancing).length;
    const danceRatio = danceFrames / totalFrames;

    const avgScore = Math.round(
      this.challengeFrames.reduce((acc, f) => acc + f.metrics.dance_score, 0) / totalFrames
    );

    const bpms = this.challengeFrames.map(f => f.metrics.bpm_estimate).filter(Boolean);
    const avgBpm = bpms.length > 0
      ? Math.round(bpms.reduce((a, b) => a + b, 0) / bpms.length)
      : null;

    const avgEnergy = (
      this.challengeFrames.reduce((acc, f) => acc + f.metrics.energy_level, 0) / totalFrames
    ).toFixed(1);

    // 판정 결과
    const isDancingFinal = (danceRatio >= 0.45 && avgScore >= 55) || (avgScore >= 65);

    // 마지막 프레임의 체크리스트 또는 과반수 체크리스트
    const lastChecklist = this.challengeFrames[this.challengeFrames.length - 1].checklist;

    this.challengePendingResult = {
      is_dancing: isDancingFinal,
      confidence: Math.round(Math.min(100, Math.max(0, isDancingFinal ? (avgScore * 1.05) : (100 - avgScore)))),
      checklist: lastChecklist,
      metrics: {
        dance_score: avgScore,
        energy_level: parseFloat(avgEnergy),
        bpm_estimate: avgBpm,
        periodicity_score: parseFloat((avgScore / 100).toFixed(2)),
        analysis_duration_sec: 7.0,
        fps: this.lastProcessedResult?.metrics?.fps || 30,
        person_count: this.lastProcessedResult?.metrics?.person_count || 1
      },
      notes: isDancingFinal ? "7초 댄스 챌린지 성공 (리듬 및 모션 기준 충족)" : "움직임 또는 리듬감 부족으로 댄스 미충족"
    };

    // 결과 모달 띄우기
    this.resIcon.textContent = isDancingFinal ? "🎉🕺" : "🤔🚶";
    this.resTitle.textContent = isDancingFinal ? "춤 판정 성공!" : "판정: 춤 미충족";
    this.resVerdict.textContent = isDancingFinal
      ? "축하합니다! 역동적인 리듬과 모션이 확인되었습니다."
      : "단순 움직임이거나 리듬 주기성이 부족합니다.";
    this.resVerdict.style.color = isDancingFinal ? "var(--neon-pink)" : "var(--text-muted)";

    this.resScore.textContent = `${avgScore}점`;
    this.resBpm.textContent = avgBpm ? `${avgBpm}` : "--";
    this.resEnergy.textContent = `${avgEnergy}`;

    this.openResultModal();
  }

  /**
   * 실시간 현재 상태 1회 스냅샷 서버 저장
   */
  async saveCurrentSnapshot() {
    if (!this.lastProcessedResult) {
      this.showToast("저장할 분석 결과가 없습니다. 먼저 카메라를 켜주세요.");
      return;
    }

    const { checklist, metrics, isDancing } = this.lastProcessedResult;

    const payload = {
      is_dancing: isDancing,
      confidence: isDancing ? metrics.dance_score : (100 - metrics.dance_score),
      checklist: {
        is_body_detected: checklist.is_body_detected,
        is_moving: checklist.is_moving,
        is_rhythmic: checklist.is_rhythmic,
        is_dynamic: checklist.is_dynamic,
        is_sustained: checklist.is_sustained
      },
      metrics: {
        dance_score: metrics.dance_score,
        energy_level: metrics.energy_level,
        bpm_estimate: metrics.bpm_estimate,
        periodicity_score: metrics.periodicity_score,
        analysis_duration_sec: 2.0,
        fps: metrics.fps,
        person_count: metrics.person_count || 1
      },
      notes: isDancing ? "실시간 스냅샷: 춤 판정" : "실시간 스냅샷: 춤 아님",
      user_agent: navigator.userAgent
    };

    await this.postLogToServer(payload);
  }

  async saveChallengeResult() {
    if (!this.challengePendingResult) return;
    const payload = {
      ...this.challengePendingResult,
      user_agent: navigator.userAgent
    };
    await this.postLogToServer(payload);
    this.closeResultModal();
  }

  async postLogToServer(payload) {
    try {
      this.saveLogBtn.disabled = true;
      const res = await fetch("/api/logs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        throw new Error(`서버 응답 오류 (${res.status})`);
      }

      const data = await res.json();
      this.showToast(`💾 판정 결과가 logs.json에 저장되었습니다! (총 ${data.total_records}건)`);
    } catch (err) {
      console.error(err);
      this.showToast(`저장 실패: ${err.message}`);
    } finally {
      this.saveLogBtn.disabled = false;
    }
  }

  async fetchLogs() {
    try {
      this.historyList.innerHTML = '<div class="empty-state">기록 불러오는 중...</div>';
      const res = await fetch("/api/logs");
      if (!res.ok) throw new Error("로그 조회 실패");
      const logs = await res.json();

      this.historyCount.textContent = `${logs.length}건의 기록`;
      if (logs.length === 0) {
        this.historyList.innerHTML = '<div class="empty-state">저장된 판정 기록이 없습니다.</div>';
        return;
      }

      this.historyList.innerHTML = logs.map(log => {
        const isDance = log.is_dancing;
        const tagClass = isDance ? "dancing" : "not-dancing";
        const cardClass = isDance ? "dancing" : "not-dancing";
        const label = isDance ? "🕺 춤 인정" : "🚶 비-댄스";
        const score = log.metrics?.dance_score || 0;
        const bpm = log.metrics?.bpm_estimate ? `${log.metrics.bpm_estimate} BPM` : "--";
        const chk = log.checklist || {};

        return `
          <div class="log-card ${cardClass}">
            <div class="log-header">
              <span class="log-tag ${tagClass}">${label} (${score}점)</span>
              <span class="log-time">${log.timestamp || ""}</span>
            </div>
            <div class="log-metrics">
              <span>BPM: ${bpm}</span>
              <span>에너지: ${log.metrics?.energy_level ?? "--"}</span>
              <span>신뢰도: ${log.confidence}%</span>
            </div>
            <div class="log-checklist-pills">
              <span class="pill ${chk.is_body_detected ? 'passed' : ''}">관절</span>
              <span class="pill ${chk.is_moving ? 'passed' : ''}">에너지</span>
              <span class="pill ${chk.is_rhythmic ? 'passed' : ''}">리듬</span>
              <span class="pill ${chk.is_dynamic ? 'passed' : ''}">가동범위</span>
              <span class="pill ${chk.is_sustained ? 'passed' : ''}">지속성</span>
            </div>
          </div>
        `;
      }).join("");
    } catch (err) {
      this.historyList.innerHTML = `<div class="empty-state">오류: ${err.message}</div>`;
    }
  }

  async clearLogs() {
    if (!confirm("저장된 모든 판정 기록(logs.json)을 초기화하시겠습니까?")) return;
    try {
      const res = await fetch("/api/logs", { method: "DELETE" });
      if (!res.ok) throw new Error("초기화 실패");
      this.showToast("모든 기록이 초기화되었습니다.");
      this.fetchLogs();
    } catch (err) {
      this.showToast(`초기화 실패: ${err.message}`);
    }
  }

  openHistoryModal() {
    this.historyModal.classList.add("open");
    this.fetchLogs();
  }

  closeHistoryModal() {
    this.historyModal.classList.remove("open");
  }

  openResultModal() {
    this.resultModal.classList.add("open");
  }

  closeResultModal() {
    this.resultModal.classList.remove("open");
  }

  showToast(message) {
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = message;
    this.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.transition = "opacity 0.4s";
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 400);
    }, 2500);
  }
}

// 앱 인스턴스 실행
window.addEventListener("DOMContentLoaded", () => {
  new DanceCheckApp();
});
