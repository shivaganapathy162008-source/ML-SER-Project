/**
 * app.js — Frontend JavaScript for Speech Emotion Detection
 *
 * Handles:
 * - Microphone recording with MediaRecorder API
 * - Audio upload to FastAPI backend
 * - Displaying prediction results
 * - Prediction history (localStorage)
 * - Backend health check
 * - All error states
 */

'use strict';

// ─── Configuration ─────────────────────────────────────────────────────────
const API_BASE = 'http://127.0.0.1:8000';
const HISTORY_KEY = 'sed_prediction_history';
const MAX_HISTORY = 20;
const MIN_RECORDING_MS = 500;  // Minimum recording before stop is allowed

// ─── Emotion Metadata (icons + colors for display only)
// The actual emotion label comes from the ML model — this is only for UI visuals
const EMOTION_META = {
  'Neutral':   { emoji: '😐', color: '#94a3b8', bg: 'rgba(148,163,184,0.15)' },
  'Calm':      { emoji: '😌', color: '#60a5fa', bg: 'rgba(96,165,250,0.15)' },
  'Happy':     { emoji: '😊', color: '#fbbf24', bg: 'rgba(251,191,36,0.15)' },
  'Sad':       { emoji: '😢', color: '#6366f1', bg: 'rgba(99,102,241,0.15)' },
  'Angry':     { emoji: '😠', color: '#ef4444', bg: 'rgba(239,68,68,0.15)' },
  'Fearful':   { emoji: '😨', color: '#a855f7', bg: 'rgba(168,85,247,0.15)' },
  'Disgust':   { emoji: '🤢', color: '#22c55e', bg: 'rgba(34,197,94,0.15)' },
  'Surprised': { emoji: '😲', color: '#f97316', bg: 'rgba(249,115,22,0.15)' },
};

// Default for unknown emotions (model may return unlisted label)
function getEmotionMeta(emotion) {
  return EMOTION_META[emotion] || { emoji: '🎭', color: '#818cf8', bg: 'rgba(129,140,248,0.15)' };
}

// ─── State ─────────────────────────────────────────────────────────────────
let mediaRecorder = null;
let audioChunks   = [];
let recordingTimer = null;
let recordingStartTime = null;
let currentStream = null;

// ─── DOM Elements ──────────────────────────────────────────────────────────
const stateIdle       = document.getElementById('stateIdle');
const stateRecording  = document.getElementById('stateRecording');
const stateProcessing = document.getElementById('stateProcessing');
const stateError      = document.getElementById('stateError');
const stateResult     = document.getElementById('stateResult');

const startBtn        = document.getElementById('startBtn');
const stopBtn         = document.getElementById('stopBtn');
const retryBtn        = document.getElementById('retryBtn');
const recordAgainBtn  = document.getElementById('recordAgainBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');

const recordTimer     = document.getElementById('recordTimer');
const resultEmoji     = document.getElementById('resultEmoji');
const resultEmotion   = document.getElementById('resultEmotion');
const confBar         = document.getElementById('confBar');
const confValue       = document.getElementById('confValue');
const probBars        = document.getElementById('probBars');
const errorTitle      = document.getElementById('errorTitle');
const errorMessage    = document.getElementById('errorMessage');
const historyList     = document.getElementById('historyList');
const historyEmpty    = document.getElementById('historyEmpty');
const modelBadge      = document.getElementById('modelBadge');
const modelStatus     = document.getElementById('modelStatus');
const badgeDot        = modelBadge.querySelector('.badge-dot');
const emotionsGrid    = document.getElementById('emotionsGrid');


// ─── State Machine ─────────────────────────────────────────────────────────
function showState(name) {
  [stateIdle, stateRecording, stateProcessing, stateError, stateResult].forEach(el => {
    el.classList.add('hidden');
  });
  const map = {
    idle: stateIdle, recording: stateRecording,
    processing: stateProcessing, error: stateError, result: stateResult
  };
  if (map[name]) map[name].classList.remove('hidden');
}


// ─── Health Check ──────────────────────────────────────────────────────────
async function checkHealth() {
  badgeDot.className = 'badge-dot loading';
  modelStatus.textContent = 'Connecting...';

  try {
    const resp = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    const data = await resp.json();

    if (data.model_loaded) {
      badgeDot.className = 'badge-dot online';
      const acc = data.test_accuracy ? ` · ${(data.test_accuracy * 100).toFixed(1)}% acc` : '';
      modelStatus.textContent = `${data.model_type || 'Model'}${acc}`;

      // Load emotion classes
      if (data.emotion_classes) {
        renderEmotionsGrid(data.emotion_classes);
      }
    } else {
      badgeDot.className = 'badge-dot offline';
      modelStatus.textContent = 'Model not loaded';
      console.warn('Model not loaded:', data.error);
    }
  } catch (err) {
    badgeDot.className = 'badge-dot offline';
    modelStatus.textContent = 'Backend offline';
    console.warn('Backend not reachable:', err.message);
  }
}


// ─── Emotions Grid (About section) ─────────────────────────────────────────
function renderEmotionsGrid(emotions) {
  if (!emotionsGrid) return;
  emotionsGrid.innerHTML = '';
  emotions.forEach(emotion => {
    const meta = getEmotionMeta(emotion);
    const chip = document.createElement('div');
    chip.className = 'emotion-chip';
    chip.style.borderColor = meta.color + '40';
    chip.innerHTML = `
      <span class="emotion-chip-icon">${meta.emoji}</span>
      <span style="color: ${meta.color}">${emotion}</span>
    `;
    emotionsGrid.appendChild(chip);
  });
}


// ─── Recording ─────────────────────────────────────────────────────────────
async function startRecording() {
  // Check browser support
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError('Browser Not Supported', 'Your browser does not support microphone access. Try Chrome, Edge, or Firefox.');
    return;
  }

  try {
    // Request microphone access
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,      // Mono
        sampleRate: 22050,    // Match training sample rate if possible
        echoCancellation: true,
        noiseSuppression: true,
      }
    });
    currentStream = stream;

    // Choose best supported MIME type
    const mimeTypes = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/ogg',
      'audio/wav',
    ];
    let mimeType = '';
    for (const type of mimeTypes) {
      if (MediaRecorder.isTypeSupported(type)) {
        mimeType = type;
        break;
      }
    }

    audioChunks = [];
    const options = mimeType ? { mimeType } : {};
    mediaRecorder = new MediaRecorder(stream, options);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      stopStream();
      await processRecording(mimeType || 'audio/webm');
    };

    mediaRecorder.onerror = (event) => {
      console.error('MediaRecorder error:', event.error);
      showError('Recording Error', 'An error occurred during recording. Please try again.');
    };

    // Start recording, collect data every 500ms
    mediaRecorder.start(500);
    recordingStartTime = Date.now();
    showState('recording');
    startTimerUI();

  } catch (err) {
    handleMicError(err);
  }
}


function stopRecording() {
  if (!mediaRecorder || mediaRecorder.state === 'inactive') return;

  const elapsed = Date.now() - recordingStartTime;
  if (elapsed < MIN_RECORDING_MS) {
    // Too short — wait a tiny bit
    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
      }
    }, MIN_RECORDING_MS - elapsed);
    return;
  }

  mediaRecorder.stop();
  stopTimerUI();
}


function stopStream() {
  if (currentStream) {
    currentStream.getTracks().forEach(track => track.stop());
    currentStream = null;
  }
}


function handleMicError(err) {
  stopStream();
  let title = 'Microphone Error';
  let msg = 'Could not access microphone.';

  if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
    title = 'Microphone Permission Denied';
    msg = 'Please allow microphone access in your browser settings, then try again.';
  } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
    title = 'No Microphone Found';
    msg = 'No microphone was detected. Please connect a microphone and try again.';
  } else if (err.name === 'NotReadableError') {
    title = 'Microphone Busy';
    msg = 'The microphone is being used by another application. Please close it and try again.';
  } else if (err.name === 'SecurityError') {
    title = 'Security Error';
    msg = 'Microphone access is blocked. Make sure you are on a secure connection (HTTPS or localhost).';
  }

  showError(title, msg);
}


// ─── Timer UI ──────────────────────────────────────────────────────────────
function startTimerUI() {
  recordingStartTime = recordingStartTime || Date.now();
  recordTimer.textContent = '00:00';
  recordingTimer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
    const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const ss = String(elapsed % 60).padStart(2, '0');
    recordTimer.textContent = `${mm}:${ss}`;
  }, 500);
}

function stopTimerUI() {
  if (recordingTimer) { clearInterval(recordingTimer); recordingTimer = null; }
}


// ─── Process Recording ─────────────────────────────────────────────────────
async function processRecording(mimeType) {
  if (audioChunks.length === 0) {
    showError('Empty Recording', 'No audio was captured. Please try again.');
    return;
  }

  // Create audio blob
  const audioBlob = new Blob(audioChunks, { type: mimeType || 'audio/webm' });

  if (audioBlob.size < 1000) {
    showError('Recording Too Short', 'The recording was too short. Please speak for at least 1 second.');
    return;
  }

  showState('processing');

  // Determine file extension from MIME type
  let ext = 'webm';
  if (mimeType.includes('ogg'))  ext = 'ogg';
  else if (mimeType.includes('wav')) ext = 'wav';
  else if (mimeType.includes('mp4')) ext = 'mp4';

  const formData = new FormData();
  formData.append('audio', audioBlob, `recording.${ext}`);

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(30000),  // 30s timeout
    });

    if (!response.ok) {
      let detail = 'Prediction failed.';
      try {
        const errData = await response.json();
        detail = errData.detail || detail;
      } catch (_) {}

      if (response.status === 503) {
        showError('Model Not Ready', 'The ML model is not loaded. Please run training first: python src/train.py');
      } else if (response.status === 400) {
        showError('Audio Issue', detail);
      } else {
        showError('Prediction Failed', detail);
      }
      return;
    }

    const result = await response.json();
    displayResult(result);
    saveToHistory(result);

  } catch (err) {
    if (err.name === 'TimeoutError' || err.name === 'AbortError') {
      showError('Request Timeout', 'The server took too long to respond. Please try a shorter recording.');
    } else if (err.name === 'TypeError' && err.message.includes('fetch')) {
      showError('Backend Offline', 'Cannot reach the backend server. Make sure it is running on port 8000.');
    } else {
      showError('Network Error', `${err.message}. Please check your connection and try again.`);
    }
  }
}


// ─── Display Results ────────────────────────────────────────────────────────
function displayResult(data) {
  const { emotion, confidence, probabilities } = data;
  const meta = getEmotionMeta(emotion);

  // Emotion icon + label
  resultEmoji.textContent = meta.emoji;
  resultEmotion.textContent = emotion.toUpperCase();
  resultEmotion.style.background = `linear-gradient(135deg, ${meta.color}, #a855f7)`;
  resultEmotion.style.webkitBackgroundClip = 'text';
  resultEmotion.style.backgroundClip = 'text';
  resultEmotion.style.webkitTextFillColor = 'transparent';

  // Confidence bar (animate after slight delay)
  const confPct = Math.round(confidence * 100);
  confValue.textContent = `${confPct}%`;
  setTimeout(() => {
    confBar.style.width = `${confPct}%`;
  }, 100);

  // Probability bars
  renderProbabilityBars(probabilities, emotion);

  showState('result');
}


function renderProbabilityBars(probabilities, topEmotion) {
  probBars.innerHTML = '';

  // Sort by probability descending
  const sorted = Object.entries(probabilities)
    .sort(([, a], [, b]) => b - a);

  sorted.forEach(([emotion, prob], index) => {
    const meta = getEmotionMeta(emotion);
    const pct = Math.round(prob * 100);
    const isTop = emotion === topEmotion;

    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span class="prob-emotion-name">${meta.emoji} ${emotion}</span>
      <div class="prob-bar-wrap">
        <div class="prob-bar-fill" 
             style="background: ${meta.color}; --delay: ${index * 0.05}s; opacity: ${isTop ? 1 : 0.65};"
             data-target="${pct}"></div>
      </div>
      <span class="prob-pct ${isTop ? 'top' : ''}">${pct}%</span>
    `;
    probBars.appendChild(row);

    // Animate bar width
    setTimeout(() => {
      const bar = row.querySelector('.prob-bar-fill');
      if (bar) bar.style.width = `${pct}%`;
    }, 150 + index * 50);
  });
}


// ─── Error Display ──────────────────────────────────────────────────────────
function showError(title, message) {
  errorTitle.textContent = title;
  errorMessage.textContent = message;
  stopStream();
  stopTimerUI();
  showState('error');
}


// ─── Prediction History (localStorage) ─────────────────────────────────────
function saveToHistory(data) {
  const history = getHistory();
  const entry = {
    id:         Date.now(),
    emotion:    data.emotion,
    confidence: data.confidence,
    timestamp:  new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    fullTime:   new Date().toLocaleString(),
  };
  history.unshift(entry);
  if (history.length > MAX_HISTORY) history.splice(MAX_HISTORY);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  renderHistory();
}

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  } catch { return []; }
}

function clearHistory() {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
}

function renderHistory() {
  const history = getHistory();
  const items = historyList.querySelectorAll('.history-item');
  items.forEach(el => el.remove());

  if (history.length === 0) {
    historyEmpty.style.display = '';
    return;
  }

  historyEmpty.style.display = 'none';

  history.forEach(entry => {
    const meta = getEmotionMeta(entry.emotion);
    const pct = Math.round(entry.confidence * 100);
    const item = document.createElement('div');
    item.className = 'history-item';
    item.style.borderLeft = `3px solid ${meta.color}`;
    item.innerHTML = `
      <span class="history-emoji">${meta.emoji}</span>
      <div class="history-info">
        <div class="history-emotion" style="color: ${meta.color}">${entry.emotion}</div>
        <div class="history-time">${entry.fullTime || entry.timestamp}</div>
      </div>
      <span class="history-conf">${pct}%</span>
    `;
    historyList.appendChild(item);
  });
}


// ─── Event Listeners ───────────────────────────────────────────────────────
startBtn.addEventListener('click', () => startRecording());
stopBtn.addEventListener('click', () => stopRecording());
retryBtn.addEventListener('click', () => showState('idle'));
recordAgainBtn.addEventListener('click', () => {
  audioChunks = [];
  showState('idle');
});
clearHistoryBtn.addEventListener('click', () => {
  if (confirm('Clear all prediction history?')) clearHistory();
});

// Smooth scroll nav
document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    const target = link.getAttribute('href');
    const section = document.querySelector(target);
    if (section) section.scrollIntoView({ behavior: 'smooth', block: 'start' });

    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    link.classList.add('active');
  });
});

// Update active nav on scroll
const sections = ['recorder', 'history', 'about'];
const navLinks = { recorder: document.getElementById('nav-recorder'), history: document.getElementById('nav-history'), about: document.getElementById('nav-about') };
window.addEventListener('scroll', () => {
  let current = 'recorder';
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el && window.scrollY >= el.offsetTop - 100) current = id;
  });
  Object.values(navLinks).forEach(l => l && l.classList.remove('active'));
  if (navLinks[current]) navLinks[current].classList.add('active');
}, { passive: true });


// ─── Init ───────────────────────────────────────────────────────────────────
(function init() {
  showState('idle');
  renderHistory();
  checkHealth();

  // Re-check health every 30 seconds
  setInterval(checkHealth, 30000);

  // Load emotions grid from API
  fetch(`${API_BASE}/emotions`)
    .then(r => r.json())
    .then(data => { if (data.emotions) renderEmotionsGrid(data.emotions); })
    .catch(() => {
      // If API offline, show default emotions
      renderEmotionsGrid(Object.keys(EMOTION_META));
    });

  console.log('🎙️ Speech Emotion Detection — Frontend loaded');
  console.log(`API: ${API_BASE}`);
})();
