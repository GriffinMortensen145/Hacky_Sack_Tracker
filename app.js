(() => {
  "use strict";

  const MODEL_SIZE = 320;
  const MODEL_URL = "./models/best.onnx";
  const CLASS_NAME = "hackysack";
  const IOU_THRESHOLD = 0.45;
  const MAX_DETECTIONS = 10;
  const MIN_INFERENCE_INTERVAL_MS = 120;

  const video = document.getElementById("camera");
  const display = document.getElementById("display");
  const displayCtx = display.getContext("2d");
  const modelCanvas = document.getElementById("modelCanvas");
  const modelCtx = modelCanvas.getContext("2d", { willReadFrequently: true });

  const startButton = document.getElementById("startButton");
  const resetButton = document.getElementById("resetButton");
  const statusText = document.getElementById("status");
  const statusDot = document.getElementById("statusDot");
  const fpsText = document.getElementById("fps");
  const confidenceSlider = document.getElementById("confidence");
  const confidenceValue = document.getElementById("confidenceValue");
  const counterText = document.getElementById("counter");
  const message = document.getElementById("message");

  let session = null;
  let stream = null;
  let running = false;
  let inferenceBusy = false;
  let lastInferenceAt = 0;
  let lastFpsAt = performance.now();
  let inferenceFrames = 0;
  let detections = [];

  let touchCount = 0;
  let history = [];
  let lastCountAt = 0;

  function setStatus(text, type = "") {
    statusText.textContent = text;
    statusDot.className = type;
  }

  function showMessage(text) {
    message.textContent = text;
    message.hidden = false;
  }

  function hideMessage() {
    message.hidden = true;
  }

  function resizeDisplay() {
    const ratio = window.devicePixelRatio || 1;
    display.width = Math.round(window.innerWidth * ratio);
    display.height = Math.round(window.innerHeight * ratio);
    display.style.width = `${window.innerWidth}px`;
    display.style.height = `${window.innerHeight}px`;
  }

  function getCoverTransform(sourceWidth, sourceHeight, targetWidth, targetHeight) {
    const scale = Math.max(targetWidth / sourceWidth, targetHeight / sourceHeight);
    const width = sourceWidth * scale;
    const height = sourceHeight * scale;
    return {
      scale,
      x: (targetWidth - width) / 2,
      y: (targetHeight - height) / 2,
      width,
      height
    };
  }

  async function loadModel() {
    if (session) return;

    setStatus("Loading model…", "busy");

    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    });

    console.log("Input names:", session.inputNames);
    console.log("Output names:", session.outputNames);
    setStatus("Model ready", "ready");
  }

  async function startCamera() {
    hideMessage();
    startButton.disabled = true;

    try {
      await loadModel();

      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });

      video.srcObject = stream;
      await video.play();

      running = true;
      startButton.textContent = "Camera running";
      setStatus("Detecting", "ready");
      requestAnimationFrame(renderLoop);
    } catch (error) {
      console.error(error);
      setStatus("Could not start", "error");
      startButton.disabled = false;
      startButton.textContent = "Try again";

      const detail = error && error.message ? error.message : String(error);
      showMessage(
        "Camera or model failed to start.\n\n" +
        "Open the site directly in Safari, use HTTPS, allow camera access, " +
        "and confirm models/best.onnx exists.\n\n" + detail
      );
    }
  }

  function stopCamera() {
    running = false;
    if (stream) {
      for (const track of stream.getTracks()) track.stop();
      stream = null;
    }
  }

  function drawVideoAndBoxes() {
    const w = display.width;
    const h = display.height;

    displayCtx.clearRect(0, 0, w, h);

    if (!video.videoWidth || !video.videoHeight) return;

    const cover = getCoverTransform(video.videoWidth, video.videoHeight, w, h);
    displayCtx.drawImage(video, cover.x, cover.y, cover.width, cover.height);

    displayCtx.lineWidth = Math.max(3, w / 250);
    displayCtx.font = `bold ${Math.max(18, w / 32)}px -apple-system, sans-serif`;
    displayCtx.textBaseline = "top";

    for (const det of detections) {
      const x = det.x1 * cover.scale + cover.x;
      const y = det.y1 * cover.scale + cover.y;
      const boxW = (det.x2 - det.x1) * cover.scale;
      const boxH = (det.y2 - det.y1) * cover.scale;

      displayCtx.strokeStyle = "#34d399";
      displayCtx.fillStyle = "rgba(52, 211, 153, 0.16)";
      displayCtx.fillRect(x, y, boxW, boxH);
      displayCtx.strokeRect(x, y, boxW, boxH);

      const label = `${CLASS_NAME} ${(det.score * 100).toFixed(0)}%`;
      const metrics = displayCtx.measureText(label);
      const labelHeight = Math.max(26, w / 24);

      displayCtx.fillStyle = "#34d399";
      displayCtx.fillRect(x, Math.max(0, y - labelHeight), metrics.width + 12, labelHeight);
      displayCtx.fillStyle = "#04120d";
      displayCtx.fillText(label, x + 6, Math.max(0, y - labelHeight + 3));
    }
  }

  function createInputTensor() {
    const sourceW = video.videoWidth;
    const sourceH = video.videoHeight;
    const scale = Math.min(MODEL_SIZE / sourceW, MODEL_SIZE / sourceH);
    const drawW = Math.round(sourceW * scale);
    const drawH = Math.round(sourceH * scale);
    const padX = Math.floor((MODEL_SIZE - drawW) / 2);
    const padY = Math.floor((MODEL_SIZE - drawH) / 2);

    modelCtx.fillStyle = "rgb(114,114,114)";
    modelCtx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
    modelCtx.drawImage(video, 0, 0, sourceW, sourceH, padX, padY, drawW, drawH);

    const pixels = modelCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
    const area = MODEL_SIZE * MODEL_SIZE;
    const input = new Float32Array(3 * area);

    for (let i = 0; i < area; i++) {
      const p = i * 4;
      input[i] = pixels[p] / 255;
      input[area + i] = pixels[p + 1] / 255;
      input[2 * area + i] = pixels[p + 2] / 255;
    }

    return {
      tensor: new ort.Tensor("float32", input, [1, 3, MODEL_SIZE, MODEL_SIZE]),
      scale,
      padX,
      padY,
      sourceW,
      sourceH
    };
  }

  function sigmoid(value) {
    return 1 / (1 + Math.exp(-value));
  }

  function normalizedScore(value) {
    return value >= 0 && value <= 1 ? value : sigmoid(value);
  }

  function decodeOutput(output, prep, threshold) {
    const dims = output.dims;
    const data = output.data;

    if (dims.length !== 3 || dims[0] !== 1) {
      throw new Error(`Unexpected ONNX output shape: [${dims.join(", ")}]`);
    }

    let featureCount;
    let candidateCount;
    let featureFirst;

    if (dims[1] <= dims[2]) {
      featureCount = dims[1];
      candidateCount = dims[2];
      featureFirst = true;
    } else {
      candidateCount = dims[1];
      featureCount = dims[2];
      featureFirst = false;
    }

    if (featureCount < 5) {
      throw new Error(`Output has only ${featureCount} features per box.`);
    }

    const at = (candidate, feature) =>
      featureFirst
        ? data[feature * candidateCount + candidate]
        : data[candidate * featureCount + feature];

    const boxes = [];

    for (let i = 0; i < candidateCount; i++) {
      const cx = at(i, 0);
      const cy = at(i, 1);
      const bw = at(i, 2);
      const bh = at(i, 3);

      let bestScore = 0;
      let bestClass = 0;

      for (let c = 4; c < featureCount; c++) {
        const score = normalizedScore(at(i, c));
        if (score > bestScore) {
          bestScore = score;
          bestClass = c - 4;
        }
      }

      if (bestScore < threshold || bestClass !== 0) continue;

      let x1 = (cx - bw / 2 - prep.padX) / prep.scale;
      let y1 = (cy - bh / 2 - prep.padY) / prep.scale;
      let x2 = (cx + bw / 2 - prep.padX) / prep.scale;
      let y2 = (cy + bh / 2 - prep.padY) / prep.scale;

      x1 = Math.max(0, Math.min(prep.sourceW, x1));
      y1 = Math.max(0, Math.min(prep.sourceH, y1));
      x2 = Math.max(0, Math.min(prep.sourceW, x2));
      y2 = Math.max(0, Math.min(prep.sourceH, y2));

      if (x2 <= x1 || y2 <= y1) continue;
      boxes.push({ x1, y1, x2, y2, score: bestScore });
    }

    return nonMaximumSuppression(boxes, IOU_THRESHOLD, MAX_DETECTIONS);
  }

  function intersectionOverUnion(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return intersection / Math.max(1e-6, areaA + areaB - intersection);
  }

  function nonMaximumSuppression(boxes, threshold, limit) {
    boxes.sort((a, b) => b.score - a.score);
    const kept = [];

    while (boxes.length && kept.length < limit) {
      const best = boxes.shift();
      kept.push(best);
      boxes = boxes.filter(box => intersectionOverUnion(best, box) < threshold);
    }

    return kept;
  }

  function updateTouchEstimate(currentDetections, now) {
    if (!currentDetections.length) {
      if (history.length && now - history[history.length - 1].time > 500) {
        history = [];
      }
      return;
    }

    const best = currentDetections[0];
    const centerY = (best.y1 + best.y2) / 2;
    history.push({ y: centerY, time: now });

    while (history.length > 8) history.shift();
    if (history.length < 5 || now - lastCountAt < 350) return;

    const velocities = [];
    for (let i = 1; i < history.length; i++) {
      const dt = Math.max(1, history[i].time - history[i - 1].time);
      velocities.push((history[i].y - history[i - 1].y) / dt);
    }

    const split = Math.floor(velocities.length / 2);
    const before = velocities.slice(0, split);
    const after = velocities.slice(split);
    const beforeAvg = before.reduce((a, b) => a + b, 0) / before.length;
    const afterAvg = after.reduce((a, b) => a + b, 0) / after.length;

    if (beforeAvg > 0.08 && afterAvg < -0.08) {
      touchCount++;
      counterText.textContent = String(touchCount);
      lastCountAt = now;
      history = history.slice(-2);
    }
  }

  async function runInference(now) {
    if (!session || inferenceBusy || !video.videoWidth) return;
    if (now - lastInferenceAt < MIN_INFERENCE_INTERVAL_MS) return;

    inferenceBusy = true;
    lastInferenceAt = now;

    try {
      const prep = createInputTensor();
      const inputName = session.inputNames[0];
      const results = await session.run({ [inputName]: prep.tensor });
      const output = results[session.outputNames[0]];
      const threshold = Number(confidenceSlider.value) / 100;

      detections = decodeOutput(output, prep, threshold);
      updateTouchEstimate(detections, now);

      inferenceFrames++;
      if (now - lastFpsAt >= 1000) {
        const fps = inferenceFrames * 1000 / (now - lastFpsAt);
        fpsText.textContent = `${fps.toFixed(1)} FPS`;
        inferenceFrames = 0;
        lastFpsAt = now;
      }
    } catch (error) {
      console.error(error);
      setStatus("Inference error", "error");
      showMessage(
        "The model loaded, but its input or output format was unexpected.\n\n" +
        (error && error.message ? error.message : String(error))
      );
      stopCamera();
    } finally {
      inferenceBusy = false;
    }
  }

  function renderLoop(now) {
    if (!running) return;
    drawVideoAndBoxes();
    runInference(now);
    requestAnimationFrame(renderLoop);
  }

  confidenceSlider.addEventListener("input", () => {
    confidenceValue.textContent = `${confidenceSlider.value}%`;
  });

  resetButton.addEventListener("click", () => {
    touchCount = 0;
    history = [];
    lastCountAt = 0;
    counterText.textContent = "0";
  });

  startButton.addEventListener("click", startCamera);
  window.addEventListener("resize", resizeDisplay);
  window.addEventListener("pagehide", stopCamera);

  resizeDisplay();

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("Camera unsupported", "error");
    showMessage("Open the HTTPS site directly in Safari.");
    startButton.disabled = true;
  }
})();
