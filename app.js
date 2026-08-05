(() => {
  "use strict";

  const MODEL_SIZE = 320;
  const MODEL_URL = "./models/best.onnx";
  const CLASS_NAME = "hackysack";

  const IOU_THRESHOLD = 0.45;
  const MAX_DETECTIONS = 10;
  const MIN_INFERENCE_INTERVAL_MS = 140;

  const video = document.getElementById("camera");
  const display = document.getElementById("display");
  const displayContext = display.getContext("2d");

  const modelCanvas = document.getElementById("modelCanvas");
  const modelContext = modelCanvas.getContext("2d", {
    willReadFrequently: true
  });

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
  let lastInferenceTime = 0;
  let lastFpsTime = performance.now();
  let inferenceFrameCount = 0;
  let detections = [];

  let touchCount = 0;
  let positionHistory = [];
  let lastTouchTime = 0;

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
    const pixelRatio = window.devicePixelRatio || 1;

    display.width = Math.round(window.innerWidth * pixelRatio);
    display.height = Math.round(window.innerHeight * pixelRatio);

    display.style.width = `${window.innerWidth}px`;
    display.style.height = `${window.innerHeight}px`;
  }

  function getCoverTransform(
    sourceWidth,
    sourceHeight,
    targetWidth,
    targetHeight
  ) {
    const scale = Math.max(
      targetWidth / sourceWidth,
      targetHeight / sourceHeight
    );

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
    if (session) {
      return;
    }

    setStatus("Loading model…", "busy");

    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

    session = await ort.InferenceSession.create(
      MODEL_URL,
      {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      }
    );

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
          width: { ideal: 640 },
          height: { ideal: 480 }
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

      showMessage(
        "Camera or model failed to start.\n\n" +
        (error?.message || String(error))
      );
    }
  }

  function stopCamera() {
    running = false;

    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }

      stream = null;
    }

    video.srcObject = null;
  }

  function drawVideoAndBoxes() {
    const canvasWidth = display.width;
    const canvasHeight = display.height;

    displayContext.clearRect(0, 0, canvasWidth, canvasHeight);

    if (!video.videoWidth || !video.videoHeight) {
      return;
    }

    const cover = getCoverTransform(
      video.videoWidth,
      video.videoHeight,
      canvasWidth,
      canvasHeight
    );

    displayContext.drawImage(
      video,
      cover.x,
      cover.y,
      cover.width,
      cover.height
    );

    displayContext.lineWidth = Math.max(3, canvasWidth / 250);
    displayContext.font =
      `bold ${Math.max(18, canvasWidth / 32)}px -apple-system, sans-serif`;
    displayContext.textBaseline = "top";

    for (const detection of detections) {
      const x = detection.x1 * cover.scale + cover.x;
      const y = detection.y1 * cover.scale + cover.y;
      const width =
        (detection.x2 - detection.x1) * cover.scale;
      const height =
        (detection.y2 - detection.y1) * cover.scale;

      displayContext.strokeStyle = "#34d399";
      displayContext.fillStyle = "rgba(52, 211, 153, 0.16)";

      displayContext.fillRect(x, y, width, height);
      displayContext.strokeRect(x, y, width, height);

      const label =
        `${CLASS_NAME} ${(detection.score * 100).toFixed(0)}%`;

      const measurement = displayContext.measureText(label);
      const labelHeight = Math.max(26, canvasWidth / 24);

      displayContext.fillStyle = "#34d399";
      displayContext.fillRect(
        x,
        Math.max(0, y - labelHeight),
        measurement.width + 12,
        labelHeight
      );

      displayContext.fillStyle = "#04120d";
      displayContext.fillText(
        label,
        x + 6,
        Math.max(0, y - labelHeight + 3)
      );
    }
  }

  function createInputTensor() {
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;

    const scale = Math.min(
      MODEL_SIZE / sourceWidth,
      MODEL_SIZE / sourceHeight
    );

    const resizedWidth = Math.round(sourceWidth * scale);
    const resizedHeight = Math.round(sourceHeight * scale);

    const paddingX = Math.floor(
      (MODEL_SIZE - resizedWidth) / 2
    );

    const paddingY = Math.floor(
      (MODEL_SIZE - resizedHeight) / 2
    );

    modelContext.fillStyle = "rgb(114, 114, 114)";
    modelContext.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);

    modelContext.drawImage(
      video,
      0,
      0,
      sourceWidth,
      sourceHeight,
      paddingX,
      paddingY,
      resizedWidth,
      resizedHeight
    );

    const pixels = modelContext.getImageData(
      0,
      0,
      MODEL_SIZE,
      MODEL_SIZE
    ).data;

    const imageArea = MODEL_SIZE * MODEL_SIZE;
    const input = new Float32Array(3 * imageArea);

    for (let index = 0; index < imageArea; index++) {
      const pixelIndex = index * 4;

      input[index] = pixels[pixelIndex] / 255;
      input[imageArea + index] = pixels[pixelIndex + 1] / 255;
      input[2 * imageArea + index] = pixels[pixelIndex + 2] / 255;
    }

    return {
      tensor: new ort.Tensor(
        "float32",
        input,
        [1, 3, MODEL_SIZE, MODEL_SIZE]
      ),
      scale,
      paddingX,
      paddingY,
      sourceWidth,
      sourceHeight
    };
  }

  function sigmoid(value) {
    return 1 / (1 + Math.exp(-value));
  }

  function normalizeScore(value) {
    return value >= 0 && value <= 1 ? value : sigmoid(value);
  }

  function decodeOutput(
    output,
    preprocessing,
    confidenceThreshold
  ) {
    const dimensions = output.dims;
    const outputData = output.data;

    if (dimensions.length !== 3 || dimensions[0] !== 1) {
      throw new Error(
        `Unexpected ONNX output shape: [${dimensions.join(", ")}]`
      );
    }

    let featureCount;
    let candidateCount;
    let featuresFirst;

    if (dimensions[1] <= dimensions[2]) {
      featureCount = dimensions[1];
      candidateCount = dimensions[2];
      featuresFirst = true;
    } else {
      candidateCount = dimensions[1];
      featureCount = dimensions[2];
      featuresFirst = false;
    }

    if (featureCount < 5) {
      throw new Error(
        `Output only has ${featureCount} features per box.`
      );
    }

    function getValue(candidate, feature) {
      return featuresFirst
        ? outputData[feature * candidateCount + candidate]
        : outputData[candidate * featureCount + feature];
    }

    const boxes = [];

    for (let candidate = 0; candidate < candidateCount; candidate++) {
      const centerX = getValue(candidate, 0);
      const centerY = getValue(candidate, 1);
      const boxWidth = getValue(candidate, 2);
      const boxHeight = getValue(candidate, 3);

      let bestScore = 0;
      let bestClass = 0;

      for (let feature = 4; feature < featureCount; feature++) {
        const score = normalizeScore(
          getValue(candidate, feature)
        );

        if (score > bestScore) {
          bestScore = score;
          bestClass = feature - 4;
        }
      }

      if (
        bestScore < confidenceThreshold ||
        bestClass !== 0
      ) {
        continue;
      }

      let x1 =
        (
          centerX -
          boxWidth / 2 -
          preprocessing.paddingX
        ) / preprocessing.scale;

      let y1 =
        (
          centerY -
          boxHeight / 2 -
          preprocessing.paddingY
        ) / preprocessing.scale;

      let x2 =
        (
          centerX +
          boxWidth / 2 -
          preprocessing.paddingX
        ) / preprocessing.scale;

      let y2 =
        (
          centerY +
          boxHeight / 2 -
          preprocessing.paddingY
        ) / preprocessing.scale;

      x1 = Math.max(0, Math.min(preprocessing.sourceWidth, x1));
      y1 = Math.max(0, Math.min(preprocessing.sourceHeight, y1));
      x2 = Math.max(0, Math.min(preprocessing.sourceWidth, x2));
      y2 = Math.max(0, Math.min(preprocessing.sourceHeight, y2));

      if (x2 <= x1 || y2 <= y1) {
        continue;
      }

      boxes.push({
        x1,
        y1,
        x2,
        y2,
        score: bestScore
      });
    }

    return nonMaximumSuppression(
      boxes,
      IOU_THRESHOLD,
      MAX_DETECTIONS
    );
  }

  function intersectionOverUnion(first, second) {
    const intersectionX1 = Math.max(first.x1, second.x1);
    const intersectionY1 = Math.max(first.y1, second.y1);
    const intersectionX2 = Math.min(first.x2, second.x2);
    const intersectionY2 = Math.min(first.y2, second.y2);

    const intersection =
      Math.max(0, intersectionX2 - intersectionX1) *
      Math.max(0, intersectionY2 - intersectionY1);

    const firstArea =
      (first.x2 - first.x1) *
      (first.y2 - first.y1);

    const secondArea =
      (second.x2 - second.x1) *
      (second.y2 - second.y1);

    return intersection / Math.max(
      0.000001,
      firstArea + secondArea - intersection
    );
  }

  function nonMaximumSuppression(
    boxes,
    threshold,
    limit
  ) {
    boxes.sort(
      (first, second) => second.score - first.score
    );

    const kept = [];

    while (boxes.length > 0 && kept.length < limit) {
      const best = boxes.shift();
      kept.push(best);

      boxes = boxes.filter(
        box =>
          intersectionOverUnion(best, box) < threshold
      );
    }

    return kept;
  }

  function updateTouchEstimate(
    currentDetections,
    currentTime
  ) {
    if (!currentDetections.length) {
      if (
        positionHistory.length > 0 &&
        currentTime -
          positionHistory[positionHistory.length - 1].time >
          500
      ) {
        positionHistory = [];
      }

      return;
    }

    const bestDetection = currentDetections[0];

    const centerY =
      (bestDetection.y1 + bestDetection.y2) / 2;

    positionHistory.push({
      y: centerY,
      time: currentTime
    });

    while (positionHistory.length > 8) {
      positionHistory.shift();
    }

    if (
      positionHistory.length < 5 ||
      currentTime - lastTouchTime < 350
    ) {
      return;
    }

    const velocities = [];

    for (let index = 1; index < positionHistory.length; index++) {
      const timeDifference = Math.max(
        1,
        positionHistory[index].time -
          positionHistory[index - 1].time
      );

      velocities.push(
        (
          positionHistory[index].y -
          positionHistory[index - 1].y
        ) / timeDifference
      );
    }

    const splitIndex = Math.floor(velocities.length / 2);
    const before = velocities.slice(0, splitIndex);
    const after = velocities.slice(splitIndex);

    const beforeAverage =
      before.reduce((total, value) => total + value, 0) /
      before.length;

    const afterAverage =
      after.reduce((total, value) => total + value, 0) /
      after.length;

    if (
      beforeAverage > 0.08 &&
      afterAverage < -0.08
    ) {
      touchCount++;
      counterText.textContent = String(touchCount);
      lastTouchTime = currentTime;
      positionHistory = positionHistory.slice(-2);
    }
  }

  async function runInference(currentTime) {
    if (
      !session ||
      inferenceBusy ||
      !video.videoWidth
    ) {
      return;
    }

    if (
      currentTime -
        lastInferenceTime <
      MIN_INFERENCE_INTERVAL_MS
    ) {
      return;
    }

    inferenceBusy = true;
    lastInferenceTime = currentTime;

    try {
      const preprocessing = createInputTensor();
      const inputName = session.inputNames[0];

      const results = await session.run({
        [inputName]: preprocessing.tensor
      });

      const output = results[session.outputNames[0]];

      const confidenceThreshold =
        Number(confidenceSlider.value) / 100;

      detections = decodeOutput(
        output,
        preprocessing,
        confidenceThreshold
      );

      updateTouchEstimate(
        detections,
        currentTime
      );

      inferenceFrameCount++;

      if (
        currentTime -
          lastFpsTime >=
        1000
      ) {
        const framesPerSecond =
          inferenceFrameCount *
          1000 /
          (
            currentTime -
            lastFpsTime
          );

        fpsText.textContent =
          `${framesPerSecond.toFixed(1)} FPS`;

        inferenceFrameCount = 0;
        lastFpsTime = currentTime;
      }
    } catch (error) {
      console.error(error);

      setStatus("Inference error", "error");

      showMessage(
        "INFERENCE ERROR\n\n" +
        (error?.message || String(error))
      );

      stopCamera();

      startButton.disabled = false;
      startButton.textContent = "Try again";
    } finally {
      inferenceBusy = false;
    }
  }

  function renderLoop(currentTime) {
    if (!running) {
      return;
    }

    drawVideoAndBoxes();
    void runInference(currentTime);
    requestAnimationFrame(renderLoop);
  }

  confidenceSlider.addEventListener("input", () => {
    confidenceValue.textContent =
      `${confidenceSlider.value}%`;
  });

  resetButton.addEventListener("click", () => {
    touchCount = 0;
    positionHistory = [];
    lastTouchTime = 0;
    counterText.textContent = "0";
  });

  startButton.addEventListener("click", startCamera);
  window.addEventListener("resize", resizeDisplay);
  window.addEventListener("pagehide", stopCamera);

  resizeDisplay();
})();
