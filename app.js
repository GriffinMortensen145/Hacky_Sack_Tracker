(() => {
  "use strict";

  // Must match the size used when exporting best.onnx.
  const MODEL_SIZE = 256;
  const MODEL_URL = "./models/hackysack_256.onnx?v=43";

  const CLASS_NAME = "hackysack";

  const IOU_THRESHOLD = 0.45;
  const MAX_DETECTIONS = 5;

  // Older iPhones need time between inference calls.
  const MIN_INFERENCE_INTERVAL_MS = 220;

  // --------------------------------------------------
  // HTML elements
  // --------------------------------------------------

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

  const confidenceSlider =
    document.getElementById("confidence");

  const confidenceValue =
    document.getElementById("confidenceValue");

  const counterText = document.getElementById("counter");
  const message = document.getElementById("message");

  // --------------------------------------------------
  // Application state
  // --------------------------------------------------

  let session = null;
  let cameraStream = null;

  let running = false;
  let inferenceBusy = false;

  let detections = [];

  let lastInferenceTime = 0;
  let lastFpsUpdateTime = performance.now();
  let completedInferenceCount = 0;

  let touchCount = 0;
  let positionHistory = [];
  let lastTouchTime = 0;

  // --------------------------------------------------
  // Interface helpers
  // --------------------------------------------------

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

  function resetTouchCounter() {
    touchCount = 0;
    positionHistory = [];
    lastTouchTime = 0;

    counterText.textContent = "0";
  }

  function resizeDisplay() {
    // Keep this at 1 on older phones to reduce canvas memory usage.
    const pixelRatio = 1;

    display.width = Math.round(
      window.innerWidth * pixelRatio
    );

    display.height = Math.round(
      window.innerHeight * pixelRatio
    );

    display.style.width = `${window.innerWidth}px`;
    display.style.height = `${window.innerHeight}px`;
  }

  // --------------------------------------------------
  // Model loading
  // --------------------------------------------------

  async function verifyModelFile(modelUrl) {
    const response = await fetch(modelUrl, {
      method: "HEAD",
      cache: "no-store"
    });

    if (!response.ok) {
      throw new Error(
        `The model file returned HTTP ${response.status}.\n\n` +
        `Model URL:\n${modelUrl}\n\n` +
        "Confirm that models/hackysack_256.onnx exists."
      );
    }
  }

  async function loadModel() {
    if (session) {
      return;
    }

    if (typeof ort === "undefined") {
      throw new Error(
        "ONNX Runtime did not load. Check the script tag in index.html."
      );
    }

    setStatus("Loading model with WebGL…", "busy");

    const modelUrl = new URL(
      MODEL_URL,
      window.location.href
    ).href;

    console.log("Model URL:", modelUrl);
    console.log("Provider: WebGL");
    console.log("Packed textures: disabled");

    /*
    * Important fix:
    * YOLO contains Resize operations using nearest-neighbor mode.
    * ONNX Runtime WebGL's packed Resize implementation does not
    * support this mode, so packed texture mode must be disabled.
    */
    ort.env.webgl.pack = false;
    ort.env.webgl.contextId = "webgl2";
    ort.env.webgl.textureCacheMode = "initializerOnly";

    try {
      session = await ort.InferenceSession.create(
        modelUrl,
        {
          executionProviders: ["webgl"],
          executionMode: "sequential",
          graphOptimizationLevel: "basic",
          enableCpuMemArena: false,
          enableMemPattern: false
        }
      );
    } catch (error) {
      session = null;

      console.error("WEBGL MODEL ERROR:", error);

      throw new Error(
        "WebGL could not open the model.\n\n" +
        (error?.message || String(error)) +
        "\n\nModel URL:\n" +
        modelUrl
      );
    }

    console.log("Model loaded successfully.");
    console.log("Inputs:", session.inputNames);
    console.log("Outputs:", session.outputNames);

    setStatus("Model ready — WebGL unpacked", "ready");
  }

  // --------------------------------------------------
  // Camera
  // --------------------------------------------------

  async function waitForVideo() {
    if (
      video.readyState >= 2 &&
      video.videoWidth > 0 &&
      video.videoHeight > 0
    ) {
      return;
    }

    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(
          new Error(
            "The camera opened, but video frames did not become ready."
          )
        );
      }, 12000);

      video.onloadeddata = () => {
        clearTimeout(timeout);
        resolve();
      };

      video.onerror = () => {
        clearTimeout(timeout);
        reject(
          new Error("The video element reported an error.")
        );
      };
    });
  }

  async function startCamera() {
    hideMessage();

    startButton.disabled = true;
    startButton.textContent = "Starting…";

    try {
      await loadModel();
    } catch (error) {
      console.error(error);

      setStatus("Model failed", "error");

      startButton.disabled = false;
      startButton.textContent = "Try again";

      showMessage(
        "MODEL ERROR\n\n" +
        (error?.message || String(error))
      );

      return;
    }

    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {
      setStatus("Camera unavailable", "error");

      startButton.disabled = false;
      startButton.textContent = "Try again";

      showMessage(
        "CAMERA ERROR\n\n" +
        "Camera access is unavailable.\n\n" +
        "Open the HTTPS site directly in Safari."
      );

      return;
    }

    try {
      setStatus("Requesting camera…", "busy");

      cameraStream =
        await navigator.mediaDevices.getUserMedia({
          audio: false,

          video: {
            facingMode: {
              ideal: "environment"
            },

            width: {
              ideal: 640
            },

            height: {
              ideal: 480
            },

            frameRate: {
              ideal: 24,
              max: 30
            }
          }
        });

      video.srcObject = cameraStream;

      await waitForVideo();
      await video.play();

      running = true;
      inferenceBusy = false;

      detections = [];

      lastInferenceTime = 0;
      lastFpsUpdateTime = performance.now();
      completedInferenceCount = 0;

      startButton.textContent = "Camera running";

      setStatus("Detecting — WebGL", "ready");

      requestAnimationFrame(renderLoop);
    } catch (error) {
      console.error("CAMERA ERROR:", error);

      setStatus("Camera failed", "error");

      startButton.disabled = false;
      startButton.textContent = "Try again";

      let extraMessage = "";

      if (error?.name === "NotAllowedError") {
        extraMessage =
          "\n\nCamera permission was denied. " +
          "Open Safari Website Settings and set Camera to Allow.";
      } else if (error?.name === "NotFoundError") {
        extraMessage =
          "\n\nSafari could not find a camera.";
      } else if (error?.name === "NotReadableError") {
        extraMessage =
          "\n\nAnother browser tab or app may be using the camera.";
      } else if (error?.name === "OverconstrainedError") {
        extraMessage =
          "\n\nThe requested camera settings are unsupported.";
      }

      showMessage(
        "CAMERA ERROR\n\n" +
        `${error?.name || "Error"}: ` +
        `${error?.message || String(error)}` +
        extraMessage
      );
    }
  }

  function stopCamera() {
    running = false;

    if (cameraStream) {
      for (const track of cameraStream.getTracks()) {
        track.stop();
      }

      cameraStream = null;
    }

    video.srcObject = null;
  }

  // --------------------------------------------------
  // Camera display
  // --------------------------------------------------

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
      width,
      height,

      x: (targetWidth - width) / 2,
      y: (targetHeight - height) / 2
    };
  }

  function drawVideoAndDetections() {
    const canvasWidth = display.width;
    const canvasHeight = display.height;

    displayContext.clearRect(
      0,
      0,
      canvasWidth,
      canvasHeight
    );

    if (!video.videoWidth || !video.videoHeight) {
      return;
    }

    const transform = getCoverTransform(
      video.videoWidth,
      video.videoHeight,
      canvasWidth,
      canvasHeight
    );

    displayContext.drawImage(
      video,
      transform.x,
      transform.y,
      transform.width,
      transform.height
    );

    displayContext.lineWidth = 3;
    displayContext.textBaseline = "top";
    displayContext.font =
      "bold 18px -apple-system, BlinkMacSystemFont, sans-serif";

    for (const detection of detections) {
      const x =
        detection.x1 * transform.scale +
        transform.x;

      const y =
        detection.y1 * transform.scale +
        transform.y;

      const width =
        (detection.x2 - detection.x1) *
        transform.scale;

      const height =
        (detection.y2 - detection.y1) *
        transform.scale;

      displayContext.fillStyle =
        "rgba(52, 211, 153, 0.15)";

      displayContext.strokeStyle = "#34d399";

      displayContext.fillRect(
        x,
        y,
        width,
        height
      );

      displayContext.strokeRect(
        x,
        y,
        width,
        height
      );

      const label =
        `${CLASS_NAME} ` +
        `${Math.round(detection.score * 100)}%`;

      const labelWidth =
        displayContext.measureText(label).width + 12;

      const labelHeight = 27;

      const labelY = Math.max(
        0,
        y - labelHeight
      );

      displayContext.fillStyle = "#34d399";

      displayContext.fillRect(
        x,
        labelY,
        labelWidth,
        labelHeight
      );

      displayContext.fillStyle = "#04120d";

      displayContext.fillText(
        label,
        x + 6,
        labelY + 4
      );
    }
  }

  // --------------------------------------------------
  // Image preprocessing
  // --------------------------------------------------

  function createInputTensor() {
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;

    const scale = Math.min(
      MODEL_SIZE / sourceWidth,
      MODEL_SIZE / sourceHeight
    );

    const resizedWidth = Math.round(
      sourceWidth * scale
    );

    const resizedHeight = Math.round(
      sourceHeight * scale
    );

    const paddingX = Math.floor(
      (MODEL_SIZE - resizedWidth) / 2
    );

    const paddingY = Math.floor(
      (MODEL_SIZE - resizedHeight) / 2
    );

    modelContext.fillStyle =
      "rgb(114, 114, 114)";

    modelContext.fillRect(
      0,
      0,
      MODEL_SIZE,
      MODEL_SIZE
    );

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

    const pixels =
      modelContext.getImageData(
        0,
        0,
        MODEL_SIZE,
        MODEL_SIZE
      ).data;

    const pixelCount =
      MODEL_SIZE * MODEL_SIZE;

    const tensorData =
      new Float32Array(pixelCount * 3);

    for (let i = 0; i < pixelCount; i++) {
      const rgbaIndex = i * 4;

      tensorData[i] =
        pixels[rgbaIndex] / 255;

      tensorData[pixelCount + i] =
        pixels[rgbaIndex + 1] / 255;

      tensorData[pixelCount * 2 + i] =
        pixels[rgbaIndex + 2] / 255;
    }

    return {
      tensor: new ort.Tensor(
        "float32",
        tensorData,
        [1, 3, MODEL_SIZE, MODEL_SIZE]
      ),

      scale,
      paddingX,
      paddingY,
      sourceWidth,
      sourceHeight
    };
  }

  // --------------------------------------------------
  // YOLO output decoding
  // --------------------------------------------------

  function sigmoid(value) {
    return 1 / (1 + Math.exp(-value));
  }

  function normalizeScore(value) {
    if (value >= 0 && value <= 1) {
      return value;
    }

    return sigmoid(value);
  }

  function decodeOutput(
    output,
    preprocessing,
    confidenceThreshold
  ) {
    const dimensions = output.dims;
    const outputData = output.data;

    if (
      dimensions.length !== 3 ||
      dimensions[0] !== 1
    ) {
      throw new Error(
        "Unexpected model output shape: " +
        `[${dimensions.join(", ")}]`
      );
    }

    let featureCount;
    let candidateCount;
    let featuresFirst;

    /*
      Typical one-class YOLOv8 output:

      [1, 5, 1344] at 256×256

      Some exports may instead return:

      [1, 1344, 5]
    */

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
        `Expected at least 5 features per detection, ` +
        `but received ${featureCount}.`
      );
    }

    function valueAt(candidateIndex, featureIndex) {
      if (featuresFirst) {
        return outputData[
          featureIndex * candidateCount +
          candidateIndex
        ];
      }

      return outputData[
        candidateIndex * featureCount +
        featureIndex
      ];
    }

    const boxes = [];

    for (
      let candidateIndex = 0;
      candidateIndex < candidateCount;
      candidateIndex++
    ) {
      const centerX =
        valueAt(candidateIndex, 0);

      const centerY =
        valueAt(candidateIndex, 1);

      const boxWidth =
        valueAt(candidateIndex, 2);

      const boxHeight =
        valueAt(candidateIndex, 3);

      let highestScore = 0;
      let highestClass = 0;

      for (
        let featureIndex = 4;
        featureIndex < featureCount;
        featureIndex++
      ) {
        const score = normalizeScore(
          valueAt(
            candidateIndex,
            featureIndex
          )
        );

        if (score > highestScore) {
          highestScore = score;
          highestClass = featureIndex - 4;
        }
      }

      if (
        highestClass !== 0 ||
        highestScore < confidenceThreshold
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

      x1 = Math.max(
        0,
        Math.min(
          preprocessing.sourceWidth,
          x1
        )
      );

      y1 = Math.max(
        0,
        Math.min(
          preprocessing.sourceHeight,
          y1
        )
      );

      x2 = Math.max(
        0,
        Math.min(
          preprocessing.sourceWidth,
          x2
        )
      );

      y2 = Math.max(
        0,
        Math.min(
          preprocessing.sourceHeight,
          y2
        )
      );

      if (x2 <= x1 || y2 <= y1) {
        continue;
      }

      boxes.push({
        x1,
        y1,
        x2,
        y2,
        score: highestScore
      });
    }

    return nonMaximumSuppression(
      boxes,
      IOU_THRESHOLD,
      MAX_DETECTIONS
    );
  }

  // --------------------------------------------------
  // Non-maximum suppression
  // --------------------------------------------------

  function calculateIoU(first, second) {
    const intersectionX1 = Math.max(
      first.x1,
      second.x1
    );

    const intersectionY1 = Math.max(
      first.y1,
      second.y1
    );

    const intersectionX2 = Math.min(
      first.x2,
      second.x2
    );

    const intersectionY2 = Math.min(
      first.y2,
      second.y2
    );

    const intersectionWidth = Math.max(
      0,
      intersectionX2 - intersectionX1
    );

    const intersectionHeight = Math.max(
      0,
      intersectionY2 - intersectionY1
    );

    const intersectionArea =
      intersectionWidth *
      intersectionHeight;

    const firstArea =
      (first.x2 - first.x1) *
      (first.y2 - first.y1);

    const secondArea =
      (second.x2 - second.x1) *
      (second.y2 - second.y1);

    const unionArea =
      firstArea +
      secondArea -
      intersectionArea;

    return intersectionArea /
      Math.max(unionArea, 0.000001);
  }

  function nonMaximumSuppression(
    boxes,
    threshold,
    limit
  ) {
    const remaining = [...boxes].sort(
      (first, second) =>
        second.score - first.score
    );

    const selected = [];

    while (
      remaining.length > 0 &&
      selected.length < limit
    ) {
      const best = remaining.shift();

      selected.push(best);

      for (
        let index = remaining.length - 1;
        index >= 0;
        index--
      ) {
        if (
          calculateIoU(
            best,
            remaining[index]
          ) >= threshold
        ) {
          remaining.splice(index, 1);
        }
      }
    }

    return selected;
  }

  // --------------------------------------------------
  // Basic touch estimate
  // --------------------------------------------------

  function updateTouchEstimate(
    currentDetections,
    currentTime
  ) {
    if (currentDetections.length === 0) {
      if (
        positionHistory.length > 0 &&
        currentTime -
          positionHistory[
            positionHistory.length - 1
          ].time >
          600
      ) {
        positionHistory = [];
      }

      return;
    }

    const bestDetection =
      currentDetections.reduce(
        (best, current) =>
          current.score > best.score
            ? current
            : best
      );

    const centerY =
      (
        bestDetection.y1 +
        bestDetection.y2
      ) / 2;

    positionHistory.push({
      y: centerY,
      time: currentTime
    });

    while (positionHistory.length > 7) {
      positionHistory.shift();
    }

    if (
      positionHistory.length < 5 ||
      currentTime - lastTouchTime < 420
    ) {
      return;
    }

    const velocities = [];

    for (
      let index = 1;
      index < positionHistory.length;
      index++
    ) {
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

    const midpoint = Math.floor(
      velocities.length / 2
    );

    const before = velocities.slice(
      0,
      midpoint
    );

    const after = velocities.slice(
      midpoint
    );

    if (
      before.length === 0 ||
      after.length === 0
    ) {
      return;
    }

    const beforeAverage =
      before.reduce(
        (sum, value) => sum + value,
        0
      ) / before.length;

    const afterAverage =
      after.reduce(
        (sum, value) => sum + value,
        0
      ) / after.length;

    /*
      Screen Y increases downward.

      Positive velocity = moving downward.
      Negative velocity = moving upward.
    */

    if (
      beforeAverage > 0.07 &&
      afterAverage < -0.07
    ) {
      touchCount++;

      counterText.textContent =
        String(touchCount);

      lastTouchTime = currentTime;

      positionHistory =
        positionHistory.slice(-2);
    }
  }

  // --------------------------------------------------
  // Inference
  // --------------------------------------------------

  async function runInference(currentTime) {
    if (
      !session ||
      inferenceBusy ||
      !video.videoWidth ||
      currentTime -
        lastInferenceTime <
        MIN_INFERENCE_INTERVAL_MS
    ) {
      return;
    }

    inferenceBusy = true;
    lastInferenceTime = currentTime;

    try {
      const preprocessing =
        createInputTensor();

      const inputName =
        session.inputNames[0];

      const feeds = {
        [inputName]: preprocessing.tensor
      };

      const results =
        await session.run(feeds);

      const outputName =
        session.outputNames[0];

      const output = results[outputName];

      const confidenceThreshold =
        Number(
          confidenceSlider.value
        ) / 100;

      detections = decodeOutput(
        output,
        preprocessing,
        confidenceThreshold
      );

      updateTouchEstimate(
        detections,
        currentTime
      );

      completedInferenceCount++;

      if (
        currentTime -
          lastFpsUpdateTime >=
        1000
      ) {
        const fps =
          completedInferenceCount *
          1000 /
          (
            currentTime -
            lastFpsUpdateTime
          );

        fpsText.textContent =
          `${fps.toFixed(1)} FPS`;

        completedInferenceCount = 0;
        lastFpsUpdateTime =
          currentTime;
      }
    } catch (error) {
      console.error(
        "INFERENCE ERROR:",
        error
      );

      setStatus(
        "Inference failed",
        "error"
      );

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

  // --------------------------------------------------
  // Animation loop
  // --------------------------------------------------

  function renderLoop(currentTime) {
    if (!running) {
      return;
    }

    drawVideoAndDetections();

    void runInference(currentTime);

    requestAnimationFrame(renderLoop);
  }

  // --------------------------------------------------
  // Events
  // --------------------------------------------------

  confidenceSlider.addEventListener(
    "input",
    () => {
      confidenceValue.textContent =
        `${confidenceSlider.value}%`;
    }
  );

  resetButton.addEventListener(
    "click",
    resetTouchCounter
  );

  startButton.addEventListener(
    "click",
    startCamera
  );

  window.addEventListener(
    "resize",
    resizeDisplay
  );

  window.addEventListener(
    "orientationchange",
    () => {
      setTimeout(resizeDisplay, 200);
    }
  );

  window.addEventListener(
    "pagehide",
    stopCamera
  );

  resizeDisplay();

  console.log(
    "Hacky Sack WebGL detector loaded — version 40"
  );
})();