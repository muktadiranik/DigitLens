const stage = document.querySelector("#stage");
const surface = document.querySelector("#surface");
const camera = document.querySelector("#camera");
const still = document.querySelector("#still");
const scanGuide = document.querySelector("#scan-guide");
const boxes = document.querySelector("#boxes");
const placeholder = document.querySelector("#placeholder");
const placeholderText = document.querySelector("#placeholder-text");
const start = document.querySelector("#start");
const stop = document.querySelector("#stop");
const upload = document.querySelector("#upload");
const fileInput = document.querySelector("#file");
const sourceIndicator = document.querySelector("#source-indicator");
const sourceText = document.querySelector("#source-text");
const notice = document.querySelector("#notice");
const sequence = document.querySelector("#sequence");
const emptyResult = document.querySelector("#empty-result");
const resultNote = document.querySelector("#result-note");
const scanModes = document.querySelector(".scan-modes");
const layoutTip = document.querySelector("#layout-tip");
const spacingTip = document.querySelector("#spacing-tip");

const FULL_IMAGE = { x: 0, y: 0, width: 1, height: 1 };
const SCAN = { x: 0.08, y: 0.21, width: 0.84, height: 0.58 };
const FRAME_INTERVAL = 250;
const TEXT_FRAME_INTERVAL = 450;
const SCAN_ENDPOINTS = {
  digits: "/api/recognize",
  text: "/api/recognize-text",
  handwriting: "/api/recognize-handwriting",
};
const MAX_UPLOAD_BYTES = 12 * 1024 * 1024;
const IMAGE_TYPES = {
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  png: "image/png",
  webp: "image/webp",
};
const canvas = document.createElement("canvas");
const context = canvas.getContext("2d", { alpha: false });

surface.style.setProperty("--scan-x", `${SCAN.x * 100}%`);
surface.style.setProperty("--scan-y", `${SCAN.y * 100}%`);
surface.style.setProperty("--scan-w", `${SCAN.width * 100}%`);
surface.style.setProperty("--scan-h", `${SCAN.height * 100}%`);

let mode = "idle";
let scanMode = "digits";
let scanRevision = 0;
let stream = null;
let timer = null;
let imageUrl = null;
let currentFile = null;
let operation = new AbortController();
let previous = [];
let stable = [];

function layoutSurface() {
  const width = mode === "live" ? camera.videoWidth : still.naturalWidth;
  const height = mode === "live" ? camera.videoHeight : still.naturalHeight;
  if (!["live", "image"].includes(mode) || !width || !height) {
    surface.style.width = "100%";
    surface.style.height = "100%";
    return;
  }
  const scale = Math.min(stage.clientWidth / width, stage.clientHeight / height);
  surface.style.width = `${width * scale}px`;
  surface.style.height = `${height * scale}px`;
}

function setMode(nextMode) {
  mode = nextMode;
  const usingCamera = mode === "starting" || mode === "live";
  start.disabled = usingCamera;
  stop.disabled = !usingCamera;
  camera.hidden = mode !== "live";
  still.hidden = mode !== "image";
  scanGuide.hidden = mode === "image" || mode === "loading";
  placeholder.hidden = mode === "live" || mode === "image";
  placeholderText.textContent = mode === "starting" ? "Starting camera…" : "Camera is off";
  if (mode === "loading") placeholderText.textContent = "Opening image…";
  sourceText.textContent = {
    idle: "Camera off",
    starting: "Starting camera",
    live: "Camera live",
    loading: "Opening image",
    image: "Uploaded image",
  }[mode];
  sourceIndicator.dataset.state = mode;
  layoutSurface();
}

function showNotice(message) {
  if (notice.textContent !== message) notice.textContent = message;
  notice.hidden = !message;
}

function showResult(items, region = FULL_IMAGE) {
  const english = scanMode !== "digits";
  const text = items.map((item) => item.label).filter(Boolean).join(english ? "\n" : "");
  if (sequence.textContent !== text) sequence.textContent = text;
  sequence.hidden = !text;
  emptyResult.hidden = Boolean(text);
  const emptyMessage = english ? "No text detected" : "No digits detected";
  if (emptyResult.textContent !== emptyMessage) emptyResult.textContent = emptyMessage;
  let note = "";
  if (items.some((item) => item.label && item.uncertain)) {
    note = english ? "Check the text outlined in amber." : "A question mark indicates an uncertain digit.";
  }
  if (resultNote.textContent !== note) resultNote.textContent = note;
  resultNote.hidden = !note;

  const elements = items.map((item) => {
    const box = document.createElement("div");
    box.className = "digit-box";
    const x = region.x + item.x * region.width;
    const y = region.y + item.y * region.height;
    box.style.left = `${x * 100}%`;
    box.style.top = `${y * 100}%`;
    box.style.width = `${item.width * region.width * 100}%`;
    box.style.height = `${item.height * region.height * 100}%`;
    box.toggleAttribute("data-uncertain", item.uncertain);
    if (!english) {
      box.toggleAttribute("data-top", y * surface.clientHeight < 23);
      const label = document.createElement("span");
      label.textContent = item.label;
      box.append(label);
    }
    return box;
  });
  boxes.replaceChildren(...elements);
}

function stabilize(items) {
  // Confirm changed labels over two successive frames. Empty frames clear immediately.
  const sameRow = items.length === previous.length && items.every((box, index) => (
    box.row === previous[index].row
    && Math.abs(box.x - previous[index].x) < 0.10
    && Math.abs(box.y - previous[index].y) < 0.17
    && Math.abs(box.width - previous[index].width) < 0.08
    && Math.abs(box.height - previous[index].height) < 0.18
  ));
  stable = items.map((box, index) => {
    if (!sameRow) {
      return scanMode === "digits"
        ? { label: "?", uncertain: true }
        : { label: box.label, uncertain: box.uncertain };
    }
    return previous[index].label === box.label
      ? { label: box.label, uncertain: box.uncertain }
      : stable[index];
  });
  previous = items;
  return items.map((box, index) => ({ ...box, ...stable[index] }));
}

function clearResult() {
  previous = [];
  stable = [];
  showResult([]);
}

function resetSource(nextMode = "idle") {
  operation.abort();
  operation = new AbortController();
  clearTimeout(timer);
  currentFile = null;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }
  camera.srcObject = null;
  if (imageUrl) {
    still.removeAttribute("src");
    URL.revokeObjectURL(imageUrl);
    imageUrl = null;
  }
  setMode(nextMode);
  clearResult();
  showNotice("");
  return operation.signal;
}

class ScanError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function recognize(image, type, signal, target) {
  let response;
  try {
    response = await fetch(SCAN_ENDPOINTS[target], {
      method: "POST",
      headers: { "Content-Type": type },
      body: image,
      signal: AbortSignal.any([signal, AbortSignal.timeout(target === "handwriting" ? 20000 : 10000)]),
    });
  } catch {
    throw new ScanError("The recognition engine did not respond. Check that DigitLens is running.");
  }
  if (response.status === 404 && target !== "digits") {
    throw new ScanError("This server is missing English OCR. Restart the updated DigitLens server using the README.", 404);
  }
  const result = await response.json().catch(() => null);
  if (!response.ok || !result) {
    throw new ScanError(result?.error ?? "The recognition engine could not process the image.", response.status);
  }
  if (target !== "digits") {
    return result.regions.map((item) => ({ ...item, label: item.text }));
  }
  return result.digits.map((item) => ({
    ...item, label: item.digit, row: 0, uncertain: item.digit === "?",
  }));
}

function captureScan() {
  const width = Math.round(camera.videoWidth * SCAN.width);
  const height = Math.round(camera.videoHeight * SCAN.height);
  const scale = Math.min(1, 960 / Math.max(width, height));
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  context.drawImage(
    camera, Math.round(camera.videoWidth * SCAN.x), Math.round(camera.videoHeight * SCAN.y),
    width, height, 0, 0, canvas.width, canvas.height,
  );
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.88));
}

async function scanFrame(signal) {
  if (signal.aborted) return;
  const started = performance.now();
  const revision = scanRevision;
  const target = scanMode;
  let delay = target === "digits" ? FRAME_INTERVAL : TEXT_FRAME_INTERVAL;
  try {
    if (!document.hidden && camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      const blob = await captureScan();
      if (signal.aborted || revision !== scanRevision) return;
      if (!blob) throw new Error("The browser could not capture a camera frame.");
      const items = await recognize(blob, blob.type, signal, target);
      if (signal.aborted || revision !== scanRevision) return;
      showNotice("");
      showResult(stabilize(items), SCAN);
    }
  } catch (error) {
    if (signal.aborted || revision !== scanRevision) return;
    if (error.status !== 429) {
      clearResult();
      showNotice(error.message);
      delay = 850;
    }
  } finally {
    if (!signal.aborted) {
      timer = setTimeout(() => scanFrame(signal), Math.max(0, delay - (performance.now() - started)));
    }
  }
}

function cameraError(error) {
  switch (error.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission was denied. Allow camera access for localhost in your browser and check Windows camera privacy settings.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera is available. Connect a webcam or upload an image.";
    case "NotReadableError":
    case "AbortError":
      return "The camera could not start. Close other apps using it, then try again.";
    default:
      return "The camera could not start. Check the connection or upload an image.";
  }
}

async function startCamera() {
  const signal = resetSource("starting");
  if (!navigator.mediaDevices?.getUserMedia) {
    resetSource();
    showNotice("Open DigitLens at the localhost address shown in Ubuntu and use a current browser for camera access.");
    return;
  }
  try {
    const granted = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: "environment" } },
      audio: false,
    });
    if (signal.aborted) {
      for (const track of granted.getTracks()) track.stop();
      return;
    }
    stream = granted;
    for (const track of stream.getVideoTracks()) {
      track.addEventListener("ended", () => {
        if (signal.aborted) return;
        resetSource();
        showNotice("The camera disconnected. Reconnect it or upload an image.");
      });
    }
    camera.srcObject = stream;
    await camera.play();
    if (signal.aborted) return;
    setMode("live");
    scanFrame(signal);
  } catch (error) {
    if (signal.aborted) return;
    resetSource();
    showNotice(cameraError(error));
  }
}

async function uploadImage(file) {
  const signal = resetSource("loading");
  currentFile = file;
  const target = scanMode;
  try {
    const extension = file.name.split(".").pop().toLowerCase();
    const type = Object.values(IMAGE_TYPES).includes(file.type) ? file.type : IMAGE_TYPES[extension];
    if (!type) throw new Error("Choose a JPEG, PNG, or WebP image.");
    if (file.size > MAX_UPLOAD_BYTES) throw new Error("Choose an image smaller than 12 MB.");
    imageUrl = URL.createObjectURL(file);
    still.src = imageUrl;
    await still.decode().catch(() => { throw new Error("Could not open that image."); });
    if (signal.aborted) return;
    setMode("image");
    showNotice("Reading image…");
    let items;
    const retryUntil = performance.now() + 5000;
    while (!signal.aborted) {
      try {
        items = await recognize(file, type, signal, target);
        break;
      } catch (error) {
        if (error.status !== 429 || performance.now() >= retryUntil) throw error;
        await new Promise((resolve) => setTimeout(resolve, TEXT_FRAME_INTERVAL));
      }
    }
    if (signal.aborted) return;
    showNotice("");
    showResult(items);
  } catch (error) {
    if (signal.aborted) return;
    if (mode === "loading") resetSource();
    showNotice(error.message);
  }
}

scanModes.addEventListener("change", (event) => {
  scanMode = event.target.value;
  scanRevision++;
  const english = scanMode !== "digits";
  sequence.classList.toggle("text-result", english);
  sequence.setAttribute("aria-label", english ? "Recognized English text" : "Recognized sequence");
  layoutTip.textContent = english
    ? "Fit a short block of horizontal text inside the scan box."
    : "Place one horizontal row inside the scan box.";
  spacingTip.textContent = english
    ? "Leave room around each line and hold the page steady."
    : "Leave a gap between digits and hold the paper steady.";
  clearResult();
  showNotice("");
  if (currentFile) uploadImage(currentFile);
});
start.addEventListener("click", startCamera);
stop.addEventListener("click", () => resetSource());
upload.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileInput.value = "";
  if (file) uploadImage(file);
});
new ResizeObserver(layoutSurface).observe(stage);
camera.addEventListener("resize", layoutSurface);
window.addEventListener("pagehide", () => resetSource());
