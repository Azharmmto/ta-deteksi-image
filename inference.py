import logging
import threading

import numpy as np
import onnxruntime as ort
from PIL import Image

import config

logger = logging.getLogger(__name__)

# Pillow >= 9.1 exposes Image.Resampling.BICUBIC; older versions only
# have the flat Image.BICUBIC constant. Support both transparently.
_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested but the ONNX model failed to load."""


class ONNXEngine:
    """
    Thin wrapper around an onnxruntime.InferenceSession.

    The session is created exactly once (at process startup, see the
    module-level `engine` instance below) and reused for every request.
    Recreating a session per-request would repeatedly re-parse the model
    graph and re-allocate execution buffers, which is wasteful on CPU
    and is exactly what this class avoids.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.load_error: str | None = None
        # onnxruntime sessions are safe for concurrent Run() calls from
        # multiple threads, but we still guard session *creation* with
        # a lock in case of concurrent app reloads.
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        with self._lock:
            try:
                so = ort.SessionOptions()
                # Single-threaded intra-op parallelism keeps CPU/memory
                # usage predictable for a small, locally-run app serving
                # one request at a time. Raise this if you need higher
                # throughput under concurrent load.
                so.intra_op_num_threads = 1
                self.session = ort.InferenceSession(
                    self.model_path,
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                logger.info(
                    "ONNX model loaded from %s (input=%r, output=%r)",
                    self.model_path,
                    self.input_name,
                    self.output_name,
                )
            except Exception as exc:  # noqa: BLE001 - we want to capture *any* load failure
                self.session = None
                self.load_error = str(exc)
                logger.error("Failed to load ONNX model at %s: %s", self.model_path, exc)

    @property
    def is_ready(self) -> bool:
        return self.session is not None

    def run(self, input_array: np.ndarray) -> float:
        """Run a forward pass and return the single raw logit as a Python float."""
        if not self.is_ready:
            raise ModelNotLoadedError(
                self.load_error or "ONNX model is not loaded."
            )
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        logit = np.asarray(outputs[0]).reshape(-1)[0]
        return float(logit)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Reproduce the notebook's eval_transforms exactly:
      1. Resize to (IMG_SIZE, IMG_SIZE) with bicubic interpolation
      2. Scale pixel values to [0, 1]  (equivalent to transforms.ToTensor())
      3. Normalize with mean=std=0.5   (equivalent to transforms.Normalize)
      4. HWC -> CHW, then add a batch dimension -> [1, 3, H, W]

    `image` must already be a PIL Image in RGB mode.
    """
    resized = image.resize((config.IMG_SIZE, config.IMG_SIZE), _BICUBIC)

    array = np.asarray(resized, dtype=np.float32) / 255.0  # HWC, [0, 1]

    mean = np.asarray(config.MEAN, dtype=np.float32)
    std = np.asarray(config.STD, dtype=np.float32)
    array = (array - mean) / std

    array = array.transpose(2, 0, 1)  # HWC -> CHW
    array = np.expand_dims(array, axis=0)  # -> [1, 3, H, W]
    return np.ascontiguousarray(array, dtype=np.float32)


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid.
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


def postprocess_logit(logit: float) -> dict:
    """
    Mirrors predict_image() from the notebook (Section 19) exactly:
        prob = sigmoid(logit)                      # P(class == "Real")
        pred_label = 1 if prob >= threshold else 0
        class_name = LABEL_MAP[pred_label]
        confidence = prob if pred_label == 1 else (1 - prob)
    """
    prob_real = _sigmoid(logit)
    prob_ai = 1.0 - prob_real

    pred_label = 1 if prob_real >= config.THRESHOLD else 0
    class_name = config.LABEL_MAP[pred_label]
    confidence = prob_real if pred_label == 1 else prob_ai

    return {
        "prediction": class_name,
        "is_ai_generated": pred_label == 0,
        "confidence": round(confidence * 100, 2),
        "probability_real": round(prob_real * 100, 2),
        "probability_ai": round(prob_ai * 100, 2),
    }


def predict(image: Image.Image) -> dict:
    """Full pipeline: preprocess -> ONNX forward pass -> postprocess."""
    input_array = preprocess_image(image)
    logit = engine.run(input_array)
    return postprocess_logit(logit)


# ── Module-level singleton ───────────────────────────────────────────
# Created once, at import time (i.e. once per server process), and
# reused for every request. app.py imports `engine` to check readiness
# and this module's `predict()` to run inference -- it never touches
# onnxruntime directly.
engine = ONNXEngine(config.ONNX_MODEL_PATH)
