
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
    def __init__(self, model_path: str):
        # simpan path model dan menyiapkan variabel sesi
        self.model_path = model_path
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.load_error: str | None = None
        self._lock = threading.Lock() 
        self._load() # Memanggil fungsi load saat class dibuat

    def _load(self):
        # Memuat model ke memori
        with self._lock:
            try:
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1 # Membatasi penggunaan thread CPU agar server tidak hang
                self.session = ort.InferenceSession(
                    self.model_path,
                    sess_options=so,
                    providers=["CPUExecutionProvider"], # Menjalankan model murni dengan CPU
                )

                # Menyimpan nama input dan output layer dari model
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
                self.load_error or "Model ONNX tidak dimuat."
            )
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        logit = np.asarray(outputs[0]).reshape(-1)[0]
        return float(logit)


def preprocess_image(image: Image.Image) -> np.ndarray:
    # Mengubah resolusi gambar menjadi 224x224 dengan algoritma BICUBIC agar tidak blur
    resized = image.resize((config.IMG_SIZE, config.IMG_SIZE), _BICUBIC)

    # Mengubah gambar ke matriks angka dan menormalisasi warna (0-255 jadi 0.0-1.0)
    array = np.asarray(resized, dtype=np.float32) / 255.0  # HWC, [0, 1]

    # Normalisasi Z-Score (Mean & Standar Deviasi) sesuai konfigurasi   
    mean = np.asarray(config.MEAN, dtype=np.float32)
    std = np.asarray(config.STD, dtype=np.float32)

    # Normalisasi pixel berdasarkan mean dan std    
    array = (array - mean) / std

    # PyTorch/ONNX membaca format gambar sebagai Channel, Height, Width (CHW)
    array = array.transpose(2, 0, 1)  # Geser posisi matriks dari [224,224,3] menjadi [3,224,224]

    # Model AI selalu meminta data dalam bentuk "Batch" (kumpulan). 
    # Karena kita memproses 1 gambar, tambahkan dimensi semu di depan.
    array = np.expand_dims(array, axis=0)  # Menjadi [1, 3, 224, 224]

    # Ini mempercepat pembacaan data oleh ONNX Runtime.
    # # Menyusun ulang blok memori RAM agar berurutan (Contiguous)
    return np.ascontiguousarray(array, dtype=np.float32)


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid.
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


def postprocess_logit(logit: float) -> dict:
    prob_real = _sigmoid(logit) # Menghitung kemungkinan gambar Asli (Real)
    prob_ai = 1.0 - prob_real # Kemungkinan AI adalah kebalikannya (100% - Real)

    # kalau kemungkinan Real lebih dari Threshold (0.5 / 50%), maka labelnya 1 (Real)
    pred_label = 1 if prob_real >= config.THRESHOLD else 0
    class_name = config.LABEL_MAP[pred_label]

    # menentukan confidence score yang ditampilkan
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
