import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model 
# Path to the exported ONNX model. See export_to_onnx.py for how to
# produce this file from the notebook's best_model.pth checkpoint.
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "models", "vit_tiny.onnx")

# Preprocessing (must match `eval_transforms` in the notebook,
#    Section 7 - "Preprocessing & Transformasi Citra")
IMG_SIZE = 224                     # CFG["img_size"]
MEAN = (0.5, 0.5, 0.5)             # CFG["mean"]  (ImageNet-21k / augreg stats)
STD = (0.5, 0.5, 0.5)              # CFG["std"]

# Postprocessing (Section 19 - "Pipeline Inferensi") 
# Label convention from the notebook is the OPPOSITE of the intuitive
# reading -- verified directly against load_split()/ArtifactDataset:
#   label 0 -> "fake" folder  -> AI-Generated
#   label 1 -> "real" folder  -> Real
THRESHOLD = 0.5                    # CFG["threshold"]
LABEL_MAP = {0: "AI-Generated", 1: "Real"}

#Upload validation
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
