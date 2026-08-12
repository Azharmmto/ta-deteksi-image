"""
Upload validation for the /predict endpoint.

Validation is layered on purpose -- extension and MIME type are cheap
checks that reject obviously-wrong requests fast, while the Pillow
verify()/load() step is what actually guards against a corrupted file
or a malicious file with a spoofed extension pretending to be an image.
"""

import io

from PIL import Image
from werkzeug.datastructures import FileStorage

import config


class ValidationError(Exception):
    """Raised for any problem with an uploaded file. Carries an HTTP status code."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def validate_and_load_image(file_storage: FileStorage) -> Image.Image:
    """
    Validate an uploaded file and return a decoded, RGB PIL Image.

    Raises ValidationError (with a clean, user-facing message) if the
    file is missing, has a disallowed extension/MIME type, is too
    large, is empty, or is not a genuine, decodable image.
    """
    if file_storage is None or file_storage.filename == "":
        raise ValidationError("No image was uploaded.")

    # ── 1. Extension check ───────────────────────────────────────────
    ext = _get_extension(file_storage.filename)
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS)).upper()
        raise ValidationError(f"File tidak valid. Extension harus '{ext}'. Allowed types: {allowed}.")

    # ── 2. Browser-reported MIME type check ──────────────────────────
    # Cheap first line of defense. Not fully trustworthy on its own
    # (easily spoofed), which is why step 4 below re-verifies the
    # actual file contents regardless of what this header claims.
    if file_storage.mimetype not in config.ALLOWED_MIME_TYPES:
        raise ValidationError(f"File tidak valid. MIME type '{file_storage.mimetype}' tidak didukung.")

    # ── 3. Size check ─────────────────────────────────────────────────
    file_bytes = file_storage.read()
    if len(file_bytes) == 0:
        raise ValidationError("Uploaded file is empty.")
    if len(file_bytes) > config.MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        max_mb = config.MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise ValidationError(f"File terlalu besar ({size_mb:.1f} MB). Maksimum yang diizinkan adalah {max_mb:.0f} MB.")

    # ── 4. Genuine image-integrity check ─────────────────────────────
    # Image.verify() confirms the file is a structurally valid image
    # without fully decoding it. It also catches extension/MIME spoofing
    # (e.g. a renamed non-image file).
    buffer = io.BytesIO(file_bytes)
    try:
        with Image.open(buffer) as probe:
            probe.verify()
    except Exception as exc:
        raise ValidationError("The uploaded file is not a valid image or is corrupted.") from exc

    # verify() leaves the underlying file object unusable for further
    # operations, so we reopen from the same in-memory buffer.
    buffer.seek(0)
    try:
        image = Image.open(buffer)
        image.load()  # force full decode now, inside try/except, to catch truncated data
        return image.convert("RGB")
    except Exception as exc:
        raise ValidationError("Failed to decode the uploaded image.") from exc
