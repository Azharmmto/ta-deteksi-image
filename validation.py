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
    # Mengecek apakah file kosong
    if file_storage is None or file_storage.filename == "":
        raise ValidationError("No image was uploaded.")

    # Cek Ekstensi File
    ext = _get_extension(file_storage.filename)
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS)).upper()
        raise ValidationError(f"File tidak valid. Extension harus '{ext}'. Allowed types: {allowed}.")

    # ── cek ukuran file
    file_bytes = file_storage.read()
    if len(file_bytes) == 0:
        raise ValidationError("Uploaded file is empty.")
    if len(file_bytes) > config.MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        max_mb = config.MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise ValidationError(f"File terlalu besar ({size_mb:.1f} MB). Maksimum yang diizinkan adalah {max_mb:.0f} MB.")

    # cek apakah file adalah gambar yang valid
    buffer = io.BytesIO(file_bytes) # gunakan buffer in-memory untuk memeriksa file
    try:
        with Image.open(buffer) as probe:
            probe.verify()
            # verify() mengecek header file.
            # jika user mengunggah file lain yang di-rename menjadi .jpg, verify() akan gagal.
    except Exception as exc:
        raise ValidationError("File tidak valid.") from exc

    # Membuka dan memuat gambar sepenuhnya
    buffer.seek(0) # Mengembalikan kursor pembacaan file ke titik awal (byte ke-0)
    try:
        image = Image.open(buffer)
        image.load()  # memastikan gambar tidak terpotong (corrupt)
        return image.convert("RGB") # Mengubah gambar ke format RGB (membuang Alpha/Transparansi jika ada)
    except Exception as exc:
        raise ValidationError("Failed to decode the uploaded image.") from exc
