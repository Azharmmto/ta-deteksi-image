import logging

from flask import Flask, jsonify, render_template, request

# (module-level singleton). jika terjadi error atau sukses saat memuat model, 
# pesan log akan langsung dicetak. Dengan mengatur logging di awal, kita memastikan 
# pesan dari `inference.py` memiliki format waktu dan level yang seragam.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

import config
import inference
import validation

app = Flask(__name__)

# Validasi akurat per byte file-nya sendiri tetap akan ditangani oleh validation.py.
# Jika pengguna iseng mengunggah file sebesar 5 GB, Flask akan memblokirnya
# own size check (config.MAX_FILE_SIZE_BYTES) is a second, more precise
# check that runs after this coarse limit already passed.
app.config["MAX_CONTENT_LENGTH"] = config.MAX_FILE_SIZE_BYTES + (1 * 1024 * 1024)


@app.route("/")
def index():

    return render_template("index.html") # membeaca tmeplate html


@app.route("/health")
def health():
    # # Mengembalikan JSON berisi status.
    # jika model gagal dimuat, status HTTP adalah 503 (Service Unavailable).
    ready = inference.engine.is_ready
    return jsonify(
        status="ok" if ready else "model_unavailable",
        model_loaded=ready,
        detail=None if ready else inference.engine.load_error,
    ), (200 if ready else 503)


@app.route("/predict", methods=["POST"])
def predict():
    # jika model ONNX gagal diload saat server menyala, kita langsung menolak 
    # request agar tidak terjadi error lanjutan di dalam sistem.
    if not inference.engine.is_ready:
        return jsonify(
            success=False,
            error=(
                "The detection model is not available on the server. "
                f"Expected an ONNX model at '{config.ONNX_MODEL_PATH}'. "
                "See export_to_onnx.py for how to produce it."
            ),
        ), 503

    try:
        image = validation.validate_and_load_image(request.files.get("image")) # Mengambil file dari form HTML yang memiliki atribut name="image"
    except validation.ValidationError as exc:

        # Menangkap error kustom dari validation.py dan mengembalikannya ke pengguna
        return jsonify(success=False, error=exc.message), exc.status_code

    try:

        # Mengirim objek gambar yang sudah valid ke mesin AI
        result = inference.predict(image)
    except inference.ModelNotLoadedError as exc:

        # Menjaga kemungkinan model tiba-tiba tidak tersedia
        return jsonify(success=False, error=str(exc)), 503
    except Exception:
        logger.exception("inferensi gagal")
        return jsonify(success=False, error="inferensi gambar"), 500

    return jsonify(success=True, **result)


@app.errorhandler(413)
def handle_too_large(_exc):
    # jika request melebihi app.config["MAX_CONTENT_LENGTH"]
    max_mb = config.MAX_FILE_SIZE_BYTES / (1024 * 1024)
    return jsonify(success=False, error=f"Ukuran terlalu besar. Maksimum {max_mb:.0f} MB."), 413


@app.errorhandler(404)
def handle_not_found(_exc):
    # jika pengguna mengakses URL yang tidak ada (misal /predik atau /test)
    return jsonify(success=False, error="Not found."), 404


@app.errorhandler(500)
def handle_server_error(_exc):
    #jika kode server mengalami crash secara internal
    return jsonify(success=False, error="Internal server error."), 500


if __name__ == "__main__":
    if not inference.engine.is_ready:
        logger.warning(
            "Starting with NO model loaded (%s). The UI will load, but /predict "
            "will return 503 until vit_tiny.onnx is placed at %s.",
            inference.engine.load_error,
            config.ONNX_MODEL_PATH,
        )
    app.run(host="127.0.0.1", port=5000, debug=False)
