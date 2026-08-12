import logging

from flask import Flask, jsonify, render_template, request

# Configured BEFORE importing inference: inference.py loads the ONNX
# model at import time (module-level singleton) and logs the outcome
# immediately, so logging must already be set up for that message to
# be formatted consistently with everything else.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

import config
import inference
import validation

app = Flask(__name__)

# Defense in depth: reject oversized request bodies at the WSGI level,
# before Flask even finishes reading them into memory. validation.py's
# own size check (config.MAX_FILE_SIZE_BYTES) is a second, more precise
# check that runs after this coarse limit already passed.
app.config["MAX_CONTENT_LENGTH"] = config.MAX_FILE_SIZE_BYTES + (1 * 1024 * 1024)


@app.route("/")
def index():
    """Serve the existing upload UI unchanged."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Lightweight readiness probe -- mainly useful for confirming the ONNX model loaded."""
    ready = inference.engine.is_ready
    return jsonify(
        status="ok" if ready else "model_unavailable",
        model_loaded=ready,
        detail=None if ready else inference.engine.load_error,
    ), (200 if ready else 503)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a single image upload under the form field "image",
    validates it, runs ONNX inference, and returns:

        {
            "success": true,
            "prediction": "AI-Generated" | "Real",
            "is_ai_generated": bool,
            "confidence": float,        # confidence of the predicted class, in %
            "probability_real": float,  # P(Real), in %
            "probability_ai": float     # P(AI-Generated), in %
        }

    or, on failure:

        { "success": false, "error": "<message>" }
    """
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
        image = validation.validate_and_load_image(request.files.get("image"))
    except validation.ValidationError as exc:
        return jsonify(success=False, error=exc.message), exc.status_code

    try:
        result = inference.predict(image)
    except inference.ModelNotLoadedError as exc:
        return jsonify(success=False, error=str(exc)), 503
    except Exception:
        logger.exception("Inference failed for an uploaded image")
        return jsonify(success=False, error="Inference failed due to an internal server error."), 500

    return jsonify(success=True, **result)


@app.errorhandler(413)
def handle_too_large(_exc):
    max_mb = config.MAX_FILE_SIZE_BYTES / (1024 * 1024)
    return jsonify(success=False, error=f"File too large. Maximum allowed is {max_mb:.0f} MB."), 413


@app.errorhandler(404)
def handle_not_found(_exc):
    return jsonify(success=False, error="Not found."), 404


@app.errorhandler(500)
def handle_server_error(_exc):
    return jsonify(success=False, error="Internal server error."), 500


if __name__ == "__main__":
    if not inference.engine.is_ready:
        logger.warning(
            "Starting with NO model loaded (%s). The UI will load, but /predict "
            "will return 503 until vit_tiny.onnx is placed at %s.",
            inference.engine.load_error,
            config.ONNX_MODEL_PATH,
        )
    # Local-only app: bind to localhost, debug off by default.
    # Set debug=True temporarily during development if you need
    # auto-reload and tracebacks in the browser.
    app.run(host="127.0.0.1", port=5000, debug=False)
