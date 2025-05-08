import os
import logging
from flask import Flask, jsonify

from app.controller.signature_controller import signature_bp, init_signature_model

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)

    # Set upload folder config
    UPLOAD_FOLDER = "./uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # Register Blueprints with /api prefix
    app.register_blueprint(signature_bp, url_prefix='/api/signature')
    init_signature_model()

    # Global test route
    @app.route('/api/hello', methods=['GET'])
    def hello():
        logger.info("Received request at /api/hello")
        return jsonify({"message": "Hello from Deep Detect AI!"})

    return app

app = create_app()

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000)
