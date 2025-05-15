import os
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.service.audio_service import AudioService

logger = logging.getLogger(__name__)
audio_bp = Blueprint('audio', __name__)

audio_service = None

def init_audio_model(config):
    try:
        global audio_service
        model_path = os.path.join(config.MODEL_PATH, "xgb_model.pkl")
        scaler_path = os.path.join(config.MODEL_PATH, "scaler.pkl")
        logger.info(f"Loading audio model from: {model_path}")

        # Check if model files exist
        if not os.path.exists(model_path):
            logger.error(f"Audio model file not found at: {model_path}")
            return
        if not os.path.exists(scaler_path):
            logger.error(f"Audio scaler file not found at: {scaler_path}")
            return

        audio_service = AudioService(model_path, scaler_path)
        logger.info("Audio model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize audio model: {str(e)}", exc_info=True)

@audio_bp.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received audio prediction request")

        if audio_service is None:
            logger.error("Audio service not initialized")
            return jsonify({"error": "Audio service not initialized"}), 500

        if 'audio' not in request.files:
            logger.warning("No audio file in request")
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if not audio_file.filename:
            logger.warning("Empty audio filename")
            return jsonify({"error": "No audio file selected"}), 400

        if not audio_file.content_type.startswith('audio/'):
            logger.warning(f"Invalid file type: {audio_file.content_type}")
            return jsonify({"error": "File must be an audio file"}), 400

        # Save the uploaded audio
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(audio_path)

        try:
            # Process the audio
            result = audio_service.predict(audio_path)

            # Clean up the uploaded file
            os.remove(audio_path)

            return jsonify({"prediction": result})
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise e

    except Exception as e:
        logger.error(f"Error in audio prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500 