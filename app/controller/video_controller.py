import os
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.service.video_service import VideoService

logger = logging.getLogger(__name__)
video_bp = Blueprint('video', __name__)

video_service = None

def init_video_model():
    try:
        global video_service
        model_path = os.path.join(current_app.config['MODEL_PATH'], "model_93_acc.pt")
        logger.info(f"Loading video model from: {model_path}")

        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Video model file not found at: {model_path}")
            return

        video_service = VideoService(model_path)
        logger.info("Video model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize video model: {str(e)}", exc_info=True)

@video_bp.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received video prediction request")

        if video_service is None:
            logger.error("Video service not initialized")
            return jsonify({"error": "Video service not initialized"}), 500

        if 'video' not in request.files:
            logger.warning("No video file in request")
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']
        if not video_file.filename:
            logger.warning("Empty video filename")
            return jsonify({"error": "No video file selected"}), 400

        if not video_file.content_type.startswith('video/'):
            logger.warning(f"Invalid file type: {video_file.content_type}")
            return jsonify({"error": "File must be a video"}), 400

        # Save the uploaded video
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)

        try:
            # Process the video
            result = video_service.predict(video_path)

            # Clean up the uploaded file
            os.remove(video_path)

            return jsonify(result)
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(video_path):
                os.remove(video_path)
            raise e

    except Exception as e:
        logger.error(f"Error in video prediction: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
