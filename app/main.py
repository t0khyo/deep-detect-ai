import os
from flask import Flask
from app.config.config import get_config
from app.config.logging_config import setup_logging
from app.utils.error_handlers import register_error_handlers
from app.controller.signature_controller import signature_bp, init_signature_model
from app.controller.video_controller import video_bp, init_video_model

# Setup logging
logger = setup_logging()

def create_app():
    # Load configuration
    config = get_config()
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Ensure upload directory exists
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register blueprints
    app.register_blueprint(signature_bp, url_prefix=f'{config.API_PREFIX}/signature')
    app.register_blueprint(video_bp, url_prefix=f'{config.API_PREFIX}/video')
    
    # Initialize models
    init_signature_model()
    init_video_model()
    
    # Health check endpoint
    @app.route(f'{config.API_PREFIX}/health', methods=['GET'])
    def health_check():
        logger.info("Health check endpoint called")
        return {"status": "healthy", "message": "Service is running"}
    
    return app

app = create_app()

if __name__ == "__main__":
    app = create_app()
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        debug=app.config["DEBUG"]
    )
