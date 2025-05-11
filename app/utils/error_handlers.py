from flask import jsonify
from werkzeug.exceptions import HTTPException
import logging

logger = logging.getLogger("deep_detect_ai")

class APIError(Exception):
    def __init__(self, message, status_code=400, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        rv['status'] = 'error'
        return rv

def register_error_handlers(app):
    @app.errorhandler(APIError)
    def handle_api_error(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        response = jsonify({
            'message': error.description,
            'status': 'error'
        })
        response.status_code = error.code
        return response

    @app.errorhandler(Exception)
    def handle_generic_error(error):
        logger.exception("Unhandled exception occurred")
        response = jsonify({
            'message': 'An unexpected error occurred',
            'status': 'error'
        })
        response.status_code = 500
        return response 