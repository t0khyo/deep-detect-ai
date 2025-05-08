import os
import logging
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
import torch
from app.config.signature_model_config import load_model
from app.service.signature_preprocess import preprocess_img

logger = logging.getLogger(__name__)
signature_bp = Blueprint('signature', __name__)

model_rms = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_signature_model():
    global model_rms
    model_path_1 = "./model/resnet50_triangular_m1_ashoj10"
    model_path_2 = "./model/logistic_model_triangular_m09_ashoj3_v2.1.pth"
    logger.info(f"Loading models on device: {device}")
    model_rms = load_model(model_path_1, model_path_2, device)
    logger.info("Models loaded successfully.")


@signature_bp.route('/hello', methods=['GET'])
def hello():
    logger.info("Received request at /hello")
    return jsonify({"message": "I hate Python!"})

@signature_bp.route('/', methods=['GET'])
def root():
    logger.info("Received request at /")
    return jsonify({"message": "Hello, Deep Detect AI!"})

@signature_bp.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received request at /predict")

        if 'genuineSignature' not in request.files or 'signatureToVerify' not in request.files:
            logger.warning("Missing images in request.")
            return jsonify({"error": "Both genuineSignature and signatureToVerify images must be provided."}), 400

        genuine_signature = request.files['genuineSignature']
        signature_to_verify = request.files['signatureToVerify']

        if not genuine_signature.content_type.startswith('image') or not signature_to_verify.content_type.startswith('image'):
            logger.warning("Invalid file types provided.")
            return jsonify({"error": "Both files must be valid image files."}), 400

        logger.info("Preprocessing images...")
        img1 = Image.open(genuine_signature.stream)
        img2 = Image.open(signature_to_verify.stream)

        img1 = preprocess_img(img1)
        img2 = preprocess_img(img2)

        transform = transforms.Compose([transforms.ToTensor()])
        input1 = transform(img1).unsqueeze(0).to(device)
        input2 = transform(img2).unsqueeze(0).to(device)

        logger.info("Performing model inference...")
        model_rms.eval()
        with torch.no_grad():
            prediction = model_rms(input1, input2)
            pred1 = model_rms.forward_once(input1)
            pred2 = model_rms.forward_once(input2)
            diff = torch.pairwise_distance(pred1, pred2)
            similarity_score = 1 / (1 + diff)
            probability_percentage = prediction.item()

            similarity_score = similarity_score.item() * 100
            probability_percentage = probability_percentage * 100

        result = {
            "similarityPercentage": f"{similarity_score:.2f}",
            "probabilityPercentage": f"{probability_percentage:.2f}",
            "signatureWasNotForged": similarity_score > 92 and probability_percentage > 92
        }

        logger.info(f"Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /predict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
