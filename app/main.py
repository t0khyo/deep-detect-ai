import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
import torch
from io import BytesIO

from app.model import load_model
from app.preprocess import preprocess_img

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create Flask instance
app = Flask(__name__)

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./model/logistic_model_triangular_m09_ashoj3.pth"
logger.info(f"Loading model from {model_path} on device {device}")
model_rms = load_model(model_path, device)
logger.info("Model loaded successfully.")

@app.route('/hello', methods=['GET'])
def hello():
    logger.info("Received request at /hello")
    return jsonify({"message": "I hate Python!"})

@app.route('/', methods=['GET'])
def root():
    logger.info("Received request at /")
    return jsonify({"message": "Hello, Deep Detect AI!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received request at /predict")

        # Validate images existence
        if 'genuineSignature' not in request.files or 'signatureToVerify' not in request.files:
            logger.warning("Missing images in request.")
            return jsonify({"error": "Both genuineSignature and signatureToVerify images must be provided."}), 400

        genuine_signature = request.files['genuineSignature']
        signature_to_verify = request.files['signatureToVerify']

        # Validate file types
        if not genuine_signature.content_type.startswith('image') or not signature_to_verify.content_type.startswith('image'):
            logger.warning("Invalid file types provided.")
            return jsonify({"error": "Both files must be valid image files."}), 400

        logger.info("Preprocessing images...")
        img1 = Image.open(genuine_signature.stream)
        img2 = Image.open(signature_to_verify.stream)

        img1 = preprocess_img(img1)
        img2 = preprocess_img(img2)

        # Convert to tensor and move to device
        transform = transforms.Compose([transforms.ToTensor()])
        input1 = transform(img1).unsqueeze(0).to(device)
        input2 = transform(img2).unsqueeze(0).to(device)

        # Model inference
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
            "signatureWasNotForged": similarity_score > 80 and probability_percentage > 80
        }

        logger.info(f"Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /predict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=8000)
