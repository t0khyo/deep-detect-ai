import os
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch

from app.model import load_model
from app.preprocess import preprocess_img


# Create FastAPI instance
app = FastAPI()

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./model/logistic_model_triangular_m09_ashoj3.pth"
model_rms = load_model(model_path, device)

@app.get("/hello")
async def hello():
    return {"message": "I hate Python!"}

@app.post("/predict")
async def predict(genuineSignature: UploadFile = File(...), signatureToVerify: UploadFile = File(...)):
    try:
        # Validate images existence
        if not genuineSignature.file.readable or not signatureToVerify.file:
            raise HTTPException(status_code=400, detail="Both genuineSignature and signatureToVerify images must be provided.")

        # Validate file types
        if not genuineSignature.content_type.startswith('image') or not signatureToVerify.content_type.startswith('image'):
            raise HTTPException(status_code=400, detail="Both files must be valid image files.")
        
        # Open and preprocess the images
        img1 = Image.open(BytesIO(await genuineSignature.read()))
        img2 = Image.open(BytesIO(await signatureToVerify.read()))

        img1 = preprocess_img(img1)  # Preprocess image 1
        img2 = preprocess_img(img2)  # Preprocess image 2

        # Convert to tensor and move to device
        transform = transforms.Compose([transforms.ToTensor()])
        input1 = transform(img1).unsqueeze(0).to(device)
        input2 = transform(img2).unsqueeze(0).to(device)

        # Model inference
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

        return JSONResponse({
            "similarityPercentage": f"{similarity_score:.2f}",
            "probabilityPercentage": f"{probability_percentage:.2f}",
            "signatureWasNotForged": similarity_score > 80 and probability_percentage > 80
        })

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise {"error": str(e)}
