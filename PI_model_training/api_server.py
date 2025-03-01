from fastapi.concurrency import asynccontextmanager
from fastapi.responses import StreamingResponse
from matplotlib import pyplot as plt
import torch
import os
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms, models
from typing import Dict

from prediction import predict
from gradcam import run_gradcam


MODEL_STATE_DICT_PATH = os.path.join(".", "best_model.pt")

model = None
transform = None
classes = ['FAKE', 'REAL']


@asynccontextmanager
async def load_model(app: FastAPI):
    global model, transform
    try:
        model = models.resnet50() 
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features,2)
        model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH, weights_only=True, map_location=torch.device('cpu'))) 
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    except Exception as e:
        raise RuntimeError(f"Error loading the model: {str(e)}")
    yield

app = FastAPI(lifespan=load_model)


@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)) -> Dict:
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Open the uploaded image
        image = Image.open(BytesIO(await file.read()))
        pred, confidence = predict(model, image, transform)

        # Make prediction
        return {
            "prediction": classes[pred].lower(),
            "confidence": float(confidence) * 100
        }

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


@app.post("/predict/gradcam")
async def predict_api(file: UploadFile = File(...)) -> Dict:
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # Open the uploaded image
        image = Image.open(BytesIO(await file.read()))
        grad_cam = run_gradcam(image)

        grad_cam_pil = Image.fromarray(grad_cam.astype('uint8'))

        buffer = io.BytesIO()
        grad_cam_pil.save(buffer, format="PNG")
        buffer.seek(0)

        # Grad cam from prediction
        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
