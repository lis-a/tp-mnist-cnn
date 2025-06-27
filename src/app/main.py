from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Autoriser les requêtes du front (utile pour Streamlit plus tard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Redéfinir le modèle CNN
class ConvNet(nn.Module):
    def __init__(self, input_size, n_kernels, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(n_kernels, n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(n_kernels * 4 * 4, 50),
            nn.Linear(50, output_size),
        )

    def forward(self, x):
        return self.net(x)


# 🔁 Charger le modèle
input_size = 28 * 28
n_kernels = 6
output_size = 10
model = ConvNet(input_size, n_kernels, output_size)
model.load_state_dict(torch.load("model/mnist-0.0.1.pt", map_location="cpu"))
model.eval()

# 🔄 Transformation à appliquer à l’image reçue
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 📥 Endpoint d’inférence
@app.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # [1, 1, 28, 28]

    with torch.no_grad():
        logits = model(img_tensor)
        prediction = torch.argmax(logits, dim=1).item()

    return {"prediction": prediction}
