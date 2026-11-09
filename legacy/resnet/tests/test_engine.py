import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import torch.nn.functional as F
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of /tests)
MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet_engine.pth")
IMAGE_PATH = os.path.join(BASE_DIR, "blurat.jpg")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print("Incarcare model")
    model = models.resnet18(weights=None) 
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model incarcat")
    except Exception as e:
        print(f"Eroare{e}")
        exit()
        
    model.to(DEVICE)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def query(image_path):
    if not os.path.exists(image_path):
        return
    model = load_model()
    
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"eroare: {e}")
        return

    print(f"Analizam {image_path}")
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        # 0 carburator, 1 injectie
        carb_score = probs[0][0].item() * 100
        inj_score = probs[0][1].item() * 100

    if carb_score > inj_score:
        print(f"Carburator: {carb_score:.2f}%")
    else:
        print(f"Injection: {inj_score:.2f}%")


if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    query(target_image)