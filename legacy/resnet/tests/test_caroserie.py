import torch
import torch.nn as nn
from torchvision import transforms, models
import torchvision.transforms.functional as F
from PIL import Image
import os
import sys
import torch.nn.functional as F_torch 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of /tests)
MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet_body.pth")
IMAGE_PATH = os.path.join(BASE_DIR, "test26.jpg")  # poza de resr
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384  # trebuie sa fie identic cu IMG_SIZE din train_caroserie.py

# Trebuie sa fie identic cu val_transforms din train_caroserie.py,
# altfel inferenta vede o distributie diferita de cea de antrenare.
class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        padding = ((max_wh - w) // 2, (max_wh - h) // 2, max_wh - w - (max_wh - w) // 2, max_wh - h - (max_wh - h) // 2)
        return F.pad(img, padding, 0, 'constant')

transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model():
    print("Incarcare model")
    model = models.resnet18(weights=None) 
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2) 
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Am incarcat resnet_body.pth")
    except Exception as e:
        print(f"eroare: {e}")
        sys.exit()
        
    model.to(DEVICE)
    model.eval()
    return model

def predict_car_body(image_path):
    if not os.path.exists(image_path):
        print(f"Image introuvable {image_path}")
        return

    model = load_model()
    
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"Eroare procesaere: {e}")
        return

    print(f"Analiza imagine: {image_path}")
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = F_torch.softmax(output, dim=1)
        
        # 0 e dacie berlina, 1 e dacie break
        berlina_score = probs[0][0].item() * 100
        break_score = probs[0][1].item() * 100
    
    if berlina_score > break_score:
        print(f"DACIA BERLINA | {berlina_score:.2f}%")
    else:
        print(f"DACIA BREAK | {break_score:.2f}%")

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    predict_car_body(target_image)