import torch
import torch.nn as nn
from torchvision import transforms, models
import torchvision.transforms.functional as F
from PIL import Image
import os
import sys
import json
import torch.nn.functional as F_torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of /tests)
MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet_model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "resnet_model_classes.json")
IMAGE_PATH = os.path.join(BASE_DIR, "cn4_test.jpg")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384  # trebuie sa fie identic cu IMG_SIZE din train_model.py

# Trebuie sa fie identic cu val_transforms din train_model.py,
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


def load_classes():
    try:
        with open(CLASSES_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"eroare citire {CLASSES_PATH}: {e}")
        sys.exit()

def load_model(num_classes):
    print("Incarcare model")
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Am incarcat resnet_model.pth")
    except Exception as e:
        print(f"eroare: {e}")
        sys.exit()

    model.to(DEVICE)
    model.eval()
    return model

def predict_model(image_path):
    if not os.path.exists(image_path):
        print(f"Image introuvable {image_path}")
        return

    classes = load_classes()
    model = load_model(len(classes))

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"Eroare procesaere: {e}")
        return

    print(f"Analiza imagine: {image_path}")

    with torch.no_grad():
        output = model(input_tensor)
        probs = F_torch.softmax(output, dim=1)[0]

    scores = [(classes[i], probs[i].item() * 100) for i in range(len(classes))]
    scores.sort(key=lambda x: x[1], reverse=True)

    top_name, top_score = scores[0]
    print(f"DACIA {top_name.upper()} | {top_score:.2f}%")
    for name, sc in scores[1:]:
        print(f"  {name}: {sc:.2f}%")

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    predict_model(target_image)
