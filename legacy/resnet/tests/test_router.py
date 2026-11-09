import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torchvision import transforms, models
import torchvision.transforms.functional as F
from PIL import Image
import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of /tests)
MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet_router.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "resnet_router_classes.json")
IMAGE_PATH = os.path.join(BASE_DIR, "mea.jpg")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256  # identic cu train_router.py

# Daca scor max < REJECT_THRESHOLD => poza nu intra la niciun specialist
# (router-ul nu e suficient de sigur). Valoare conservativa, se ajusteaza dupa validare.
REJECT_THRESHOLD = 0.55

# Mapare clasa router -> specialist(i) chemati. "altceva"
# nu are inca un specialist propriu; ramane doar eticheta deocamdata.
SPECIALIST_MAP = {
    "fata":     ["caroserie", "model"],   # model = 1300 / cn1 / cn3 / cn4
    "spate":    ["caroserie"],
    "motor":    ["motor"],
    "bord":     ["bord"],
    "altceva":  [],
}

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
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_model(num_classes):
    print("Incarcare model router")
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Am incarcat {MODEL_PATH}")
    except Exception as e:
        print(f"eroare: {e}")
        sys.exit()
    model.to(DEVICE)
    model.eval()
    return model

def route(image_path):
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

    # afisare distributie completa
    scores = [(classes[i], probs[i].item()) for i in range(len(classes))]
    scores.sort(key=lambda x: -x[1])
    for name, p in scores:
        print(f"  {name:10s} {p*100:6.2f}%")

    top_class, top_prob = scores[0]

    if top_prob < REJECT_THRESHOLD or top_class == "altceva":
        print(f"REJECT | top={top_class} ({top_prob*100:.2f}%) sub prag {REJECT_THRESHOLD*100:.0f}%")
        return

    specialisti = SPECIALIST_MAP.get(top_class, [])
    print(f"DISPATCH -> {top_class} ({top_prob*100:.2f}%) | specialisti: {specialisti}")

if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    route(target_image)
