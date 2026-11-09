import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torchvision.transforms.functional as F
from PIL import Image
import glob
import random
import matplotlib.pyplot as plt
from common import seed_all, set_bn_eval, class_weights, grouped_split


BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 30
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# direc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of /scripts)
DATA_DIR = os.path.join(BASE_DIR, 'train', 'router')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
GRAPHICS_DIR = os.path.join(BASE_DIR, 'graphics')
os.makedirs(GRAPHICS_DIR, exist_ok=True)

# preprocesare
class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        padding = ((max_wh - w) // 2, (max_wh - h) // 2, max_wh - w - (max_wh - w) // 2, max_wh - h - (max_wh - h) // 2)
        return F.pad(img, padding, 0, 'constant')

# Crop mai conservator decat la caroserie - aici contextul global al pozei
# (cadru bord vs cadru motor) e ce ne ajuta sa decidem, nu detaliile fine.
train_transforms = transforms.Compose([
    PadToSquare(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    PadToSquare(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Dataset stil train_caroserie.py - lista (path, label) cu transform aplicat la getitem.
# Avantaj fata de ImageFolder: putem sari peste folderele de clasa goale
# (deocamdata bord/altceva nu au date).
class RouterDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # o imagine corupta opreste antrenarea in loc sa devina un tensor negru
        # etichetat cu clasa reala (asociere falsa invatata silentios)
        image = Image.open(path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label

def discover_classes(root):
    classes = []
    samples_per_class = {}
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        files = [p for p in glob.glob(os.path.join(sub, "*")) if os.path.splitext(p)[1].lower() in IMG_EXTS]
        if not files:
            print(f"  (sar peste {name} - 0 imagini)")
            continue
        classes.append(name)
        samples_per_class[name] = files
    return classes, samples_per_class

def get_model(num_classes):
    print("Incarcare resnet18")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model.to(DEVICE)

def train():
    seed_all()  # inlocuieste random.seed(42) de mai jos: acopera si torch/cuda
    classes, samples_per_class = discover_classes(DATA_DIR)
    print(f"Clase router ({len(classes)}): {classes}")
    for c in classes:
        print(f"  {c}: {len(samples_per_class[c])}")

    # salvam ordinea claselor pentru inferenta
    with open(os.path.join(MODELS_DIR, "resnet_router_classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False)

    # construim lista (path, label) si amestecam
    all_data = []
    for idx, c in enumerate(classes):
        all_data.extend([(p, idx) for p in samples_per_class[c]])
    # 80/20 stratificat pe clasa si grupat pe vehicul: copiile din router
    # mostenesc prefixul car<ID>__ din sursa, deci aceeasi masina nu apare
    # si in train si in val (altfel val acc urca fals la 100%)
    train_files, val_files = grouped_split(all_data, 0.2)

    train_loader = DataLoader(RouterDataset(train_files, transform=train_transforms), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RouterDataset(val_files, transform=val_transforms), batch_size=BATCH_SIZE, shuffle=False)

    print(f"Split: {len(train_files)} Antrenare | {len(val_files)} Validare")

    model = get_model(len(classes))

    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    # spate are ~2x mai multe imagini decat fata/motor: ponderam clasele
    weights = class_weights([len(samples_per_class[c]) for c in classes], DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_acc = 0.0
    loss_history = []
    acc_history = []

    print("Incepe antrenarea")

    for epoch in range(EPOCHS):
        model.train()
        set_bn_eval(model)  # backbone inghetat: blocam si running stats BN
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        avg_loss = train_loss / len(train_loader)

        loss_history.append(avg_loss)
        acc_history.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.1f}%")

        scheduler.step(val_acc)

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "resnet_router.pth"))

    # grafic
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS+1), loss_history, label='Loss', color='purple')
    plt.title('Training Loss')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS+1), acc_history, label='Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, 'graph_router.png'))

if __name__ == "__main__":
    train()
