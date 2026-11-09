import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as F
from PIL import Image
import random
import matplotlib.pyplot as plt
from common import seed_all, set_bn_eval, class_weights, grouped_split

# config
# Dataset mic (~zeci de poze pe clasa), pastram batch mic si rezolutie mare
# pentru ca diferenta dintre variante e in detalii fine din fata masinii.
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 20
IMG_SIZE = 384
VAL_RATIO = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# direc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of /scripts)
DATA_DIR = os.path.join(BASE_DIR, 'train', 'model')
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

# Pozele din 1300 au ~850px latime, cele din cn3/cn4 ~300-500px. Dupa Resize
# la 384, cele mici ies blurate si "claritate = 1300" devine un shortcut corelat
# cu clasa. Aducem aleator toate imaginile la o latime din intervalul claselor
# mici, ca rezolutia sa nu mai fie informativa.
class RandomDownscale:
    def __init__(self, min_w=200, max_w=500):
        self.min_w, self.max_w = min_w, max_w
    def __call__(self, img):
        w, h = img.size
        target = random.randint(self.min_w, self.max_w)
        if w <= target:
            return img
        return img.resize((target, max(1, round(h * target / w))), Image.BILINEAR)

train_transforms = transforms.Compose([
    RandomDownscale(),
    PadToSquare(),
    # diferenta intre variante e in grila si farurile din fata
    # scale aproape de 1.0 ca sa nu pierdem detaliile importante
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# la validare aplicam acelasi downscale (determinist, la mijlocul intervalului)
# ca sa evaluam in aceleasi conditii in care va veni si o poza reala de telefon
val_transforms = transforms.Compose([
    RandomDownscale(350, 350),
    PadToSquare(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# dataset
class CarModelDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.samples = file_list
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # o imagine corupta opreste antrenarea in loc sa devina un tensor negru
        # etichetat cu clasa reala (asociere falsa invatata silentios)
        image = Image.open(path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label

def get_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # freeze tot, apoi dezghetam layer4 ca sa invete features specifice
    # variantelor (grila/faruri) - features generice ImageNet nu sunt
    # destul de discriminative pentru fine-grained
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model.to(DEVICE)

# autodetect clase din subdirectoare (sare peste cele goale)
def discover_classes(data_dir):
    classes = []
    samples = []
    for name in sorted(os.listdir(data_dir)):
        cls_dir = os.path.join(data_dir, name)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in glob.glob(os.path.join(cls_dir, "*.*"))
                 if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
        if not files:
            continue
        label = len(classes)
        classes.append(name)
        samples.extend((p, label) for p in files)
    return classes, samples

def train():
    seed_all()  # split si antrenare reproductibile
    classes, all_data = discover_classes(DATA_DIR)
    counts = {c: 0 for c in classes}
    for _, lbl in all_data:
        counts[classes[lbl]] += 1
    print(f"Clase ({len(classes)}): " + ", ".join(f"{c}={counts[c]}" for c in classes))

    # split pe vehicul (nu pe imagine): pozele aceleiasi masini nu ajung
    # si in train si in val - altfel val acc = 100% fals (vezi common.grouped_split)
    train_files, val_files = grouped_split(all_data, VAL_RATIO)
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    train_loader = DataLoader(CarModelDataset(train_files, transform=train_transforms), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CarModelDataset(val_files, transform=val_transforms), batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(len(classes))
    # LR mic pentru backbone (ne adaptam features pretrained, nu rescriem)
    # si LR mare pentru head (porneste de la zero)
    optimizer = optim.Adam([
        {"params": model.layer4.parameters(), "lr": LEARNING_RATE * 0.1},
        {"params": model.fc.parameters(),     "lr": LEARNING_RATE},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=8, verbose=False)
    weights = class_weights([counts[c] for c in classes], DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_acc = 0.0
    best_state = None
    loss_history = []
    acc_history = []

    for epoch in range(EPOCHS):
        model.train()
        # blocam BN in TOT modelul, inclusiv layer4 (care se antreneaza):
        # la batch 4, statisticile pe batch sunt prea zgomotoase, deci pastram
        # running stats ImageNet; convolutiile din layer4 se antreneaza normal
        set_bn_eval(model)
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        torch.save(best_state, os.path.join(MODELS_DIR, "resnet_model.pth"))
        with open(os.path.join(MODELS_DIR, "resnet_model_classes.json"), "w") as f:
            json.dump(classes, f)
        print(f"Salvat resnet_model.pth + resnet_model_classes.json (best val acc: {best_acc:.2f}%)")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS+1), loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS+1), acc_history)
    plt.axhline(best_acc, color='black', linestyle='--', label=f'Best {best_acc:.1f}%')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Acc (%)')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, 'graph_model.png'))

if __name__ == "__main__":
    train()
