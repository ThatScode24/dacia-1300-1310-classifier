import os
import glob
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
# La 384x384 fiecare sample foloseste ~3x memorie fata de 224x224,
# asa ca scadem batch_size ca sa incapa pe GPU-uri mai modeste.
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 60      # mai mult, atinge un platou si e inutil
IMG_SIZE = 384   # rezolutie marita - diferenta berlina/break e in detalii fine
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# direc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of /scripts)
BERLINA_DIR = os.path.join(BASE_DIR, 'train', 'caroserie', 'berlina')
BREAK_DIR =   os.path.join(BASE_DIR, 'train', 'caroserie', 'break')
MODELS_DIR =  os.path.join(BASE_DIR, 'models')
GRAPHICS_DIR = os.path.join(BASE_DIR, 'graphics')
os.makedirs(GRAPHICS_DIR, exist_ok=True)

# preprocesare
class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        padding = ((max_wh - w) // 2, (max_wh - h) // 2, max_wh - w - (max_wh - w) // 2, max_wh - h - (max_wh - h) // 2)
        return F.pad(img, padding, 0, 'constant')

train_transforms = transforms.Compose([
    PadToSquare(),
    # incercma sa ne concentram pe forme si nu pe imagini in intregime 
    # (break si berlina au forme fundamental diferite in spate)

    # scale aproape de 1.0 ca sa nu pierdem partea din spate
    # (acolo e diferenta intre berlina si break)
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomGrayscale(p=0.2),
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

# aici dataset
class CarBodyDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.samples = file_list
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # fara try/except aici: o imagine corupta trebuie sa opreasca antrenarea,
        # nu sa fie inlocuita silentios cu un tensor negru etichetat gresit
        image = Image.open(path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label

def get_model():
    print("Incarcare resnet18")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model.to(DEVICE)

# antrenare
def train():
    seed_all()  # split si antrenare reproductibile
    berlina_files = glob.glob(os.path.join(BERLINA_DIR, "*.*"))
    break_files = glob.glob(os.path.join(BREAK_DIR, "*.*"))
    
    # Verificare rapida
    print(f"Date: {len(berlina_files)} Berlina vs {len(break_files)} Break")

    # etichete: 0 berlina, 1 break
    all_data = [(p, 0) for p in berlina_files] + [(p, 1) for p in break_files]
    # 80/20 grupat pe vehicul (aceeasi masina nu apare si in train si in val)
    train_files, val_files = grouped_split(all_data, 0.2)
    
    print(f"Split: {len(train_files)} Antrenare | {len(val_files)} Validare")

    train_loader = DataLoader(CarBodyDataset(train_files, transform=train_transforms), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CarBodyDataset(val_files, transform=val_transforms), batch_size=BATCH_SIZE, shuffle=False)

    model = get_model()
    
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=8, verbose=True)
    
    # ponderi de clasa (usor dezechilibrat) + label smoothing pentru date putine
    weights = class_weights([len(berlina_files), len(break_files)], DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_acc = 0.0    # avem nevoie pentru a genera graficul
    loss_history = []
    acc_history = []

    print("Incepe antrenarea")

    for epoch in range(EPOCHS):
        model.train()
        set_bn_eval(model)  # backbone-ul e inghetat: blocam si running stats BN
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

        if val_acc >= best_acc:  # vrem sa salvam sistematic modelul cu cel mai bun value accuracy
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "resnet_body.pth"))

    # generare grafic 
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
    plt.savefig(os.path.join(GRAPHICS_DIR, 'graph_caroserie.png'))

if __name__ == "__main__":
    train()