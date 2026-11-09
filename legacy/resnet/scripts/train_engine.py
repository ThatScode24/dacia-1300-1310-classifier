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

# conf
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 30
EARLY_STOP_PATIENCE = 15   # opreste daca val acc nu se imbunatateste atatea epoci la rand
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dir 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (parent of /scripts)
CARB_DIR = os.path.join(BASE_DIR, 'train', 'motor', 'carburatie')
INJ_DIR = os.path.join(BASE_DIR, 'train', 'motor', 'injectie')
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

train_transforms = transforms.Compose([
    PadToSquare(),
    transforms.RandomResizedCrop(224, scale=(0.3, 1.0)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20), 
    transforms.RandomGrayscale(p=0.3), 
    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    PadToSquare(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# dataset
class EngineDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.samples = file_list
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # o imagine corupta opreste antrenarea in loc sa devina un tensor negru
        # cu eticheta reala (ar invata modelul o asociere falsa, silentios)
        image = Image.open(path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label

# setup
def get_model():
    print("Incarcare ResNet18")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model.to(DEVICE)

# train
def train():
    seed_all()  # split si antrenare reproductibile
    # date
    carb_files = glob.glob(os.path.join(CARB_DIR, "*.*"))
    inj_files = glob.glob(os.path.join(INJ_DIR, "*.*"))
    all_data = [(p, 0) for p in carb_files] + [(p, 1) for p in inj_files]
    
    if len(all_data) == 0: return

    # 80/20 grupat pe vehicul (aceeasi masina nu apare si in train si in val)
    train_files, val_files = grouped_split(all_data, 0.2)
    
    print(f"Data Split: {len(train_files)} Antrenare | {len(val_files)} Validare")

    train_loader = DataLoader(EngineDataset(train_files, transform=train_transforms), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EngineDataset(val_files, transform=val_transforms), batch_size=BATCH_SIZE, shuffle=False)

    model = get_model()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=8, verbose=True
    )
    # 46 carburatie vs 66 injectie: ponderam clasele + label smoothing
    weights = class_weights([len(carb_files), len(inj_files)], DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    best_acc = 0.0
    epochs_no_improve = 0

    loss_history = []
    acc_history = []

    print("Start antrenare")

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
        
        # pt grafci
        loss_history.append(avg_loss)
        acc_history.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.1f}%")

        scheduler.step(val_acc)

        # salvam doar la imbunatatire reala si numaram epocile fara progres
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "resnet_engine.pth"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping la epoca {epoch+1}: {EARLY_STOP_PATIENCE} "
                  f"epoci fara imbunatatire (best {best_acc:.1f}%)")
            break


    epochs_done = len(loss_history)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs_done+1), loss_history, label='Training Loss', color='red')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs_done+1), acc_history, label='Validation Accuracy', color='blue')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHICS_DIR, 'graph_motor.png'))

if __name__ == "__main__":
    train()