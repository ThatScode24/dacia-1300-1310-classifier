"""Preprocesarea imaginilor si backbone-ul DINOv2, comune antrenarii si inferentei.

Pipeline: PadToSquare -> Resize 224x224 -> ToTensor -> normalizare ImageNet.
"""
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

IMG_SIZE = 224          # multiplu al dimensiunii de patch (14)
EMBED_DIM = 384         # dimensiunea embedding-ului DINOv2 ViT-S/14
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PadToSquare:
    """Completeaza imaginea cu negru pana la format patrat, ca redimensionarea
    sa nu distorsioneze proportiile (important la caroseriile break)."""

    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        padding = ((max_wh - w) // 2, (max_wh - h) // 2,
                   max_wh - w - (max_wh - w) // 2,
                   max_wh - h - (max_wh - h) // 2)
        return F.pad(img, padding, 0, 'constant')


def build_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_backbone(device):
    """DINOv2 ViT-S/14 inghetat (descarcat prin torch.hub la prima rulare)."""
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    return backbone.eval().to(device)
