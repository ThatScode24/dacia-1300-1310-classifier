"""Construieste datele routerului din folderele specialistilor.

Copiaza imaginile din train/model/* -> train/router/fata,
train/caroserie/* -> train/router/spate si train/motor/* -> train/router/motor,
prefixand numele cu folderul sursa (ex: model_cn3__car0012__05.jpg).
Fisierele deja copiate sunt sarite, deci scriptul poate fi rulat repetat.

Utilizare:
    python scripts/populate_router.py
"""
import os
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from dacia.data import is_image

SRC_ROOT = os.path.join(BASE_DIR, "train")
DST_ROOT = os.path.join(BASE_DIR, "train", "router")

# clasa router -> folderele sursa (cn1 / papuc nu au inca date; sunt sarite)
MAPPING = {
    "fata":   ["model/1300", "model/cn1", "model/cn3", "model/cn4"],
    "spate":  ["caroserie/berlina", "caroserie/break", "caroserie/papuc"],
    "motor":  ["motor/carburatie", "motor/injectie"],
}

# clase planificate, populate manual (deocamdata goale -> ignorate la antrenare)
MANUAL_CLASSES = ["bord", "altceva"]


def count_images(folder):
    return sum(1 for f in os.listdir(folder) if is_image(f))


def populate():
    total_copied = 0
    for dst_class, src_dirs in MAPPING.items():
        dst = os.path.join(DST_ROOT, dst_class)
        os.makedirs(dst, exist_ok=True)
        existing = set(os.listdir(dst))

        copied = 0
        for sub in src_dirs:
            src = os.path.join(SRC_ROOT, sub)
            if not os.path.isdir(src):
                continue
            prefix = sub.replace("/", "_")
            for fname in os.listdir(src):
                if not is_image(fname):
                    continue
                new_name = f"{prefix}__{fname}"
                if new_name in existing:
                    continue
                shutil.copy2(os.path.join(src, fname), os.path.join(dst, new_name))
                copied += 1
        total_copied += copied
        print(f"{dst_class:10s} +{copied:4d} copiate | {count_images(dst)} total")

    for name in MANUAL_CLASSES:
        d = os.path.join(DST_ROOT, name)
        os.makedirs(d, exist_ok=True)
        print(f"{name:10s} (manual)    | {count_images(d)} total")

    print(f"\nTotal copiate sesiunea asta: {total_copied}")


if __name__ == "__main__":
    populate()
