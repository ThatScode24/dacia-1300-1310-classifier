"""Utilitare pentru setul de date: seed, ponderi de clasa, grupare pe vehicul."""
import os
import re
import random

import torch

from .preprocessing import IMG_EXTS

SEED = 42


def seed_all(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image(path):
    return os.path.splitext(path)[1].lower() in IMG_EXTS


def class_weights(counts, device):
    """Ponderi invers proportionale cu frecventa clasei."""
    total = sum(counts)
    k = len(counts)
    w = [total / (k * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(w, dtype=torch.float32, device=device)


# ID-ul vehiculului e codificat in numele fisierului: car<ID>__<nume>.jpg
_GROUP_RE = re.compile(r"car(\d+)__", re.IGNORECASE)


def group_of(path):
    m = _GROUP_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def group_report(paths):
    """Returneaza (nr. imagini cu ID, nr. imagini fara ID, nr. vehicule distincte)."""
    groups = [group_of(p) for p in paths]
    grouped = [g for g in groups if g is not None]
    return len(grouped), len(groups) - len(grouped), len(set(grouped))
