"""Antreneaza clasificatoarele (strat liniar peste DINOv2 ViT-S/14 inghetat).

Pentru fiecare sarcina:
  1. extrage embeddings DINOv2 (imagine originala + oglindita orizontal);
  2. evalueaza prin validare incrucisata cu 5 folduri, grupata pe vehicul
     (StratifiedGroupKFold) -> results/cv_dinov2.txt + grafice in graphics/;
  3. antreneaza modelul final pe toate datele -> models/dinov2_<sarcina>.pth.

Utilizare:
    python scripts/train_dinov2.py                  # toate sarcinile
    python scripts/train_dinov2.py model motor      # doar unele
"""
import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from dacia.preprocessing import IMG_SIZE, build_transform, load_backbone
from dacia.data import SEED, seed_all, is_image, class_weights, group_of, group_report

TRAIN_ROOT = os.path.join(BASE_DIR, "train")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPHICS_DIR = os.path.join(BASE_DIR, "graphics")
REPORT_PATH = os.path.join(RESULTS_DIR, "cv_dinov2.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
PROBE_EPOCHS = 200
PROBE_LR = 1e-3
PROBE_WD = 1e-4

# fiecare sarcina = un subfolder din train/, cu cate un subfolder per clasa
TASKS = ["router", "caroserie", "model", "motor"]

PREPROC = build_transform(IMG_SIZE)


def discover(task):
    classes, paths, labels = [], [], []
    root = os.path.join(TRAIN_ROOT, task)
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        files = sorted(p for p in glob.glob(os.path.join(sub, "*")) if is_image(p))
        if not files:
            continue
        label = len(classes)
        classes.append(name)
        paths.extend(files)
        labels.extend([label] * len(files))
    return classes, paths, np.array(labels)


@torch.no_grad()
def extract_features(backbone, paths):
    feats_o, feats_f = [], []
    batch, batch_f = [], []

    def flush():
        if not batch:
            return
        xo = torch.stack(batch).to(DEVICE)
        xf = torch.stack(batch_f).to(DEVICE)
        fo = backbone(xo)
        ff = backbone(xf)
        feats_o.append(torch.nn.functional.normalize(fo, dim=1).cpu())
        feats_f.append(torch.nn.functional.normalize(ff, dim=1).cpu())
        batch.clear(); batch_f.clear()

    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        x = PREPROC(img)
        batch.append(x)
        batch_f.append(torch.flip(x, dims=[2]))
        if len(batch) == 32:
            flush()
        if (i + 1) % 128 == 0:
            print(f"    {i+1}/{len(paths)} imagini")
    flush()
    return torch.cat(feats_o).numpy(), torch.cat(feats_f).numpy()


def train_probe(X_train, y_train, n_classes, counts, val=None):
    Xt = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    probe = nn.Linear(Xt.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    crit = nn.CrossEntropyLoss(weight=class_weights(counts, DEVICE),
                               label_smoothing=0.1)
    history = {"loss": [], "val_acc": []}
    for _ in range(PROBE_EPOCHS):
        probe.train()
        opt.zero_grad()
        loss = crit(probe(Xt), yt)
        loss.backward()
        opt.step()
        history["loss"].append(float(loss.item()))
        if val is not None:
            probe.eval()
            pred = predict_probe(probe, val[0], val[1])
            history["val_acc"].append(float((pred == val[2]).mean()))
    probe.eval()
    return probe, history


@torch.no_grad()
def predict_probe(probe, X_orig, X_flip):
    lo = probe(torch.tensor(X_orig, dtype=torch.float32, device=DEVICE))
    lf = probe(torch.tensor(X_flip, dtype=torch.float32, device=DEVICE))
    return ((lo + lf) / 2).argmax(dim=1).cpu().numpy()


def counts_of(y, n_classes):
    return [int((y == c).sum()) for c in range(n_classes)]


def plot_task(task, classes, histories, accs, cm):
    epochs = range(1, PROBE_EPOCHS + 1)
    mean_acc = float(np.mean(accs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, h in enumerate(histories, 1):
        ax1.plot(epochs, h["loss"], label=f"Fold {i}")
        ax2.plot(epochs, [a * 100 for a in h["val_acc"]], label=f"Fold {i}")
    ax1.set_title(f"{task}: loss de antrenare (strat liniar)")
    ax1.set_xlabel("Epoca"); ax1.set_ylabel("Cross-entropy"); ax1.grid(True, alpha=.3); ax1.legend()
    ax2.set_title(f"{task}: acuratete de validare (vehicule nevazute)")
    ax2.set_xlabel("Epoca"); ax2.set_ylabel("Acuratete (%)"); ax2.set_ylim(0, 102)
    ax2.axhline(mean_acc * 100, color="black", linestyle="--", label=f"Media finala {mean_acc*100:.1f}%")
    ax2.grid(True, alpha=.3); ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHICS_DIR, f"dinov2_{task}_training.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(1, len(accs) + 1)
    bars = ax.bar(xs, [a * 100 for a in accs], color="#4a7ebb")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{a*100:.1f}", ha="center", fontsize=9)
    ax.axhline(mean_acc * 100, color="black", linestyle="--",
               label=f"Media {mean_acc*100:.1f}% ± {np.std(accs)*100:.1f}%")
    ax.set_xticks(xs); ax.set_xticklabels([f"Fold {i}" for i in xs])
    ax.set_ylim(0, 108); ax.set_ylabel("Acuratete (%)")
    ax.set_title(f"{task}: validare incrucisata grupata pe vehicul ({N_FOLDS} folduri)")
    ax.grid(True, axis="y", alpha=.3); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHICS_DIR, f"dinov2_{task}_folds.png"), dpi=150)
    plt.close(fig)

    n = len(classes)
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    pct = cm / row_sum
    fig, ax = plt.subplots(figsize=(1.6 * n + 3, 1.4 * n + 2.5))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]*100:.0f}%", ha="center", va="center",
                    color="white" if pct[i, j] > 0.5 else "black", fontsize=10)
    ax.set_xticks(range(n)); ax.set_xticklabels(classes)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes)
    ax.set_xlabel("Clasa prezisa"); ax.set_ylabel("Clasa reala")
    ax.set_title(f"{task}: matrice de confuzie agregata ({N_FOLDS} folduri)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Procent din clasa reala")
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHICS_DIR, f"dinov2_{task}_confuzie.png"), dpi=150)
    plt.close(fig)


def run_task(task, backbone, report_lines):
    print(f"\n=== Sarcina: {task} ===")
    classes, paths, y = discover(task)
    n_classes = len(classes)
    print(f"  Clase: {classes} | {len(paths)} imagini")

    print("  Extragere embeddings DINOv2...")
    Xo, Xf = extract_features(backbone, paths)

    n_grouped, n_ungrouped, n_vehicles = group_report(paths)
    use_groups = n_ungrouped == 0 and n_vehicles >= N_FOLDS
    if use_groups:
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        groups = np.array([group_of(p) for p in paths])
        split_iter = splitter.split(Xo, y, groups)
        split_note = f"StratifiedGroupKFold pe {n_vehicles} vehicule"
    else:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        split_iter = splitter.split(Xo, y)
        split_note = (f"StratifiedKFold LA NIVEL DE IMAGINE "
                      f"({n_grouped}/{len(paths)} imagini au ID de vehicul; "
                      f"e nevoie de toate + minim {N_FOLDS} vehicule). "
                      f"ATENTIE: acuratetea e umflata de scurgerea de date "
                      f"acelasi-vehicul-in-train-si-validare.")
    print(f"  Split: {split_note}")

    accs = []
    histories = []
    cm_total = np.zeros((n_classes, n_classes), dtype=int)
    for fold, (tr, va) in enumerate(split_iter, 1):
        X_train = np.concatenate([Xo[tr], Xf[tr]])
        y_train = np.concatenate([y[tr], y[tr]])
        probe, hist = train_probe(X_train, y_train, n_classes, counts_of(y[tr], n_classes),
                                  val=(Xo[va], Xf[va], y[va]))
        histories.append(hist)
        pred = predict_probe(probe, Xo[va], Xf[va])
        acc = float((pred == y[va]).mean())
        accs.append(acc)
        cm_total += confusion_matrix(y[va], pred, labels=range(n_classes))
        print(f"  Fold {fold}: {acc*100:.1f}%  ({int((pred == y[va]).sum())}/{len(va)})")

    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    print(f"  CV: {mean_acc*100:.1f}% +/- {std_acc*100:.1f}%")
    plot_task(task, classes, histories, accs, cm_total)

    report_lines.append(f"== {task} ==")
    report_lines.append(f"Clase: {classes}")
    report_lines.append(f"Imagini: {len(paths)} | Split: {split_note}")
    report_lines.append(f"Acuratete CV ({N_FOLDS} folduri): "
                        f"{mean_acc*100:.1f}% +/- {std_acc*100:.1f}%")
    report_lines.append("Per fold: " + ", ".join(f"{a*100:.1f}%" for a in accs))
    report_lines.append("Matrice de confuzie agregata (rand = clasa reala):")
    header = " " * 12 + "".join(f"{c[:10]:>12s}" for c in classes)
    report_lines.append(header)
    for i, c in enumerate(classes):
        row = f"{c[:10]:>12s}" + "".join(f"{cm_total[i, j]:>12d}" for j in range(n_classes))
        report_lines.append(row)
    for i, c in enumerate(classes):
        support = cm_total[i].sum()
        rec = cm_total[i, i] / support if support else 0.0
        report_lines.append(f"  recall {c}: {rec*100:.1f}% ({cm_total[i, i]}/{support})")
    report_lines.append("")

    X_all = np.concatenate([Xo, Xf])
    y_all = np.concatenate([y, y])
    probe, _ = train_probe(X_all, y_all, n_classes, counts_of(y, n_classes))
    out = os.path.join(MODELS_DIR, f"dinov2_{task}.pth")
    torch.save({
        "arch": "dinov2_vits14+linear",
        "img_size": IMG_SIZE,
        "classes": classes,
        "state_dict": {k: v.cpu() for k, v in probe.state_dict().items()},
        "cv_mean_acc": mean_acc,
        "cv_std_acc": std_acc,
        "grouped_split": use_groups,
    }, out)
    print(f"  Salvat {out}")


def write_report(new_lines):
    sections = {}
    if os.path.exists(REPORT_PATH):
        cur = None
        with open(REPORT_PATH, encoding="utf-8") as f:
            for line in f.read().split("\n"):
                if line.startswith("== ") and line.endswith(" =="):
                    cur = line[3:-3]
                    sections[cur] = []
                if cur is not None:
                    sections[cur].append(line)
    cur = None
    for line in new_lines:
        if line.startswith("== ") and line.endswith(" =="):
            cur = line[3:-3]
            sections[cur] = []
        sections[cur].append(line)
    out = []
    for t in TASKS:
        if t in sections:
            out.extend(sections[t])
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out).rstrip("\n") + "\n")


def main():
    seed_all()
    wanted = sys.argv[1:] or TASKS
    unknown = [t for t in wanted if t not in TASKS]
    if unknown:
        sys.exit(f"Sarcina necunoscuta: {unknown} (valide: {TASKS})")
    for d in (MODELS_DIR, RESULTS_DIR, GRAPHICS_DIR):
        os.makedirs(d, exist_ok=True)

    print("Incarcare DINOv2 ViT-S/14 (backbone inghetat)...")
    backbone = load_backbone(DEVICE)

    report = []
    for t in wanted:
        run_task(t, backbone, report)

    write_report(report)
    print(f"\nRaport CV scris in {REPORT_PATH}")


if __name__ == "__main__":
    main()
