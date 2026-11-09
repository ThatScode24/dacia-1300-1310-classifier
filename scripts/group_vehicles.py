"""Grupare semi-automata a imaginilor pe vehicul (pentru validarea grupata).

Pozele aceluiasi vehicul au embeddings DINOv2 foarte apropiate, deci propunem
grupe pe baza similaritatii cosinus (>= SIM_THRESHOLD), le verificam manual,
apoi codificam ID-ul vehiculului in numele fisierului (car<ID>__<nume>.jpg).

Utilizare:
    python scripts/group_vehicles.py propose
        -> results/groups_proposed.txt (editabil) + results/groups_review.html
    python scripts/group_vehicles.py apply
        -> redenumeste fisierele dupa groups_proposed.txt si reconstruieste
           datele routerului (populate_router.py)
"""
import os
import sys
import glob
import html
import shutil
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from dacia.preprocessing import IMG_SIZE, build_transform, load_backbone
from dacia.data import is_image, group_of
import populate_router

TRAIN_ROOT = os.path.join(BASE_DIR, "train")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TXT_PATH = os.path.join(RESULTS_DIR, "groups_proposed.txt")
HTML_PATH = os.path.join(RESULTS_DIR, "groups_review.html")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_DIRS = [
    "caroserie/berlina", "caroserie/break",
    "model/1300", "model/cn3", "model/cn4",
    "motor/carburatie", "motor/injectie",
]

ROUTER_CLASSES = ["fata", "spate", "motor"]

SIM_THRESHOLD = 0.90


PREPROC = build_transform(IMG_SIZE)


@torch.no_grad()
def embed(backbone, paths):
    out = []
    for i in range(0, len(paths), 32):
        batch = []
        for p in paths[i:i + 32]:
            x = PREPROC(Image.open(p).convert("RGB"))
            batch.append(x)
            batch.append(torch.flip(x, dims=[2]))
        feats = backbone(torch.stack(batch).to(DEVICE))
        feats = feats.view(-1, 2, feats.shape[1]).mean(dim=1)
        feats = torch.nn.functional.normalize(feats, dim=1)
        out.append(feats.cpu())
    return torch.cat(out).numpy()


def cluster(feats, threshold):
    n = len(feats)
    sim = feats @ feats.T
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def propose():
    print("Incarcare DINOv2 ViT-S/14...")
    backbone = load_backbone(DEVICE)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    txt_lines = []
    html_parts = ["<meta charset='utf-8'><title>Review grupe vehicule</title>",
                  "<style>body{font-family:sans-serif;background:#f4f4f4}"
                  ".grp{background:#fff;margin:10px;padding:10px;border-radius:8px}"
                  ".grp h3{margin:0 0 6px 0}"
                  "img{height:130px;margin:3px;border-radius:4px;vertical-align:top}"
                  "figure{display:inline-block;margin:3px;text-align:center}"
                  "figcaption{font-size:10px;max-width:180px;overflow-wrap:anywhere}"
                  "</style>",
                  "<h1>Propuneri de grupe (acelasi vehicul)</h1>",
                  "<p>Verifica in special grupele cu mai multe imagini: toate "
                  "pozele dintr-o grupa trebuie sa arate ACELASI exemplar de "
                  "masina. Corecteaza in results/groups_proposed.txt.</p>"]

    car_id = 0
    multi, single = 0, 0
    for rel in CLASS_DIRS:
        folder = os.path.join(TRAIN_ROOT, rel)
        paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if is_image(p))
        if not paths:
            continue
        already = [p for p in paths if group_of(p)]
        if already:
            print(f"  {rel}: {len(already)} fisiere au deja prefix car<ID>__ "
                  f"- le las neatinse, grupez restul")
            paths = [p for p in paths if not group_of(p)]

        print(f"  {rel}: {len(paths)} imagini de grupat")
        if not paths:
            continue
        feats = embed(backbone, paths)
        groups = cluster(feats, SIM_THRESHOLD)

        txt_lines.append(f"# === task: {rel} ===")
        html_parts.append(f"<h2>{html.escape(rel)}</h2>")
        for g in groups:
            car_id += 1
            gid = f"car{car_id:04d}"
            txt_lines.append(f"[{gid}]")
            if len(g) > 1:
                multi += 1
                figs = []
                for idx in g:
                    fname = os.path.basename(paths[idx])
                    txt_lines.append(fname)
                    src = html.escape(os.path.join("..", "train", rel, fname)
                                      .replace(os.sep, "/"))
                    figs.append(f"<figure><img src='{src}' loading='lazy'>"
                                f"<figcaption>{html.escape(fname)}</figcaption></figure>")
                html_parts.append(f"<div class='grp'><h3>{gid} "
                                  f"({len(g)} imagini)</h3>{''.join(figs)}</div>")
            else:
                single += 1
                txt_lines.append(os.path.basename(paths[g[0]]))

    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")
    html_parts.append(f"<p>Grupele de o singura imagine ({single}) nu sunt "
                      f"afisate aici - sunt deja corecte prin definitie daca "
                      f"nicio alta poza nu arata aceeasi masina.</p>")
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"\nPropuneri: {multi} grupe multi-imagine, {single} singletons")
    print(f"  de editat : {TXT_PATH}")
    print(f"  de vazut  : {HTML_PATH}  (deschide in browser)")
    print("Dupa verificare: python scripts/group_vehicles.py apply")


def parse_txt():
    mapping = {}
    rel, gid = None, None
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# === task:"):
                rel = line.split("task:")[1].strip(" =")
                mapping.setdefault(rel, {})
            elif line.startswith("#"):
                continue
            elif line.startswith("[") and line.endswith("]"):
                gid = line[1:-1].strip()
            else:
                if rel is None or gid is None:
                    raise ValueError(f"Linie inainte de header task/grupa: {line}")
                if line in mapping[rel]:
                    raise ValueError(f"Fisier duplicat in txt: {rel}/{line}")
                mapping[rel][line] = gid
    return mapping


def apply():
    mapping = parse_txt()
    renamed, missing = 0, 0
    for rel, files in mapping.items():
        folder = os.path.join(TRAIN_ROOT, rel)
        for fname, gid in files.items():
            src = os.path.join(folder, fname)
            if not os.path.exists(src):
                print(f"  LIPSA (sar): {rel}/{fname}")
                missing += 1
                continue
            if group_of(fname):
                continue
            digits = "".join(ch for ch in gid if ch.isdigit()) or "0"
            dst = os.path.join(folder, f"car{int(digits):04d}__{fname}")
            os.rename(src, dst)
            renamed += 1
    print(f"Redenumite: {renamed} | lipsa: {missing}")

    for cls in ROUTER_CLASSES:
        d = os.path.join(TRAIN_ROOT, "router", cls)
        if os.path.isdir(d):
            shutil.rmtree(d)
    populate_router.populate()
    print("\nGata. Ruleaza acum: python scripts/train_dinov2.py")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "propose":
        propose()
    elif cmd == "apply":
        apply()
    else:
        sys.exit(__doc__)
