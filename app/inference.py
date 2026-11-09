"""
Motorul de inferenta - incarca modelele si ruleaza predictiile.
Foloseste aceeasi preprocesare ca antrenarea (dacia/preprocessing.py).
Importat atat de tab-ul automat cat si de cel manual.

Arhitectura (identica pentru toate sarcinile): backbone DINOv2 ViT-S/14
inghetat, incarcat o singura data si partajat, + un strat liniar per sarcina
(models/dinov2_<task>.pth, care contine si lista de clase).
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)            # parintele lui /app
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from dacia.preprocessing import EMBED_DIM, IMG_SIZE, build_transform, load_backbone


# Registrul modelelor: sarcina -> fisierul cu stratul liniar. Ordinea claselor
# si img_size se citesc din checkpoint (salvate de train_dinov2.py).
MODEL_REGISTRY = {
    "router":    {"label": "Router (clasificator de vedere)",       "weights": "dinov2_router.pth"},
    "caroserie": {"label": "Caroserie (berlina / break)",           "weights": "dinov2_caroserie.pth"},
    "model":     {"label": "Varianta / Generatie",                  "weights": "dinov2_model.pth"},
    "motor":     {"label": "Motorizare (carburatie / injectie)",    "weights": "dinov2_motor.pth"},
}

# Clasa de vedere data de router -> specialistii care se cheama pentru ea.
# Nota: "fata" NU cheama clasificatorul de caroserie - acela e antrenat doar
# pe vederi din spate, iar berlina si break sunt identice vazute din fata.
SPECIALIST_MAP = {
    "fata":    ["model"],
    "spate":   ["caroserie"],
    "motor":   ["motor"],
    "bord":    [],
    "altceva": [],
}

# Eticheta lizibila pentru clasele router-ului.
ROUTER_VIEW_LABELS = {
    "fata": "Partea frontala",
    "spate": "Partea din spate",
    "motor": "Compartiment motor",
    "bord": "Bord / interior",
    "altceva": "Altceva (irelevant)",
}

# Cum se traduce rezultatul fiecarui specialist intr-un camp din fisa tehnica.
FISA_FIELDS = {
    "caroserie": {"field": "Tip caroserie",
                  "values": {"berlina": "Berlina", "break": "Break", "papuc": "Pickup"}},
    "model":     {"field": "Varianta / Generatie",
                  "values": {"1300": "Dacia 1300", "1310": "Dacia 1310",
                             "cn1": "1310 CN1", "cn3": "1310 CN3", "cn4": "1310 CN4"}},
    "motor":     {"field": "Alimentare",
                  "values": {"carburatie": "Carburatie", "injectie": "Injectie"}},
}

# Ordinea campurilor in fisa tehnica.
FISA_ORDER = ["Varianta / Generatie", "Capacitate cilindrica",
              "Tip caroserie", "Alimentare", "Aprindere"]

# Reguli de domeniu: din varianta detectata deducem specificatii care NU se vad
# direct in poza. Ex: o Dacia 1300 are intotdeauna motorul 1289 cm3 cu carburator
# simplu corp - deci nu avem nevoie de un model pentru asta, e o regula fixa.
# Valorile completate aici se marcheaza in fisa ca "dedus", nu ca masurat.
DERIVED_SPECS = {
    "1300": {
        "Capacitate cilindrica": "1289 cm³",
        "Alimentare": "Carburator simplu corp (Carfil 32 IRM / Solex 32 SEIA)",
        "Aprindere": "Distribuitor Ducellier 4224",
    },
    "cn1": {
        "Aprindere": "Distribuitor I.E.P.S 3239",
    },
    "cn3": {
        "Capacitate cilindrica": "1289 / 1397 cm³",
        "Alimentare": "Carburator Carfil",
        "Aprindere": "Distribuitor I.E.P.S 3239",
    },
    "cn4": {
        "Alimentare": "Injectie monopunct Bosch",
        "Aprindere": "Electronica, comandata de ECU",
    },
    # Nota: pentru variantele 1310 (cn1/cn3/cn4) alimentarea NU se deduce -
    # 1310 a avut atat carburatie cat si injectie, deci ramane pe seama
    # specialistului "motor". Doar aprinderea e caracteristica fixa pe varianta.
}


def pretty_value(spec_key, class_name):
    """Traduce numele clasei interne in eticheta de afisat in fisa."""
    info = FISA_FIELDS.get(spec_key)
    if info is None:
        return class_name
    return info["values"].get(class_name, class_name)


def derive_specs(variant_class):
    """Returneaza specificatiile deduse dintr-o varianta (dict camp->valoare)."""
    return dict(DERIVED_SPECS.get(variant_class, {}))


# Reguli derivate din rezultatul specialistului de motor, cand NU cunoastem
# varianta exacta (ex: poza din compartimentul motor). Un motor cu carburatie
# are aprindere clasica - distribuitor Ducellier sau I.E.P.S (nu stim care anume
# fara varianta). Injectia a aparut doar pe 1310 CN4, deci o motorizare pe
# injectie ne spune si varianta, si ca aprinderea e electronica.
MOTOR_DERIVED = {
    "carburatie": {"Aprindere": "Distribuitor Ducellier sau I.E.P.S"},
    "injectie": {
        "Varianta / Generatie": "1310 CN4",
        "Capacitate cilindrica": "1397 cm³",
        "Aprindere": "Electronica, comandata de ECU",
    },
}


def derive_from_motor(motor_class):
    """Returneaza specificatiile deduse din tipul de alimentare detectat."""
    return dict(MOTOR_DERIVED.get(motor_class, {}))


class Engine:
    """Tine device-ul, directorul de modele si un cache cu straturile liniare.

    Backbone-ul DINOv2 e comun tuturor sarcinilor si se incarca o singura data.
    Inferenta foloseste TTA: media probabilitatilor pe imaginea originala si
    pe oglindirea ei orizontala (la fel ca la evaluarea din train_dinov2.py).
    """

    def __init__(self, models_dir=None, device="auto"):
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.device = self._resolve_device(device)
        self._cache = {}       # key -> (probe, classes, img_size)
        self._dinov2 = None    # backbone-ul comun, incarcat o singura data

    def _resolve_device(self, device):
        if device == "cpu":
            return torch.device("cpu")
        if device == "cuda":
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _weights_path(self, key):
        return os.path.join(self.models_dir, MODEL_REGISTRY[key]["weights"])

    def available(self, key):
        """True daca exista fisierul de greutati pentru sarcina."""
        return os.path.exists(self._weights_path(key))

    def _get_dinov2(self):
        if self._dinov2 is None:
            self._dinov2 = load_backbone(self.device)
        return self._dinov2

    def get_model(self, key):
        if key in self._cache:
            return self._cache[key]
        wpath = self._weights_path(key)
        if not os.path.exists(wpath):
            raise FileNotFoundError(f"Lipseste fisierul de model: {wpath}")
        ckpt = torch.load(wpath, map_location=self.device, weights_only=True)
        classes = ckpt["classes"]
        probe = nn.Linear(EMBED_DIM, len(classes))
        probe.load_state_dict(ckpt["state_dict"])
        probe.to(self.device).eval()
        entry = (probe, classes, ckpt.get("img_size", IMG_SIZE))
        self._cache[key] = entry
        return entry

    def predict(self, key, image_path):
        """Returneaza lista [(clasa, probabilitate)] sortata descrescator.

        Foloseste TTA (test-time augmentation): predictia finala e media
        probabilitatilor pe imaginea originala si pe cea oglindita orizontal.
        """
        probe, classes, img_size = self.get_model(key)
        tf = build_transform(img_size)
        image = Image.open(image_path).convert("RGB")
        x = tf(image).unsqueeze(0).to(self.device)
        x_flip = torch.flip(x, dims=[3])

        with torch.no_grad():
            backbone = self._get_dinov2()
            feats = backbone(torch.cat([x, x_flip]))
            feats = F_torch.normalize(feats, dim=1)
            out = probe(feats)
            probs = F_torch.softmax(out, dim=1).mean(dim=0)

        scores = [(classes[i], float(probs[i])) for i in range(len(classes))]
        scores.sort(key=lambda t: -t[1])
        return scores

    def route(self, image_path, reject_threshold=0.55):
        """Ruleaza routerul si decide ce specialisti se cheama (sau reject)."""
        scores = self.predict("router", image_path)
        top_class, top_prob = scores[0]
        rejected = top_prob < reject_threshold or top_class == "altceva"
        specialists = [] if rejected else SPECIALIST_MAP.get(top_class, [])
        return {
            "scores": scores,
            "top_class": top_class,
            "top_prob": top_prob,
            "rejected": rejected,
            "specialists": specialists,
        }
