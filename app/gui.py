
"""Interfata grafica (Tkinter) care completeaza fisa tehnica din imagini.

Utilizare:
    python app/gui.py
"""
import os
import sys
import json
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ne asiguram ca putem importa inference.py indiferent de unde pornim scriptul
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from PIL import Image, ImageTk
import inference as inf

CONFIG_PATH = os.path.join(APP_DIR, "config.json")
DEFAULT_CONFIG = {
    "reject_threshold": 0.55,
    "device": "auto",
    "models_dir": inf.DEFAULT_MODELS_DIR,
}

PREVIEW_W, PREVIEW_H = 360, 270


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    # caile relative din config se rezolva fata de folderul app/, nu fata de cwd
    if not os.path.isabs(cfg["models_dir"]):
        cfg["models_dir"] = os.path.normpath(os.path.join(APP_DIR, cfg["models_dir"]))
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Fisa tehnica Dacia 1300/1310")
        self.root.geometry("1140x710")
        self.root.minsize(980, 620)

        self.cfg = load_config()
        self.engine = inf.Engine(self.cfg["models_dir"], self.cfg["device"])

        # cai catre imaginile selectate in fiecare tab
        self.auto_image_path = None
        self.manual_image_path = None
        self.auto_photo = None      # referinte ca sa nu fie sterse de GC
        self.manual_photo = None

        # coada pentru rezultatele din thread-uri (Tkinter nu e thread-safe)
        self._task_queue = queue.Queue()

        self._setup_style()
        self._build_layout()
        self._poll_queue()

    # ---------------------------------------------------------------- stil
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Fisa.TFrame", background="white")
        style.configure("FisaTitle.TLabel", background="white",
                        font=("Georgia", 16, "bold"))
        style.configure("FisaSub.TLabel", background="white",
                        foreground="#555", font=("Georgia", 9, "italic"))
        style.configure("FisaField.TLabel", background="white",
                        font=("Segoe UI", 10, "bold"))
        style.configure("FisaValue.TLabel", background="white",
                        font=("Segoe UI", 11))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))

    # -------------------------------------------------------------- layout
    def _build_layout(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self._build_auto_tab()
        self._build_manual_tab()
        self._build_settings_tab()

        self._build_fisa(right)

    # ----------------------------------------------------- tab: automat
    def _build_auto_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="  Auto  ")

        ttk.Label(tab, text="Image Query",
                  style="Header.TLabel").pack(anchor="w")
        ttk.Label(tab, foreground="#555",
                  text="Routerul decide tipul de imagine, apoi da query la clasificatorul specializat.").pack(anchor="w", pady=(0, 8))

        holder = tk.Frame(tab, width=PREVIEW_W, height=PREVIEW_H,
                          bg="#eee", relief="solid", bd=1)
        holder.pack(pady=4)
        holder.pack_propagate(False)   # pastreaza dimensiunea in pixeli
        self.auto_preview = tk.Label(holder, bg="#eee")
        self.auto_preview.pack(fill=tk.BOTH, expand=True)

        btns = ttk.Frame(tab)
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text="Choose image",
                   command=self._auto_pick).pack(side=tk.LEFT)
        self.auto_run_btn = ttk.Button(btns, text="Query",
                                       command=self._auto_run, state="disabled")
        self.auto_run_btn.pack(side=tk.LEFT, padx=6)

        ttk.Label(tab, text="Log", style="Header.TLabel").pack(anchor="w", pady=(8, 2))
        self.auto_log = tk.Text(tab, height=9, wrap="word", state="disabled",
                                font=("Consolas", 9))
        self.auto_log.pack(fill=tk.BOTH, expand=True)

    def _auto_pick(self):
        path = self._ask_image()
        if not path:
            return
        self.auto_image_path = path
        self._show_preview(self.auto_preview, path, "auto")
        self.auto_run_btn.config(state="normal")

    def _auto_run(self):
        if not self.auto_image_path:
            return
        self._clear_fisa()   # fisa porneste goala la fiecare analiza
        self._log(self.auto_log, f"→ Analiza: {os.path.basename(self.auto_image_path)}")
        self._set_busy(True)
        thr = self.cfg["reject_threshold"]
        path = self.auto_image_path

        def work():
            result = self.engine.route(path, reject_threshold=thr)
            specialist_scores = {}
            for key in result["specialists"]:
                specialist_scores[key] = self.engine.predict(key, path)
            return result, specialist_scores

        self._run_bg(work, self._auto_done)

    def _auto_done(self, payload, err):
        self._set_busy(False)
        if err:
            self._log(self.auto_log, f"  EROARE: {err}")
            return
        result, specialist_scores = payload
        view = inf.ROUTER_VIEW_LABELS.get(result["top_class"], result["top_class"])
        self._log(self.auto_log,
                  f"  Router: {view} ({result['top_prob']*100:.1f}%)")
        if result["rejected"]:
            prag = self.cfg["reject_threshold"] * 100
            self._log(self.auto_log,
                      f"  RESPINS: sub pragul {prag:.0f}% "
                      f"sau irelevant. Nu se completeaza fisa.")
            messagebox.showwarning(
                "Imagine respinsa",
                f"Scor de incredere prea mic: {result['top_prob']*100:.1f}% "
                f"(prag minim {prag:.0f}%).\n\n"
                f"Vedere estimata: {view}.\n"
                f"Imaginea pare neclara sau irelevanta. Fisa nu a fost completata.")
            return
        for key, scores in specialist_scores.items():
            top_cls, top_p = scores[0]
            field_info = inf.FISA_FIELDS.get(key)
            label = field_info["field"] if field_info else inf.MODEL_REGISTRY[key]["label"]
            pretty = inf.pretty_value(key, top_cls)
            self._log(self.auto_log, f"  {label}: {pretty} ({top_p*100:.1f}%)")
            if field_info:
                self._set_fisa_field(field_info["field"], pretty, prob=top_p)

        # reguli de domeniu derivate din alimentare (cand nu stim varianta)
        if "motor" in specialist_scores:
            self._apply_motor_derived(specialist_scores["motor"][0][0], log=True)

        # reguli de domeniu: deducem specificatiile motorului din varianta
        # (ruleaza ultima ca sa aiba prioritate fata de derivarile generice)
        if "model" in specialist_scores:
            self._apply_derived(specialist_scores["model"][0][0], log=True)

    # ----------------------------------------------------- tab: manual
    def _build_manual_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="  Manual Query  ")

        ttk.Label(tab, text="Ruleaza un model anume",
                  style="Header.TLabel").pack(anchor="w")
        ttk.Label(tab, foreground="#555",
                  text="Alegi tu specialistul si imaginea; vezi distributia "
                       "completa de probabilitati.").pack(anchor="w", pady=(0, 8))

        row = ttk.Frame(tab)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Model:").pack(side=tk.LEFT)
        self.manual_model_var = tk.StringVar()
        # routerul nu e un specialist — nu are sens sa-l rulezi manual aici
        self._model_label_to_key = {v["label"]: k for k, v in inf.MODEL_REGISTRY.items()
                                    if k != "router"}
        combo = ttk.Combobox(row, textvariable=self.manual_model_var,
                             values=list(self._model_label_to_key.keys()),
                             state="readonly", width=38)
        combo.pack(side=tk.LEFT, padx=6)
        combo.current(0)

        mholder = tk.Frame(tab, width=PREVIEW_W, height=PREVIEW_H,
                           bg="#eee", relief="solid", bd=1)
        mholder.pack(pady=6)
        mholder.pack_propagate(False)
        self.manual_preview = tk.Label(mholder, bg="#eee")
        self.manual_preview.pack(fill=tk.BOTH, expand=True)

        btns = ttk.Frame(tab)
        btns.pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="Choose image",
                   command=self._manual_pick).pack(side=tk.LEFT)
        self.manual_run_btn = ttk.Button(btns, text="Query",
                                         command=self._manual_run, state="disabled")
        self.manual_run_btn.pack(side=tk.LEFT, padx=6)
        self.manual_add_btn = ttk.Button(btns, text="Add",
                                         command=self._manual_add_to_fisa, state="disabled")
        self.manual_add_btn.pack(side=tk.LEFT)

        cols = ("clasa", "prob")
        self.manual_tree = ttk.Treeview(tab, columns=cols, show="headings", height=6)
        self.manual_tree.heading("clasa", text="Class")
        self.manual_tree.heading("prob", text="Prob")
        self.manual_tree.column("clasa", width=200)
        self.manual_tree.column("prob", width=120, anchor="e")
        self.manual_tree.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self._manual_last = None  # (model_key, scores) pentru "Adauga in fisa"

    def _manual_pick(self):
        path = self._ask_image()
        if not path:
            return
        self.manual_image_path = path
        self._show_preview(self.manual_preview, path, "manual")
        self.manual_run_btn.config(state="normal")

    def _manual_run(self):
        if not self.manual_image_path:
            return
        key = self._model_label_to_key[self.manual_model_var.get()]
        path = self.manual_image_path
        self._set_busy(True)
        self._run_bg(lambda: (key, self.engine.predict(key, path)), self._manual_done)

    def _manual_done(self, payload, err):
        self._set_busy(False)
        for r in self.manual_tree.get_children():
            self.manual_tree.delete(r)
        if err:
            messagebox.showerror("Eroare", str(err))
            return
        key, scores = payload
        for cls, p in scores:
            pretty = inf.pretty_value(key, cls)
            self.manual_tree.insert("", tk.END, values=(pretty, f"{p*100:.2f}%"))
        self._manual_last = (key, scores)
        # activam "Adauga in fisa" doar daca modelul corespunde unui camp
        can_add = key in inf.FISA_FIELDS
        self.manual_add_btn.config(state="normal" if can_add else "disabled")

    def _manual_add_to_fisa(self):
        if not self._manual_last:
            return
        key, scores = self._manual_last
        info = inf.FISA_FIELDS.get(key)
        if not info:
            return
        top_cls, top_p = scores[0]
        self._set_fisa_field(info["field"], inf.pretty_value(key, top_cls), prob=top_p)
        if key == "model":
            self._apply_derived(top_cls)
        elif key == "motor":
            self._apply_motor_derived(top_cls)

    # ----------------------------------------------------- tab: setari
    def _build_settings_tab(self):
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="  Settings  ")

        # prag reject
        ttk.Label(tab, text="Reject Threshold",
                  style="Header.TLabel").pack(anchor="w")
        ttk.Label(tab, foreground="#555",
                  text="Sub acest scor, routerul respinge imaginea ca "
                       "nesigura/irelevanta.").pack(anchor="w")
        rrow = ttk.Frame(tab)
        rrow.pack(fill=tk.X, pady=(2, 12))
        self.thr_var = tk.DoubleVar(value=self.cfg["reject_threshold"])
        self.thr_label = ttk.Label(rrow, width=6,
                                   text=f"{self.cfg['reject_threshold']:.2f}")
        scale = ttk.Scale(rrow, from_=0.0, to=1.0, variable=self.thr_var,
                          command=lambda v: self.thr_label.config(
                              text=f"{float(v):.2f}"))
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.thr_label.pack(side=tk.LEFT, padx=8)

        # device
        ttk.Label(tab, text="Device",
                  style="Header.TLabel").pack(anchor="w")
        self.device_var = tk.StringVar(value=self.cfg["device"])
        ttk.Combobox(tab, textvariable=self.device_var,
                     values=["auto", "cpu", "cuda"], state="readonly",
                     width=12).pack(anchor="w", pady=(2, 12))


        ttk.Label(tab, text="Path",
                  style="Header.TLabel").pack(anchor="w")
        mrow = ttk.Frame(tab)
        mrow.pack(fill=tk.X, pady=(2, 12))
        self.models_dir_var = tk.StringVar(value=self.cfg["models_dir"])
        ttk.Entry(mrow, textvariable=self.models_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(mrow, text="Browse",
                   command=self._pick_models_dir).pack(side=tk.LEFT, padx=6)

        ttk.Button(tab, text="Save",
                   command=self._save_settings).pack(anchor="w", pady=4)

        # status modele disponibile
        ttk.Label(tab, text="Stare modele",
                  style="Header.TLabel").pack(anchor="w", pady=(14, 2))
        self.status_text = tk.Text(tab, height=7, wrap="word", state="disabled",
                                   font=("Consolas", 9))
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self._refresh_status()

    def _pick_models_dir(self):
        d = filedialog.askdirectory(initialdir=self.models_dir_var.get())
        if d:
            self.models_dir_var.set(d)

    def _save_settings(self):
        self.cfg["reject_threshold"] = round(float(self.thr_var.get()), 2)
        self.cfg["device"] = self.device_var.get()
        self.cfg["models_dir"] = self.models_dir_var.get()
        save_config(self.cfg)
        # recream motorul ca sa preia device-ul / directorul nou (cache curatat)
        self.engine = inf.Engine(self.cfg["models_dir"], self.cfg["device"])
        self._refresh_status()
        messagebox.showinfo("Setari", "Setarile au fost salvate.")

    def _refresh_status(self):
        lines = [f"Device efectiv: {self.engine.device}",
                 f"Director: {self.engine.models_dir}", ""]
        for key, spec in inf.MODEL_REGISTRY.items():
            ok = self.engine.available(key)
            if ok:
                lines.append(f"[OK] {spec['label']}  ({spec['weights']}, dinov2)")
            else:
                lines.append(f"[--] {spec['label']}  ({spec['weights']})")
        self._set_text(self.status_text, "\n".join(lines))

    # --------------------------------------------------------- fisa tehnica
    def _build_fisa(self, parent):
        outer = ttk.Frame(parent, style="Fisa.TFrame", padding=18)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text="FISA TEHNICA", style="FisaTitle.TLabel").pack()
        ttk.Label(outer, text="Autoturisme Dacia 1300/1310",
                  style="FisaSub.TLabel").pack(pady=(0, 14))

        sep = ttk.Separator(outer)
        sep.pack(fill=tk.X, pady=(0, 10))

        self.fisa_vars = {}
        grid = ttk.Frame(outer, style="Fisa.TFrame")
        grid.pack(fill=tk.X)
        for i, field in enumerate(inf.FISA_ORDER):
            ttk.Label(grid, text=field + ":", style="FisaField.TLabel",
                      width=20, anchor="w").grid(row=i, column=0, sticky="w", pady=7)
            var = tk.StringVar(value="...")
            self.fisa_vars[field] = var
            ttk.Label(grid, textvariable=var, style="FisaValue.TLabel",
                      anchor="w").grid(row=i, column=1, sticky="w", padx=8)

        ttk.Separator(outer).pack(fill=tk.X, pady=14)

        btns = ttk.Frame(outer, style="Fisa.TFrame")
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Clear",
                   command=self._clear_fisa).pack(side=tk.LEFT)
        ttk.Button(btns, text="Export",
                   command=self._export_fisa).pack(side=tk.LEFT, padx=6)

    def _set_fisa_field(self, field, value, prob=None, note=None):
        if field not in self.fisa_vars:
            return
        if note:
            self.fisa_vars[field].set(f"{value}   ({note})")
        elif prob is not None:
            self.fisa_vars[field].set(f"{value}   ({prob*100:.0f}%)")
        else:
            self.fisa_vars[field].set(value)

    def _apply_derived(self, variant_class, log=False):
        """Aplica regulile de domeniu pornind de la varianta (ex: 1300 -> 1289 cm³)."""
        derived = inf.derive_specs(variant_class)
        if not derived:
            return
        for field, value in derived.items():
            self._set_fisa_field(field, value)
            if log:
                self._log(self.auto_log, f"  [derivat] {field}: {value}")

    def _apply_motor_derived(self, motor_class, log=False):
        """Reguli deduse din alimentare (ex: carburatie -> distribuitor)."""
        derived = inf.derive_from_motor(motor_class)
        if not derived:
            return
        for field, value in derived.items():
            self._set_fisa_field(field, value)
            if log:
                self._log(self.auto_log, f"  [derivat] {field}: {value}")

    def _clear_fisa(self):
        for var in self.fisa_vars.values():
            var.set("...")

    def _export_fisa(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text", "*.txt")],
            initialfile="fisa_tehnica.txt")
        if not path:
            return
        lines = ["FISA TEHNICA Dacia 1300/1310", "=" * 40, ""]
        for field in inf.FISA_ORDER:
            lines.append(f"{field:24s}: {self.fisa_vars[field].get()}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        messagebox.showinfo("Export", f"Fisa a fost salvata in:\n{path}")

    # ------------------------------------------------------------- utilitare
    def _ask_image(self):
        return filedialog.askopenfilename(
            title="Alege o imagine",
            filetypes=[("Imagini", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("Toate", "*.*")])

    def _show_preview(self, widget, path, which):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((PREVIEW_W, PREVIEW_H))
            photo = ImageTk.PhotoImage(img)
        except Exception as e:
            messagebox.showerror("Eroare", f"Nu pot deschide imaginea:\n{e}")
            return
        if which == "auto":
            self.auto_photo = photo
        else:
            self.manual_photo = photo
        widget.config(image=photo)

    def _log(self, widget, msg):
        widget.config(state="normal")
        widget.insert(tk.END, msg + "\n")
        widget.see(tk.END)
        widget.config(state="disabled")

    def _set_text(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state="disabled")

    def _set_busy(self, busy):
        state = "disabled" if busy else "normal"
        if self.auto_image_path:
            self.auto_run_btn.config(state=state)
        if self.manual_image_path:
            self.manual_run_btn.config(state=state)
        self.root.config(cursor="watch" if busy else "")

    # --- rulare in fundal ca sa nu inghete interfata in timpul inferentei ---
    def _run_bg(self, fn, on_done):
        def worker():
            try:
                res, err = fn(), None
            except Exception as e:  # noqa
                res, err = None, e
            self._task_queue.put(lambda: on_done(res, err))
        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                cb = self._task_queue.get_nowait()
                cb()
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
