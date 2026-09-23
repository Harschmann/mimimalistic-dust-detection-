"""
Training Studio -- standalone anomaly-detection model trainer
----------------------------------------------------------------
Separate tool from dust_inspector_app.py (the production/operator app) --
see the original design note below the imports for why.

ARCHITECTURE (why it's built this way):

  - AnomalyModelPlugin (abstract base) + MODEL_REGISTRY: every supported
    model (PatchCore, PaDiM, EfficientAd, ...) is a small self-contained
    class implementing one method (`build()`). The UI and the training
    job both iterate MODEL_REGISTRY generically -- neither has an
    if/elif chain naming specific models. Adding a new model later means
    writing ONE new plugin class and one `register_model(...)` call;
    nothing else in the file changes. This is the Open/Closed Principle:
    open for extension (new plugins), closed for modification (existing
    code doesn't need editing).

  - ProfileManager: owns "which phone model (S26, A36, ...) does this
    training run belong to, and where does it live on disk". The UI and
    the training job both go through this instead of building paths
    themselves -- if you ever change how runs are organized on disk, it's
    a one-class change.

  - run_training_job(): pure background-thread function, knows nothing
    about Tkinter. Takes a profile + a plugin + folders, reports progress
    via a callback, and returns a result dict. Testable on its own,
    independent of the GUI.

  - TrainingStudioApp: the GUI. Only this part touches Tkinter/customtkinter.

REQUIRES (on the machine you run this on, not the operator's PC):
    pip install anomalib torch torchvision

This was written against anomalib's current (Lightning Engine-based) API.
It has NOT been execution-tested against a real anomalib install -- the
environment that generated it couldn't install the ~2GB torch/anomalib
stack (disk space) or reach PyTorch's CPU-wheel index (network egress).
The GUI code WAS exercised end-to-end against stubbed Tkinter/customtkinter
modules (catches structural/logic bugs), but that can't verify real
customtkinter widget API compliance or the actual anomalib training path.
Please pip install on your own machine and try it; if you hit an error,
send me the exact traceback and I'll fix it against the anomalib version
you actually have (the Folder/Engine API has shifted across versions).

Run:  python training_studio.py
"""

import os
import sys
import io
import json
import base64
import shutil
import inspect
import threading
import traceback
from abc import ABC, abstractmethod
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import customtkinter as ctk
from PIL import Image

# ------------------------------------------------------------------ theme --
BG = "#0b0d10"
BG_SIDEBAR = "#0e1013"
BG_CARD = "#14171c"
BG_CARD_ALT = "#1b1f26"
BG_CANVAS = "#101317"
BORDER = "#23272e"
ACCENT = "#4f8cff"
ACCENT_HOVER = "#3f74e0"
ACCENT_SOFT = "#182337"
TEXT = "#e6e8eb"
TEXT_MUTED = "#8a919c"
SUCCESS = "#22c55e"
SUCCESS_HOVER = "#16a34a"
DANGER = "#ef4444"
WARNING = "#f59e0b"

ctk.set_appearance_mode("dark")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.join(BASE_DIR, "training_runs")
os.makedirs(WORKDIR, exist_ok=True)

# Small standalone settings file for THIS app only (separate from the
# operator app's own settings.json -- they're two independent processes,
# normally on two different machines) -- currently just remembers the one
# thing that makes folder auto-detect possible: where the operator app's
# source_images folder lives on disk.
TS_SETTINGS_PATH = os.path.join(BASE_DIR, "training_studio_settings.json")


def load_ts_settings():
    if os.path.exists(TS_SETTINGS_PATH):
        try:
            with open(TS_SETTINGS_PATH, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {"source_images_root": data.get("source_images_root")}
        except Exception:
            pass
    return {"source_images_root": None}


def save_ts_settings(settings):
    try:
        with open(TS_SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


def _safe_model_folder_name(name):
    """Mirrors the operator app's own _safe_folder_name() exactly -- the
    folder on disk under source_images/ was created BY that sanitizer, so
    auto-detect has to reproduce it here to compute the same path, rather
    than assuming the raw Model Profile name is always already a valid/
    identical folder name (spaces, slashes, etc. would otherwise silently
    make auto-detect look in the wrong place)."""
    name = (name or "").strip()
    if not name:
        return "UNSPECIFIED"
    keep = "-_.() "
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in name)
    return cleaned.strip() or "UNSPECIFIED"


# ============================================================ MODEL PLUGINS
# Adding a new model = write one class below + one register_model() call.
# Nothing else in this file needs to change.
class AnomalyModelPlugin(ABC):
    key = ""
    display_name = ""
    icon = "cpu_white"
    description = ""
    needs_gpu_for_speed = False

    @abstractmethod
    def _import_class(self):
        """Returns the anomalib model CLASS (not an instance). Imports
        are local to this method so the app can still open (and show
        every plugin in the picker) even before anomalib is installed --
        this only gets called when Start Training is pressed, or by
        check_available() below for the readiness badge."""
        raise NotImplementedError

    def build(self):
        return self._import_class()()

    def check_available(self):
        """Just the import, no instantiation -- instantiating some of
        these downloads a pretrained backbone on first use, which a
        background status check should never trigger on its own.
        Returns (True, None) or (False, error_message)."""
        try:
            self._import_class()
            return True, None
        except Exception as e:
            return False, str(e)


class PatchCorePlugin(AnomalyModelPlugin):
    key = "PatchCore"
    display_name = "PatchCore"
    icon = "layers_white"
    description = ("No backprop training -- extracts features with a pretrained CNN and "
                    "builds a coreset memory bank. Runs fine on CPU. Good default.")
    needs_gpu_for_speed = False

    def _import_class(self):
        from anomalib.models import Patchcore
        return Patchcore


class PaDiMPlugin(AnomalyModelPlugin):
    key = "PaDiM"
    display_name = "PaDiM"
    icon = "cpu_white"
    description = ("Similar idea (pretrained-feature + statistical model), lighter "
                    "memory footprint, slightly lower accuracy typically. CPU-friendly.")
    needs_gpu_for_speed = False

    def _import_class(self):
        from anomalib.models import Padim
        return Padim


class EfficientAdPlugin(AnomalyModelPlugin):
    key = "EfficientAd"
    display_name = "EfficientAd"
    icon = "zap_white"
    description = ("Actual small-network training (student-teacher distillation). Much "
                    "faster inference (sub-5ms on GPU) -- training itself is far faster "
                    "with a GPU too, but works on CPU, just slower.")
    needs_gpu_for_speed = True

    def _import_class(self):
        from anomalib.models import EfficientAd
        return EfficientAd


class FastFlowPlugin(AnomalyModelPlugin):
    key = "FastFlow"
    display_name = "FastFlow"
    icon = "barchart_white"
    description = ("Normalizing flows on top of CNN features -- estimates how 'likely' "
                    "each spatial location is under the normal distribution directly, "
                    "which tends to give strong pixel-level localization for fine defects. "
                    "No memory bank to store (unlike PatchCore), lighter at inference.")
    needs_gpu_for_speed = True

    def _import_class(self):
        from anomalib.models import Fastflow
        return Fastflow


MODEL_REGISTRY = {}


def register_model(plugin_cls):
    instance = plugin_cls()
    MODEL_REGISTRY[instance.key] = instance
    return plugin_cls


register_model(PatchCorePlugin)
register_model(PaDiMPlugin)
register_model(EfficientAdPlugin)
register_model(FastFlowPlugin)


# =========================================================== PROFILE MANAGER
class ProfileManager:
    """Owns where a given phone-model's (S26, A36, ...) training runs live
    on disk. Each phone model can have several camera positions (cam1,
    cam2, ... -- an S25 Ultra has 5, an S26 might have 3), and different
    positions on the same phone differ in FOV/megapixel/optics, so each
    one needs its OWN trained model with its own idea of "normal". That
    means a "run" -- and everything about it (the manifest, run history,
    active run, last-used dataset folders) -- is scoped to a
    (profile, camera) PAIR, not just the profile alone: training S26's
    cam1 and S26's cam2 are two separate, independent training problems
    that happen to share a phone-model name. This mirrors the camera
    labels (cam1, cam2, ...) the operator app now assigns per ROI and
    saves images under (<Model>/<camN>/) -- point Training Studio's
    dataset folders at that same <camN> folder for a given profile/camera
    and the two line up directly. Nothing else in this file builds these
    paths directly -- if the on-disk layout ever needs to change, this is
    the only class that changes."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def list_profiles(self):
        return sorted(d for d in os.listdir(self.base_dir)
                      if os.path.isdir(os.path.join(self.base_dir, d)))

    def profile_dir(self, name):
        d = os.path.join(self.base_dir, name)
        os.makedirs(d, exist_ok=True)
        return d

    def create_profile(self, name):
        name = name.strip()
        if not name:
            raise ValueError("Profile name can't be empty")
        self.profile_dir(name)
        return name

    def list_cameras(self, profile):
        """Camera positions (cam1, cam2, ...) that already exist under
        this profile -- i.e. have been set up (or trained) before via
        create_camera(). Doesn't invent labels on its own."""
        pdir = self.profile_dir(profile)
        return sorted(d for d in os.listdir(pdir)
                      if os.path.isdir(os.path.join(pdir, d)) and
                      os.path.exists(os.path.join(pdir, d, "manifest.json")))

    def camera_dir(self, profile, camera):
        d = os.path.join(self.profile_dir(profile), camera)
        os.makedirs(d, exist_ok=True)
        return d

    def create_camera(self, profile, camera):
        camera = camera.strip()
        if not camera:
            raise ValueError("Camera label can't be empty")
        if not os.path.exists(self.manifest_path(profile, camera)):
            self.save_manifest(profile, camera, self.load_manifest(profile, camera))
        return camera

    def manifest_path(self, profile, camera):
        return os.path.join(self.camera_dir(profile, camera), "manifest.json")

    def load_manifest(self, profile, camera):
        path = self.manifest_path(profile, camera)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"profile": profile, "camera": camera, "runs": [], "active_run_id": None,
                "last_good_dir": "", "last_defect_dir": "", "last_model": "PatchCore"}

    def save_manifest(self, profile, camera, manifest):
        with open(self.manifest_path(profile, camera), "w") as f:
            json.dump(manifest, f, indent=2)

    def remember_last_settings(self, profile, camera, good_dir, defect_dir, model_key):
        manifest = self.load_manifest(profile, camera)
        manifest["last_good_dir"] = good_dir
        manifest["last_defect_dir"] = defect_dir
        manifest["last_model"] = model_key
        self.save_manifest(profile, camera, manifest)

    def record_run(self, profile, camera, run_info, make_active=True):
        manifest = self.load_manifest(profile, camera)
        manifest["runs"].append(run_info)
        if make_active:
            manifest["active_run_id"] = run_info["run_id"]
        self.save_manifest(profile, camera, manifest)
        return manifest

    def set_active_run(self, profile, camera, run_id):
        manifest = self.load_manifest(profile, camera)
        manifest["active_run_id"] = run_id
        self.save_manifest(profile, camera, manifest)


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder):
    """Returns sorted image file paths in folder, or [] if it doesn't
    exist / isn't a folder. Cheap -- just a directory listing, no
    decoding -- safe to call on every folder-picker change."""
    if not folder or not os.path.isdir(folder):
        return []
    try:
        return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                      if f.lower().endswith(IMAGE_EXTS))
    except OSError:
        return []


def detect_gpu():
    """Lightweight GPU presence check that does NOT import torch (torch
    import alone can take a couple seconds and isn't needed just to
    answer 'is there an NVIDIA GPU here') -- just checks for nvidia-smi."""
    return shutil.which("nvidia-smi") is not None


def detect_anomalib():
    """Actually imports anomalib to verify it's really usable (a
    package being pip-installed doesn't guarantee all its own
    dependencies resolved cleanly) -- this genuinely takes a few seconds
    since it pulls in torch, so ALWAYS call this from a background
    thread, never on the GUI thread. Returns (available, version_or_None,
    error_or_None)."""
    try:
        import anomalib
        return True, getattr(anomalib, "__version__", "unknown"), None
    except Exception as e:
        return False, None, str(e)


# ---------------------------------------------------------------- training --
def _make_progress_callback(progress_cb, max_epochs):
    """BEST-EFFORT epoch-progress reporting via a PyTorch Lightning
    callback (anomalib's Engine wraps a Lightning Trainer under the
    hood). Wrapped so that if the Callback import or the Engine's
    `callbacks=` kwarg isn't supported by your installed anomalib/
    lightning version, training still proceeds fine -- the progress bar
    just won't move, and the text log keeps updating at each named stage
    regardless. This is the single piece most likely to need adjusting
    for your exact anomalib version -- if it silently doesn't work,
    that's expected-possible, not a sign something else is broken.

    PatchCore/PaDiM don't do real gradient-descent training (effectively
    one pass), so their bar will jump straight to 100% -- that's correct,
    not a bug; EfficientAd is where this is actually informative."""
    try:
        try:
            from lightning.pytorch.callbacks import Callback
        except ImportError:
            from pytorch_lightning.callbacks import Callback

        class _EpochProgress(Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                epoch = trainer.current_epoch + 1
                pct = int(100 * epoch / max(max_epochs, 1))
                progress_cb(f"PROGRESS:{pct}:Epoch {epoch}/{max_epochs} complete")

        return _EpochProgress()
    except Exception:
        return None


def _apply_jet_colormap(gray_u8):
    """Maps a single-channel 0-255 array to an RGB heatmap using a
    hand-rolled jet-style colormap (blue -> cyan -> green -> yellow -> red)
    instead of just lighting up the red channel. No matplotlib dependency
    needed -- this is a small vectorized piecewise-linear approximation of
    the classic 'jet' colormap that Grad-CAM-style visualizations use, so
    low anomaly scores show as cool blue/green and only genuinely high
    scores show as hot red, which is much easier to read at a glance than
    a single red channel."""
    import numpy as np
    x = gray_u8.astype("float32") / 255.0
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype("uint8")


def generate_defect_heatmaps(onnx_path, defect_dir, run_dir, progress_cb=None, max_examples=20):
    """Runs the just-exported ONNX model DIRECTLY against every image in
    `defect_dir` (the user's own validation/defect folder) and saves one
    heatmap per image -- not a sample drawn from anomalib's whole
    datamodule (which silently includes auto-held-out 'good' images too).
    This guarantees: (1) exactly one heatmap per image you actually put in
    your defect folder, no more, no less, and (2) a real multi-color
    (jet-style) heatmap instead of anomalib's red-only overlay, since we
    build the overlay ourselves from the raw anomaly-map output.

    Best-effort like the function below: if onnxruntime isn't installed,
    the export failed, or the model's output shape doesn't look like a
    pixel-level anomaly map, this returns None and the caller falls back
    to the older anomalib-based method."""
    try:
        import numpy as np
        import onnxruntime as ort

        if not onnx_path or not os.path.isfile(onnx_path):
            return None
        image_files = sorted(
            f for f in os.listdir(defect_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        )
        if not image_files:
            return None

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        in_h, in_w = 256, 256
        shape = input_meta.shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            in_h, in_w = shape[2], shape[3]
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")

        examples_dir = os.path.join(run_dir, "examples")
        os.makedirs(examples_dir, exist_ok=True)
        saved = 0
        for fname in image_files[:max_examples]:
            try:
                src_path = os.path.join(defect_dir, fname)
                pil_img = Image.open(src_path).convert("RGB")
                resized = pil_img.resize((in_w, in_h))
                arr = np.asarray(resized).astype("float32") / 255.0
                arr = (arr - mean) / std
                arr = np.transpose(arr, (2, 0, 1))[None, ...].astype("float32")

                outputs = session.run(None, {input_meta.name: arr})
                # Pick the output with the most elements -- that's the
                # per-pixel anomaly map; a plain anomaly score is just 1
                # number and won't have anything worth drawing.
                amap = max(outputs, key=lambda o: o.size)
                amap = np.squeeze(amap)
                if amap.ndim != 2:
                    continue  # this output wasn't a 2D map -- nothing to draw

                amap_norm = (amap - amap.min()) / max(amap.max() - amap.min(), 1e-6)
                amap_u8 = (amap_norm * 255).astype("uint8")
                heat_rgb = _apply_jet_colormap(amap_u8)
                if heat_rgb.shape[:2] != (in_h, in_w):
                    heat_rgb = np.array(Image.fromarray(heat_rgb).resize((in_w, in_h)))

                base = np.asarray(resized).astype("uint8")
                overlay = (0.55 * base + 0.45 * heat_rgb).astype("uint8")
                out_img = Image.fromarray(overlay)
                # Upscale for a bigger, clearer preview than the raw
                # model-input resolution (which can be quite small).
                view_size = max(320, in_w, in_h)
                out_img = out_img.resize((view_size, view_size), Image.LANCZOS)

                stem = os.path.splitext(fname)[0]
                out_img.save(os.path.join(examples_dir, f"{stem}_heatmap.png"))
                saved += 1
            except Exception:
                continue
        return examples_dir if saved > 0 else None
    except Exception:
        return None


def save_example_heatmaps(model, engine, datamodule, run_dir, max_examples=4):
    """FALLBACK ONLY -- used when generate_defect_heatmaps() above can't
    run (e.g. ONNX export failed or onnxruntime isn't installed). This
    older method predicts over anomalib's whole datamodule rather than
    just your defect folder, so you may see images here that weren't in
    your defect folder (anomalib auto-holds-out some 'good' images as
    normal test samples), and its overlay is a single red channel rather
    than a real heatmap colormap. BEST-EFFORT example anomaly-heatmap
    thumbnails. anomalib's exact prediction output shape/attribute names
    (`batch.image`, `batch.anomaly_map`, tensor layout) have varied across
    versions, so every step here is guarded. If anything doesn't match
    your installed version, this returns None and training results are
    otherwise unaffected -- you simply won't get preview thumbnails this
    run. If you want this working, send me the exact attribute/shape your
    installed anomalib actually returns from engine.predict() and I'll
    adjust this to match precisely."""
    try:
        import numpy as np

        examples_dir = os.path.join(run_dir, "examples")
        os.makedirs(examples_dir, exist_ok=True)
        predictions = engine.predict(model=model, datamodule=datamodule)
        if not predictions:
            return None

        saved = 0
        for batch in predictions:
            images = getattr(batch, "image", None)
            anomaly_maps = getattr(batch, "anomaly_map", None)
            if images is None or anomaly_maps is None:
                continue
            for i in range(len(images)):
                if saved >= max_examples:
                    return examples_dir if saved > 0 else None
                img_t, amap_t = images[i], anomaly_maps[i]
                img_np = img_t.detach().cpu().numpy() if hasattr(img_t, "detach") else np.asarray(img_t)
                amap_np = amap_t.detach().cpu().numpy() if hasattr(amap_t, "detach") else np.asarray(amap_t)
                if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
                    img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
                amap_np = amap_np.squeeze()
                img_np = (img_np - img_np.min()) / max(img_np.max() - img_np.min(), 1e-6)
                img_u8 = (img_np * 255).astype("uint8")
                base = img_u8 if img_u8.ndim == 3 else np.stack([img_u8] * 3, axis=-1)

                amap_norm = (amap_np - amap_np.min()) / max(amap_np.max() - amap_np.min(), 1e-6)
                heat = np.zeros((*amap_norm.shape, 3), dtype="uint8")
                heat[..., 0] = (amap_norm * 255).astype("uint8")  # red channel = anomaly strength
                if base.shape[:2] != heat.shape[:2]:
                    heat = np.array(Image.fromarray(heat).resize((base.shape[1], base.shape[0])))
                overlay = (0.6 * base + 0.4 * heat).astype("uint8")

                Image.fromarray(overlay).save(os.path.join(examples_dir, f"example_{saved}.png"))
                saved += 1
        return examples_dir if saved > 0 else None
    except Exception:
        return None


def write_model_card(run_dir, profile, camera, plugin, good_dir, defect_dir, max_epochs, metrics, onnx_path):
    """A short human-readable summary of exactly what was trained, on
    what data, and how well it did -- dropped into the run folder next to
    the checkpoint/export, so a run is self-documenting even months later
    without needing to remember what settings produced it."""
    lines = [
        f"# Model Card -- {profile} / {camera} / {plugin.display_name}",
        "",
        f"**Trained:** {datetime.now().isoformat(timespec='seconds')}",
        f"**Profile:** {profile}",
        f"**Camera:** {camera}",
        f"**Model:** {plugin.display_name} (`{plugin.key}`)",
        f"**Max epochs setting:** {max_epochs}",
        "",
        "## Dataset",
        f"- Good (training) images: `{good_dir}`",
        f"- Defect (validation) images: `{defect_dir or 'none provided'}`",
        "",
        "## Validation Metrics",
    ]
    if metrics:
        for k, v in metrics.items():
            try:
                lines.append(f"- **{k}:** {float(v):.4f}")
            except (TypeError, ValueError):
                lines.append(f"- **{k}:** {v}")
    else:
        lines.append("- No validation metrics (no defect folder was given)")
    lines += [
        "",
        "## Export",
        f"- ONNX path: `{onnx_path or 'not exported'}`",
        "",
        "## About this model",
        f"- {plugin.description}",
    ]
    path = os.path.join(run_dir, "MODEL_CARD.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def run_training_job(profile, camera, good_dir, defect_dir, plugin, run_dir, max_epochs, progress_cb, done_cb):
    """Runs entirely on a background thread. Knows nothing about Tkinter
    or profiles -- just trains `plugin`'s model on `good_dir` (one
    camera position's images, e.g. <Model>/<camN>/ from the operator app),
    validates against `defect_dir` if given, and saves everything under
    `run_dir` (the caller/ProfileManager decides what that path actually
    is -- now nested per profile AND per camera). progress_cb(str) reports
    status lines; done_cb(result_dict_or_None, error_str_or_None) is
    called exactly once at the end."""
    try:
        progress_cb("Importing anomalib / torch (first run can take a while)...")
        from anomalib.data import Folder
        from anomalib.engine import Engine

        os.makedirs(run_dir, exist_ok=True)

        progress_cb(f"Preparing dataset (good={good_dir}, defect={defect_dir or 'none'})...")
        datamodule_kwargs = dict(
            name="camera_module",
            root=os.path.dirname(good_dir),
            normal_dir=os.path.basename(good_dir),
        )
        if defect_dir and os.path.isdir(defect_dir) and os.listdir(defect_dir):
            datamodule_kwargs["abnormal_dir"] = os.path.basename(defect_dir)
        # 'task' used to be a required Folder() argument in older anomalib
        # versions; newer ones (where Folder infers/doesn't take a task
        # kwarg at all) raise TypeError on it. Rather than hardcode either
        # way and break on whichever anomalib version isn't that one,
        # check this install's actual Folder signature and only pass it
        # if it's accepted.
        folder_params = inspect.signature(Folder.__init__).parameters
        if "task" in folder_params:
            datamodule_kwargs["task"] = "classification"
        try:
            datamodule = Folder(**datamodule_kwargs)
        except TypeError as e:
            # Last-resort fallback in case the signature check above still
            # missed something (e.g. a **kwargs-based __init__ that hides
            # 'task' from inspect but rejects it at runtime, or vice
            # versa) -- retry once with 'task' toggled rather than failing
            # a whole training run over one argument-name mismatch.
            if "task" in str(e) and "task" in datamodule_kwargs:
                progress_cb("(This anomalib version's Folder() rejected the 'task' argument -- retrying without it.)")
                datamodule_kwargs.pop("task")
                datamodule = Folder(**datamodule_kwargs)
            elif "task" in str(e):
                progress_cb("(This anomalib version's Folder() needs a 'task' argument -- retrying with task='classification'.)")
                datamodule_kwargs["task"] = "classification"
                datamodule = Folder(**datamodule_kwargs)
            else:
                raise
        datamodule.setup()

        progress_cb(f"Building model: {plugin.display_name}...")
        model = plugin.build()

        progress_cb("Starting training...")
        epoch_cb = _make_progress_callback(progress_cb, max_epochs)
        try:
            engine = Engine(default_root_dir=run_dir, max_epochs=max_epochs,
                             callbacks=[epoch_cb] if epoch_cb else [])
        except TypeError:
            # this anomalib/lightning version's Engine doesn't accept
            # callbacks= the way expected -- training still runs fine,
            # just without the live progress-bar updates
            progress_cb("(Couldn't attach the live epoch-progress callback for this anomalib "
                        "version -- training will still run, just without the progress bar moving.)")
            engine = Engine(default_root_dir=run_dir, max_epochs=max_epochs)
        engine.fit(datamodule=datamodule, model=model)

        metrics = {}
        if defect_dir and os.path.isdir(defect_dir) and os.listdir(defect_dir):
            progress_cb("Running validation against your defect images...")
            test_results = engine.test(datamodule=datamodule, model=model)
            if test_results:
                metrics = dict(test_results[0])
        else:
            progress_cb("No defect folder given -- skipping validation metrics.")

        progress_cb("Exporting to ONNX...")
        onnx_dir = os.path.join(run_dir, "export")
        os.makedirs(onnx_dir, exist_ok=True)
        onnx_path = None
        try:
            from anomalib.deploy import ExportType
            exported = engine.export(model=model, export_type=ExportType.ONNX, export_root=onnx_dir)
            onnx_path = str(exported) if exported else None
        except Exception as export_err:
            progress_cb(f"ONNX export step raised: {export_err} "
                        "(training + metrics still succeeded).")

        examples_dir = None
        if defect_dir and os.path.isdir(defect_dir) and os.listdir(defect_dir):
            progress_cb("Generating example heatmaps from your defect folder...")
            if onnx_path:
                examples_dir = generate_defect_heatmaps(onnx_path, defect_dir, run_dir)
            if not examples_dir:
                progress_cb("(Falling back to anomalib's own prediction pass for heatmaps "
                            "-- this may include a few extra images beyond just your defect folder.)")
                examples_dir = save_example_heatmaps(model, engine, datamodule, run_dir)
            if examples_dir:
                progress_cb(f"Saved example heatmaps to {examples_dir}")
            else:
                progress_cb("Couldn't generate example heatmaps for this anomalib version "
                            "(training results are otherwise complete).")

        model_card_path = write_model_card(run_dir, profile, camera, plugin, good_dir, defect_dir, max_epochs, metrics, onnx_path)

        result = {
            "run_id": os.path.basename(run_dir),
            "run_dir": run_dir,
            "camera": camera,
            "model_key": plugin.key,
            "metrics": metrics,
            "onnx_path": onnx_path,
            "model_card_path": model_card_path,
            "examples_dir": examples_dir,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join(run_dir, "result_summary.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)

        progress_cb("Done.")
        done_cb(result, None)
    except Exception:
        done_cb(None, traceback.format_exc())


ICONS_B64 = {
    "alertTriangle": (
        "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAYAAABccqhmAAAACXBIWXMAAAsTAAALEwEAmpwYAAAKfElEQVR4nO3dW6xdZRXF8VGK"
        "CipX8ZYYExQRkYuoiGIiCl5IKLQVqEIp2hYsYIE3EhOlgKICioitlquWR2MiWEBRUC61ICKKlwcSVKBKEUFEFDic0mVW2Ce29ezN"
        "vqy95vi+7/9Lxosh0I4159p29vRUAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQP926gRAxuol"
        "XyBphaTbJT0mqdoi9f+2VtLyzj/LiwFI2AxJsyVdL+nZaRb+hTIh6TpJR3T+XQASMVfSPUMsfbfU/6450T8pAL29RtL3Glz8LbNa"
        "0ut4CICfg7v82r7pPCrp/dE/WQD/s2jIX+cPm/o+sJAHAMSrL/YbW1z+qdT/zcXRP3mgZPWVfzJg+adS/7cPjy4BKNGukp4IXP6p"
        "PClpt+gygJJsJelmg+WfyhpJM6NLAUpxusHSb5n6xwRgzF4t6XGDhd8y9S9HXsvTB8brKoNl75ZVPHxgfN4b9Ft+/ab+sR3EAADN"
        "q49svzFY8hfK7yRtzQAAzTrNYLn7zak8fCD/w1+3cBAEGrTKYKkHzXeYAGB0B5of/rql/jG/jwEARjv8/dpgmYfNbzkIAsM71WCJ"
        "R81SBgAY3KsSO/x1CwdBYAjfNljepnIlEwDkf/jrFg6CwACHv7sNlrbpcBAE+rDUYFnHlU8zAUB3r+h8190q0/yjc9wEMI0rDZZ0"
        "3LmCJw/8v/0lPWewoG0cBN/DAACbf4+/Ow2Ws63UR06+hyDQcYrBUradk3n6QP6Hv14HwVcyACjdFQbLGJXLo8sHIr2zkMNft9Q/"
        "93czgij18PcLgyWMzq84CKJEJxssn0tOin4YQNuHv78bLJ5LOAiiKJcbLJ1bLot+KEAbSj/8dQsHQWSPw1/vl8BdHASRs5MMPmnd"
        "syT6IQHjsDOHv75eAI9J2oURRG4uNfh0TSWXRD8soEnv4PA38EHwAEYQuRz+7jD4VE3xIFh3ByRticEypZpPRT88YBQc/kZ7AXAQ"
        "RNIuMfgUTT0rox8iMOzhb4PBAqUeDoJIDoe/Zl8CHASRlBMNPjlzywnRDxXo9/D3iMHC5BYOgkjCSoNlyTXfin64QC9v5/A39oPg"
        "uxhBuB7+bjf4lMw9v+QrBOHoBIPlKCWLox82sKmdOPy1+gJ4tPN9FQEL3zT4VCwtK6IfOlDj8BfzAuAgiHAzJK01+DQsNfXfqMwf"
        "GUaYxQZLUHoWMf+IwOHPIxwEEWKFwfCT5ztYzg6gTfvxFX92B8H9WQG0gcNf/MJPFw6CaMUig2En03ewkB3AOG0vaT0LaPsC+lvn"
        "OAuMxXKDISe9O/gGs49x2FvSJAto/wLa0DnSAo0e/m4xGG7SXwc/7zwzoBGfZPmSe/l8gtlHU4e/hwwGmgx+ENyRFcCo6qMSy5dm"
        "Bxcz/hjFXhz+kj8Ivo0VwDDqI9LNBkNMRutgDQdBDKM+IrF8eXRwPCuAQXD4yysPcxDEIL5uMLSk2Q4uYgXQDw5/+R4E92UF0AuH"
        "v7zDQRA9HW8wpGS8HSxgBzAdDn9lvHw4CGJaFxkMJ2mng6+xA9jUWyU9ywIW8wKq/1j3vqwApg5/PzMYStJuB7fxFYKoHcfyFfvy"
        "mc8KlG07SX81GEQSdxDcIXoIEac+BrF8ZXdwIQtYJg5/8cvnkEkOgmXi8Be/fC65jYNgWeYbDB3x6uDY6KFEOzj8xS+bY9ZzECzD"
        "hQbDRjw7+Gr0cGK89uQr/sKXzP0guA9LmK+fGgwZ8e7gVg6CeTrWYLhIGh0cEz2saP7w9xeDwSJpdMBBMDNfMRgqklYHF0QPLZrB"
        "4S9+mVI9CO7NEqbvJoNhIml2cCsHwbQdYzBEJO0OPh49xBgOh7/45cnlILg9S5ieCwyGh+TRwfnRw4zBcPiLX5qcMslBMC0c/uKX"
        "JrfcwkEwDR8zGBaSZwfzoocbvb1U0v0Gg0Ly7GCdpJezhL7ONxgSkncH50UPOab3JknPGAwIybuDCUlvYQn9/NBgOEgZHdwYPezY"
        "3DyDoSBldXA0S+iBw1/8MpSYdRwEPZxnMAykzA6+HD38pePwF78EpR8E94hegpJx+ItfgtLzk+glKNXRBg+f0EEl6ajoZSgNhz8W"
        "z+nlu46DYLu+ZPDQCR1Um3TwxZZ3oFgc/lg8x5fvBAfBdlxv8LAJHVTTdMBBcMzqYwvLRwfOM3DkuJeg5MPfnw0eMKGDqkcHD0p6"
        "WfSy5Kg+srB8dJDCDJwbvSy52Y0/6hs+1EQDHQTfHL00OeHwxwKm9gL6cfTS5OKjBg+T0EE1RAdzo5cnddty+OPlk/DL50EOgqM5"
        "1+AhEjqoRujgCw19GBZ5+Hua4eMFlPgMTHAQHM51Bg+P0EHVQAc3NPzhmL36eMLy0UFOMzAneqlSOvz9yeCBETqoGuzgAQ6C/fk8"
        "g8fLJ9MZOGfMH57JeyOHv/AhJRrrQXD36CVzdi0DyAJmPgM3RC+ZqzkGD4fQQdVCB7Ojl80Nhz8Wr6SX7wMcBDd3jsFDIXRQtdjB"
        "2UEftnY4/LF4Jb58n+Eg+DwOf/HDSBTSwY9UuPoYwvDRQckzcIQKxeEvfviILA6C9fe7LM7ZBuUTOqgMOjhLheHwFz90RDYdFHcQ"
        "XG1QOqGDyqiDYg6ChxuUTeigMuxglgo4/P3RoGhCB5VhB/fnfhA8y6BkQgfOM7BMmXq9pP8YFEzowHkGnpL0BmXoBwblEjpIYQau"
        "UWY+YlAqoYOUZmCWMvESSfcaFEroIKUZuE/SNsrAMoMyCR2kOANnKnEc/uKHiKR9ENxVCbvGoERCBynPwNVKFIe/+OEheXRwmBLD"
        "4S9+aEg+HdyX2kHwcwalETrIaQY+q4QOf/82KIzQQU4z8FQqB8GrDcoidJDjDHxf5j5sUBKhg5xn4DCZmiHpHoOCCB3kPAN/kLSV"
        "DB1lUA6hgxJm4EgZutugGEIHJczAXTKzh0EphA5KmoE9ZYRv8R0/EKSsDpbJyJ0GhRA6KGkG7pCJF3e+r3l0IYQOSpqBpyW9SAb2"
        "MyiD0EGJM7CPDMwxKILQQYkzMEsGFhoUQeigxBlYIAOnGRRB6KDEGVgqAycaFEHooMQZWCgD8wyKIHRQ4gzMlYGDDIogdFDiDBwo"
        "AztI2mhQBqGDkmbgOUnbyQR/42/8QJCyOrhXRi4zKITQQUkzsFJG+E5A8QNByurgYBmpvyb5EYNSCB2UMAMPS9paZvhW4PGDQcro"
        "4DMyVP9uwOMG5RA6yHkG/ilpR5k6w6AgQgc5z8AZMjaTbw4SPiAk728CMlPm9uJvBgofFJJfB0+6fR/AXuq/vGCDQWmEDnKYgQ2S"
        "Zisxp/AlwuGDQ9LvYKOkJUrUcZImDEokdJDiDExImq/EHdL5woXoMgkdpDQDD0n6gDJR/77lVQalEjpIYQa+K2kXZehDktYaFEzo"
        "oDLsYI2kD6oAh3T+H8G/DEondFAFdvCEpFVuf7inLdtKOlTSmZKulfR7SeslTTKUvJgym4HJzmzXM766M/OHdnYAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADI138B6jViML0PIKwAAAAASUVORK5CYII="
    ),
    "barchart_muted": ("iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAYAAABccqhmAAAD/UlEQVR4nO3XUW5bORQFwedB9r9lz5cRwYlkx07IS3XVBnjwALao"
        "6wIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC+42X3AFjsdcEZx9yrY4bCN6y49PeMvmP/7R4A/9jOyz/h/IdG1wm+YeLFG3ff"
        "vAB4RhMv/3UN3CUAPJtxl+ydUfsEAMIEgGcy6tf1gTE7BYBnMeZSfdKIvQIAYQIAYQLAMxjxnP6C7bsFAMIEAMIEAMIEAMIEAMIE"
        "AMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIE"
        "AMIEAMIEAMIEAMIEAMIEAMIEAMIEAMJ+7B7AX/e64IyXBWewgAA8hxWX/t55YnAwfwHOt/ryTzufb/ACONeki/e2xWvgMF4AZ5p0"
        "+W9N3cUdAnCe6Zds+j5uCACECcBZTvl1PWVnngCc47RLddreJAGAMAGAMAE4w6nP6VN3ZwgAhAkAhAkAhAkAhAkAhAkAhAkAhAkA"
        "hAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkA"
        "hAkAhAkAhAkAhAkAhAkAhP3YPeALXhec8bLgDNjulACsuPT3zhMDntYJfwFWX/5p58M/M/kFMOnivW3xGuCpTH0BTLr8t6bugi+Z"
        "GIDpl2z6Pvi0iQEAFpkWgFN+XU/ZCQ9NCsBpl+q0vfCLSQEAFhMACJsSgFOf06fuhuu65gQA2EAAIEwAIEwAIEwAIEwAIEwAIEwA"
        "IEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwA"
        "IEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwA"
        "IEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIGxKAF52D/iiVbt9nxnn/G3bd08JALCBAEDYpABsfw79odV7fZ9Z533XiL2T"
        "AnBdQz7KJ+za6fvMPPdPjdk5LQDAQhMDMKaOd+zet/v8j+zet/v8j4zaNzEA1zXsI92YsmvKjvem7Jqy471xu8YN+o3X3QOu2d/J"
        "93nM93lg6gvg1u6Pt/v8j+zet/v8j+zet/v8h0aPu2NF0U/8Lm98n8d8HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAOCn/wHm4DiCjtIzcQAAAABJRU5ErkJggg=="),
    "barchart_white": ("iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAYAAABccqhmAAAD/UlEQVR4nO3XUW5bORQFwedB9r9lz5cRwYlkx07IS3XVBnjwALao"
        "6wIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC+42X3AFjsdcEZx9yrY4bCN6y49PeMvmP/7R4A/9jOyz/h/IdG1wm+YeLFG3ff"
        "vAB4RhMv/3UN3CUAPJtxl+ydUfsEAMIEgGcy6tf1gTE7BYBnMeZSfdKIvQIAYQIAYQLAMxjxnP6C7bsFAMIEAMIEAMIEAMIEAMIE"
        "AMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIEAMIE"
        "AMIEAMIEAMIEAMIEAMIEAMIEAMIEAMJ+7B7AX/e64IyXBWewgAA8hxWX/t55YnAwfwHOt/ryTzufb/ACONeki/e2xWvgMF4AZ5p0"
        "+W9N3cUdAnCe6Zds+j5uCACECcBZTvl1PWVnngCc47RLddreJAGAMAGAMAE4w6nP6VN3ZwgAhAkAhAkAhAkAhAkAhAkAhAkAhAkA"
        "hAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkAhAkA"
        "hAkAhAkAhAkAhAkAhAkAhP3YPeALXhec8bLgDNjulACsuPT3zhMDntYJfwFWX/5p58M/M/kFMOnivW3xGuCpTH0BTLr8t6bugi+Z"
        "GIDpl2z6Pvi0iQEAFpkWgFN+XU/ZCQ9NCsBpl+q0vfCLSQEAFhMACJsSgFOf06fuhuu65gQA2EAAIEwAIEwAIEwAIEwAIEwAIEwA"
        "IEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwA"
        "IEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwA"
        "IEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIEwAIGxKAF52D/iiVbt9nxnn/G3bd08JALCBAEDYpABsfw79odV7fZ9Z533XiL2T"
        "AnBdQz7KJ+za6fvMPPdPjdk5LQDAQhMDMKaOd+zet/v8j+zet/v8j4zaNzEA1zXsI92YsmvKjvem7Jqy471xu8YN+o3X3QOu2d/J"
        "93nM93lg6gvg1u6Pt/v8j+zet/v8j+zet/v8h0aPu2NF0U/8Lm98n8d8HwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAOCn/wHm4DiCjtIzcQAAAABJRU5ErkJggg=="),
    "check_accent": (
        "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"
        "ERERERERERERERERERERERERERERERERERERERERERERERERYQz8H4XiIebnmsT4AAAAAElFTkSuQmCC"
    ),
    "check_success": (
        "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"
        "ERERERERERERERERERERERERERERERERERERERERERERERERYQz8H4XiIebnmsT4AAAAAElFTkSuQmCC"
    ),
    "check_white": (
        "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"
        "ERERERERERERERERERERERERERERERERERERERERERERERERYQz8H4XiIebnmsT4AAAAAElFTkSuQmCC"
    ),
    "cpu_muted": (
        "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"
    ),
    "cpu_white": (
        "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"
    ),
    "externalLink": (
        "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"
    ),
    "folder_muted": (
        "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"
    ),
    "folder_white": (
        "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"
    ),
    "hardDrive_white": (
        "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"
    ),
    "layers_muted": (
        "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAYAAABccqhmAAAACXBIWXMAAAsTAAALEwEAmpwYAAAGlUlEQVR4nO3dMa4cRRQF0BfZ"
        "fwckjtgFEhIrYAsOnThw6C0QWXJISkhKSEIw23BUyewATWQ00lhCbYn/gfl9u+qdIz3J4Vep7q3u6hFUAQAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAsIuXVfX2Ntd/Aw28qKo3VTWq6vNtzlX1vqoe0n8csF/wt6MIoGHwFQEs5r8EXxHA5O4RfEUAk3mO4CsCOLg9gq8I4GAS"
        "wVcEEHaE4CsC2NkRg68I4JnNEHxFAHc2Y/AVAfxPKwRfEcC/tGLwFQE8okPwFQFsdAy+IqA9wVcENCT4nghoSPC9GtCQ4LsjoCHB"
        "d1lIQ4LvqwENCb7PhzQk+H5HQEOCnw++HxSxO8HPB10RsDvBzwdbEbA7wc8HWRGwO8HPB1cRsDvBzwdVEbA7wc8HUxGwO8HPBzE9"
        "Z/9b9H4EPx+8o81ZEaxP8PNBO/qcFcF6BD8frNnmrAjWCP7rqvp0gA1l5lyDUVXvquohvZl5OsHPB2e1GYrg+AQ/H5TVZyiC4xH8"
        "fDC6zVAEeYKfD0L3GYpA8NOb0OTXYCgCJ356E5r8GgxF4FE/vQlNfg2GIvCOn96EJr8GQxG43EtvQpNfg6EI3OqnN6HJr8FQBD7n"
        "pTehya/BUAS+46c3oan4GrQsAj/gyW88c6w1GB2KQPDzG80cew3GikUg+PmNZeZag7FCEQh+fiOZuddgzFoE393++PQCGmuwwh4Y"
        "t0xN5fuq+v0Ai2eswcx74FRVP9bEFEF+E5n51uA0e/C3FEF+U5njr8FpteBvKYL8JjPHW4PT6sHfUgT5TWcqvgbtgr+lCPKb0JTg"
        "pykCQexQRKfuJ/5jFEF+k5oS/DRFIIgrFNHJia8I0pvQCP70PBEI8gxFdnLiK4L0JjSCvzxPBIJ+hKI7OfEVQXoTGsFvzxOBInDi"
        "owgUgUd9FIEi8I6PIlAELve4ckegDNzqowgUgc95KAJF4Ds+iqBtEZz8gIe/c0eQD6XgE6cI8iF14hOnCPKh9ahPnCKYc7zjc1eK"
        "IB9qwSdOEeRD7sQnThHkQ+9RnzhFIPigCJz4oAg86oMi8I4PV+4IXO6BInCrD4rA5zxQBL7jQ+c7Ar/Vh4ZFIPjQsAgEHxoWgeBD"
        "wyIQfGhYBIIPDYtA8KFhEQg+NCwCwYeGRSD40LAIBB8aFoHgQ8MiEHxoWASCDw2LQPChkR+q6o/bXP8Nd/Oqqj5W1Z+B79PGGqy0"
        "By5V9UtVfTtjPymC/AYyc67BZebgbymC/IYyc6zBZaXgbymC/AYzx1yDy8rB31IE+Q1n6hBr0Cr4W4ogvwFNCX6aIhDELkV06Xzi"
        "P0YR5DeoKcFPUwSCuEoRXZz4iiC9CY3gT80TgRDPUmIXJ74iSG9CI/hL80Qg5EcpuYsTXxGkN6ER/NY8ESgBJz6KQBF41EcRKALv"
        "+CgCReByj6tvquon/4UihfDEW/2fb4cHi1EESkDwUQSKwImPIuhcBBeP+nzh1SAfSMEnThHkA+rEJ04R5APrUZ84RTDveMfnbhRB"
        "PtCCT5wiyAfciU+cIsgH3qM+cYpA8EEROPFBEXjUB0XgHR+u3BG43ANF4FYfFIHPeaAIfMeH7ncEfqsPDYtA8KFhEQg+NCwCwYeG"
        "RSD40LAIBB8aFoHgQ8MiEHxoWASCDw2LQPChYREIPjQsAsGHhkUg+NCwCAQfGhaB4EPDIhB87u5VVX2oql/Nodfgt9uk/w5T/7gG"
        "H26ZmsqLqnpdVZ8CP1Ix1mCFPTCq6l1VPdTEFEF+I5m51mCsEPwtRZDfWObYazBWDP6WIshvNHOsNRgdgr+lCPIbz5TgpykCQexW"
        "RKPjif8YRZDfmKYEP00RCOJqRTSc+IogvQmN4E/JE4HwzlZew4mvCNKb0Aj+kjwRCPfRym048RVBehMawW/JE4HwO/FRBIrAoz6K"
        "QBF4x0cRKAKXe3y5I3hzu6l1OlqDp+yBc1W991v9tSgC4Rd8FIEicOKjCBRBedRHEXQsAu/4fMUdQT6Ygk+cIsgH1YlPnCLIB9ej"
        "PnGKYL7xjs/dKYJ8sAWfOEWQD7oTnzhFIPigCJz4oAg86oMi8I4PV+4IXO6BInCrD4rA5zxQBL7jw5U7Aj/ZhZZF4Lf60LAIBB8a"
        "FoHgQ8MiEHxoWASCDw2LQPChYREIPjQsAsGHhkUg+NCwCAQfGhaB4EPDIhB8aFgEgg8Ni0DwoWERCD4087Kq3t7m+m8AAAAAAAAA"
        "AAAAAAAAAAAAAAAAAADqmf0FyceS96sETmgAAAAASUVORK5CYII="
    ),
    "layers_white": (
        "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"
    ),
    "play_white": (
        "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"
    ),
    "target_white": (
        "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"
    ),
    "zap_muted": (
        "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"
    ),
    "zap_white": (
        "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"
    ),
}



def _icon(name, size=20):
    """Decodes an embedded base64 icon into a CTkImage. Returns None (icon
    just gets skipped) rather than crashing -- icons are decoration, never
    load-bearing."""
    b64 = ICONS_B64.get(name)
    if not b64:
        return None
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
        return ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
    except Exception:
        return None


class TrainingStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Training Studio -- Anomaly Model Trainer")
        self.root.geometry("1320x900")
        self.root.minsize(1080, 720)
        self.root.configure(fg_color=BG)

        self.f_title = ctk.CTkFont(size=26, weight="bold")
        self.f_subtitle = ctk.CTkFont(size=13)
        self.f_section = ctk.CTkFont(size=15, weight="bold")
        self.f_body = ctk.CTkFont(size=13)
        self.f_body_bold = ctk.CTkFont(size=14, weight="bold")
        self.f_small = ctk.CTkFont(size=11)
        self.f_stat = ctk.CTkFont(size=26, weight="bold")

        self.icons = {}
        for name in ICONS_B64:
            self.icons[f"{name}_20"] = _icon(name, 20)
            self.icons[f"{name}_28"] = _icon(name, 28)

        self.profile_mgr = ProfileManager(WORKDIR)
        self.gpu_available = detect_gpu()
        self.ts_settings = load_ts_settings()

        self.profile_var = tk.StringVar(value="")
        self.camera_var = tk.StringVar(value="")
        self.source_root_var = tk.StringVar(value=self.ts_settings.get("source_images_root") or "")
        self.good_dir_var = tk.StringVar(value="")
        self.defect_dir_var = tk.StringVar(value="")
        self.model_var = tk.StringVar(value=next(iter(MODEL_REGISTRY)))
        self.epochs_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="Idle")
        self.training_running = False
        self._last_result = None
        self.model_card_widgets = {}
        self.dataset_previews = {}
        self._compare_selection = []
        self._history_runs_by_id = {}
        self._example_photos = []

        self._build_ui()
        self._refresh_profile_list(select_first=True)
        threading.Thread(target=self._check_environment_background, daemon=True).start()

    # ------------------------------------------------------------ helpers
    def ic(self, name, size=20):
        return self.icons.get(f"{name}_{size}")

    def _btn_primary(self, parent, text, command, icon=None, **kw):
        opts = dict(text=text, command=command, corner_radius=8, fg_color=ACCENT,
                    hover_color=ACCENT_HOVER, text_color="#ffffff", font=self.f_body)
        if icon:
            opts["image"] = self.ic(icon)
            opts["compound"] = "left"
        opts.update(kw)
        return ctk.CTkButton(parent, **opts)

    def _btn_secondary(self, parent, text, command, icon=None, **kw):
        opts = dict(text=text, command=command, corner_radius=8, fg_color=BG_CARD_ALT,
                    hover_color=BORDER, text_color=TEXT, font=self.f_body)
        if icon:
            opts["image"] = self.ic(icon)
            opts["compound"] = "left"
        opts.update(kw)
        return ctk.CTkButton(parent, **opts)

    def _card(self, parent, **kw):
        defaults = dict(fg_color=BG_CARD, corner_radius=14, border_width=1, border_color=BORDER)
        defaults.update(kw)
        return ctk.CTkFrame(parent, **defaults)

    def _pill(self, parent, textvariable, text_color=TEXT_MUTED):
        """A small rounded status badge. Built as a frame + inner label so
        padding always uses .pack(padx=,pady=), which every customtkinter
        version supports -- rather than passing padx/pady to the CTkLabel
        constructor itself, which isn't consistently supported."""
        frame = ctk.CTkFrame(parent, fg_color=BG_CARD_ALT, corner_radius=10)
        label = ctk.CTkLabel(frame, textvariable=textvariable, font=self.f_small, text_color=text_color)
        label.pack(padx=10, pady=3)
        return frame, label

    def _icon_badge(self, parent, icon, size=48, bg=ACCENT, icon_size=28):
        badge = ctk.CTkFrame(parent, fg_color=bg, corner_radius=size // 4, width=size, height=size)
        badge.pack_propagate(False)
        ctk.CTkLabel(badge, text="", image=self.ic(icon, icon_size)).pack(expand=True)
        return badge

    # ------------------------------------------------------------ layout
    def _build_ui(self):
        outer = ctk.CTkFrame(self.root, fg_color=BG, corner_radius=0)
        outer.pack(fill="both", expand=True, padx=24, pady=22)

        # ---- header ----
        header = ctk.CTkFrame(outer, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        self._icon_badge(header, "cpu_white", size=52, icon_size=28).pack(side="left", padx=(0, 16))
        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left")
        ctk.CTkLabel(title_box, text="Training Studio", font=self.f_title, text_color=TEXT).pack(anchor="w")
        ctk.CTkLabel(title_box, text="Train an anomaly-detection model per phone model, validate, export to ONNX",
                     font=self.f_subtitle, text_color=TEXT_MUTED).pack(anchor="w", pady=(2, 0))

        gpu_frame, gpu_label = self._pill(header, tk.StringVar(value="GPU: Detected" if self.gpu_available else "GPU: Not detected"),
                                           text_color=SUCCESS if self.gpu_available else TEXT_MUTED)
        gpu_frame.pack(side="right", pady=(6, 0), padx=(0, 8))

        self.anomalib_status_var = tk.StringVar(value="Anomalib: Checking...")
        self.anomalib_frame, self.anomalib_label = self._pill(header, self.anomalib_status_var, text_color=TEXT_MUTED)
        self.anomalib_frame.pack(side="right", pady=(6, 0), padx=(0, 8))
        self._btn_secondary(header, "Recheck", self._recheck_environment, width=90, height=28).pack(side="right", pady=(4, 0))

        # ---- two-column dashboard body ----
        body = ctk.CTkFrame(outer, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=5)
        body.grid_columnconfigure(1, weight=6)
        body.grid_rowconfigure(0, weight=1)

        # CTkScrollableFrame, not a plain CTkFrame -- the left column's
        # content (profile+camera picker, model cards, dataset pickers,
        # train button) can be taller than the window, especially at the
        # 1080x720 minsize or with several model cards -- a plain frame
        # would just get cut off with the lower cards/buttons unreachable
        # and no way to scroll down to them. Right column already handles
        # its own overflow (history_body is its own scrollable frame).
        left = ctk.CTkScrollableFrame(body, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        self._build_left_column(left)
        self._build_right_column(right)

    # ------------------------------------------------------- left column
    def _build_left_column(self, parent):
        # ---- 0. profile (which phone model this training run is for) ----
        profile_card = self._card(parent)
        profile_card.pack(fill="x", pady=(0, 14))
        head0 = ctk.CTkFrame(profile_card, fg_color="transparent")
        head0.pack(fill="x", padx=18, pady=(16, 10))
        ctk.CTkLabel(head0, text="", image=self.ic("target_white")).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(head0, text="Model Profile (e.g. S26, A36)", font=self.f_section, text_color=TEXT).pack(side="left")

        profile_row = ctk.CTkFrame(profile_card, fg_color="transparent")
        profile_row.pack(fill="x", padx=18, pady=(0, 8))
        self.profile_menu = ctk.CTkOptionMenu(profile_row, values=["(no profiles yet)"], variable=self.profile_var,
                                               command=self._on_profile_change, fg_color=BG_CARD_ALT,
                                               button_color=BG_CARD_ALT, button_hover_color=BORDER,
                                               text_color=TEXT, dropdown_fg_color=BG_CARD_ALT,
                                               dropdown_hover_color=BORDER, dropdown_text_color=TEXT)
        self.profile_menu.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._btn_secondary(profile_row, "+ New Profile", self._new_profile, width=150).pack(side="left")
        ctk.CTkLabel(profile_card, text="Each profile keeps its own trained models and remembers its last-used "
                                         "dataset folders and model choice -- switching profiles switches all of that.",
                     font=self.f_small, text_color=TEXT_MUTED, wraplength=460, justify="left").pack(anchor="w", padx=18, pady=(0, 10))

        cam_head = ctk.CTkFrame(profile_card, fg_color="transparent")
        cam_head.pack(fill="x", padx=18, pady=(0, 10))
        ctk.CTkLabel(cam_head, text="", image=self.ic("layers_muted")).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(cam_head, text="Camera Position (cam1, cam2, ...)", font=self.f_section, text_color=TEXT).pack(side="left")

        camera_row = ctk.CTkFrame(profile_card, fg_color="transparent")
        camera_row.pack(fill="x", padx=18, pady=(0, 8))
        self.camera_menu = ctk.CTkOptionMenu(camera_row, values=["(no cameras yet)"], variable=self.camera_var,
                                              command=self._on_camera_change, fg_color=BG_CARD_ALT,
                                              button_color=BG_CARD_ALT, button_hover_color=BORDER,
                                              text_color=TEXT, dropdown_fg_color=BG_CARD_ALT,
                                              dropdown_hover_color=BORDER, dropdown_text_color=TEXT)
        self.camera_menu.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._btn_secondary(camera_row, "+ New Camera", self._new_camera, width=150).pack(side="left")
        ctk.CTkLabel(profile_card, text="Different camera positions on the same phone (main, ultrawide, telephoto, "
                                         "...) differ in FOV/megapixel/optics, so each one is trained SEPARATELY as "
                                         "its own model -- a run always belongs to one profile AND one camera "
                                         "together, matching the cam1/cam2/... labels the operator app assigns per ROI.",
                     font=self.f_small, text_color=TEXT_MUTED, wraplength=460, justify="left").pack(anchor="w", padx=18, pady=(0, 16))

        # ---- 1. model ----
        model_card = self._card(parent)
        model_card.pack(fill="x", pady=(0, 14))
        head = ctk.CTkFrame(model_card, fg_color="transparent")
        head.pack(fill="x", padx=18, pady=(16, 10))
        ctk.CTkLabel(head, text="", image=self.ic("layers_muted")).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(head, text="Model", font=self.f_section, text_color=TEXT).pack(side="left")

        self.model_cards_frame = ctk.CTkFrame(model_card, fg_color="transparent")
        self.model_cards_frame.pack(fill="x", padx=14, pady=(0, 16))
        self._build_model_cards_once()

        # ---- 2. dataset ----
        data_card = self._card(parent)
        data_card.pack(fill="x", pady=(0, 14))
        head2 = ctk.CTkFrame(data_card, fg_color="transparent")
        head2.pack(fill="x", padx=18, pady=(16, 10))
        ctk.CTkLabel(head2, text="", image=self.ic("folder_muted")).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(head2, text="Dataset", font=self.f_section, text_color=TEXT).pack(side="left")

        # ---- source root (makes "Good images" auto-detect possible) ----
        root_row = ctk.CTkFrame(data_card, fg_color="transparent")
        root_row.pack(fill="x", padx=18, pady=(0, 10))
        ctk.CTkLabel(root_row, text="Source images root (operator app's own source_images folder)",
                     font=self.f_body, text_color=TEXT).pack(anchor="w", pady=(0, 6))
        root_inner = ctk.CTkFrame(root_row, fg_color="transparent")
        root_inner.pack(fill="x")
        ctk.CTkEntry(root_inner, textvariable=self.source_root_var, fg_color=BG_CARD_ALT, border_color=BORDER,
                     text_color=TEXT, corner_radius=8).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._btn_secondary(root_inner, "Set Folder", self._set_source_root, icon="folder_white", width=110).pack(side="left")
        ctk.CTkLabel(root_row, text="Set this ONCE, pointed at the exact source_images folder the operator app on "
                                     "this same machine saves crops into (only works if this Training Studio and "
                                     "that operator app share a disk -- if training happens on a separate server, "
                                     "point this at wherever the images get copied/synced to). Once set, picking a "
                                     "Model Profile + Camera above auto-fills 'Good images' below whenever that "
                                     "camera already has a folder with images.",
                     font=self.f_small, text_color=TEXT_MUTED, wraplength=460, justify="left").pack(anchor="w", pady=(6, 0))

        self._build_folder_picker(data_card, "Good images (training set)", self.good_dir_var, "good",
                                   autodetect_cmd=self._autodetect_good_dir)
        self._build_folder_picker(data_card, "Defect images (validation, optional)", self.defect_dir_var, "defect")

        ctk.CTkLabel(data_card, text="Point 'Good images' at the operator app's own "
                                      "source_images/<Model>/<camN>/ folder for the profile/camera selected above "
                                      "-- those are already clean, circular, per-camera crops, no relabeling "
                                      "needed. Training only ever uses the good images; the defect folder (if you "
                                      "have known-bad examples for this camera) just measures how well the model "
                                      "separates good from bad.",
                     font=self.f_small, text_color=TEXT_MUTED, wraplength=460, justify="left").pack(anchor="w", padx=18, pady=(4, 16))

        # ---- 3. train ----
        train_card = self._card(parent)
        train_card.pack(fill="x")
        head3 = ctk.CTkFrame(train_card, fg_color="transparent")
        head3.pack(fill="x", padx=18, pady=(16, 10))
        ctk.CTkLabel(head3, text="", image=self.ic("play_white")).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(head3, text="Train", font=self.f_section, text_color=TEXT).pack(side="left")

        epochs_row = ctk.CTkFrame(train_card, fg_color="transparent")
        epochs_row.pack(fill="x", padx=18, pady=(0, 6))
        ctk.CTkLabel(epochs_row, text="Max epochs", font=self.f_small, text_color=TEXT_MUTED).pack(side="left", padx=(0, 8))
        ctk.CTkEntry(epochs_row, textvariable=self.epochs_var, width=60, fg_color=BG_CARD_ALT,
                     border_color=BORDER, text_color=TEXT).pack(side="left")
        ctk.CTkLabel(epochs_row, text="(only matters for EfficientAd -- PatchCore/PaDiM don't do gradient-descent training)",
                     font=self.f_small, text_color=TEXT_MUTED).pack(side="left", padx=(10, 0))

        self.train_btn = self._btn_primary(train_card, "Start Training", self.start_training,
                                            icon="play_white", width=220, height=48, font=self.f_body_bold)
        self.train_btn.pack(anchor="w", padx=18, pady=(12, 18))

    def _build_folder_picker(self, parent, label_text, var, preview_key, autodetect_cmd=None):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(0, 10))
        ctk.CTkLabel(row, text=label_text, font=self.f_body, text_color=TEXT).pack(anchor="w", pady=(0, 6))
        inner = ctk.CTkFrame(row, fg_color="transparent")
        inner.pack(fill="x")
        ctk.CTkEntry(inner, textvariable=var, fg_color=BG_CARD_ALT, border_color=BORDER,
                     text_color=TEXT, corner_radius=8).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._btn_secondary(inner, "Browse", lambda: self._browse(var, preview_key), icon="folder_white", width=110).pack(side="left")
        if autodetect_cmd:
            self._btn_secondary(inner, "Auto-detect", autodetect_cmd, width=120).pack(side="left", padx=(8, 0))

        count_label = ctk.CTkLabel(row, text="", font=self.f_small, text_color=TEXT_MUTED)
        count_label.pack(anchor="w", pady=(8, 0))
        thumbs_row = ctk.CTkFrame(row, fg_color="transparent")
        thumbs_row.pack(fill="x", pady=(4, 0))
        self.dataset_previews[preview_key] = {"count_label": count_label, "thumbs_row": thumbs_row, "photos": []}
        self._refresh_dataset_preview(preview_key, var.get())

    def _refresh_dataset_preview(self, key, folder):
        info = self.dataset_previews.get(key)
        if not info:
            return
        for w in info["thumbs_row"].winfo_children():
            w.destroy()
        info["photos"] = []
        if not folder:
            info["count_label"].configure(text="")
            return
        images = list_images(folder)
        info["count_label"].configure(text=f"{len(images)} image(s) found" if images else "No images found in this folder")
        for path in images[:6]:
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((56, 56))
                photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                info["photos"].append(photo)  # keep a reference -- Tkinter drops images with no owner
                ctk.CTkLabel(info["thumbs_row"], text="", image=photo).pack(side="left", padx=(0, 6))
            except Exception:
                continue

    # --------------------------------------------- model cards (no flicker)
    # Built ONCE; selecting a different model only reconfigures colors and
    # shows/hides the checkmark on the existing widgets. The earlier
    # version destroyed and rebuilt all 3 cards on every click, which is
    # what caused the visible flicker -- Tkinter has to tear down and
    # re-lay-out the whole widget subtree, which is never free. This has
    # nothing to do with threading (training already runs on its own
    # background thread) -- it's purely a "don't rebuild what you can
    # just recolor" fix.
    def _build_model_cards_once(self):
        self.model_card_widgets = {}
        for key, plugin in MODEL_REGISTRY.items():
            card = ctk.CTkFrame(self.model_cards_frame, corner_radius=10, border_width=2, cursor="hand2")
            card.pack(fill="x", pady=4)
            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.pack(fill="x", padx=14, pady=10)

            icon_circle = ctk.CTkFrame(inner, corner_radius=8, width=38, height=38)
            icon_circle.pack(side="left", padx=(0, 12))
            icon_circle.pack_propagate(False)
            ctk.CTkLabel(icon_circle, text="", image=self.ic(plugin.icon)).pack(expand=True)

            text_box = ctk.CTkFrame(inner, fg_color="transparent")
            text_box.pack(side="left", fill="x", expand=True)
            name_row = ctk.CTkFrame(text_box, fg_color="transparent")
            name_row.pack(fill="x", anchor="w")
            ctk.CTkLabel(name_row, text=plugin.display_name, font=self.f_body_bold, text_color=TEXT).pack(side="left")
            checkmark = ctk.CTkLabel(name_row, text="", image=self.ic("check_accent"))
            if plugin.needs_gpu_for_speed:
                gpu_txt = "GPU detected -- will be fast" if self.gpu_available else "no GPU detected -- will be slow"
                gpu_color = SUCCESS if self.gpu_available else WARNING
                ctk.CTkLabel(name_row, text=gpu_txt, font=self.f_small, text_color=gpu_color).pack(side="left", padx=(10, 0))
            ctk.CTkLabel(text_box, text=plugin.description, font=self.f_small, text_color=TEXT_MUTED,
                         wraplength=380, justify="left").pack(anchor="w", pady=(3, 0))
            avail_label = ctk.CTkLabel(text_box, text="Checking availability...", font=self.f_small, text_color=TEXT_MUTED)
            avail_label.pack(anchor="w", pady=(3, 0))

            for widget in (card, inner, icon_circle, text_box, name_row):
                widget.bind("<Button-1>", lambda e, k=key: self._select_model(k))

            self.model_card_widgets[key] = {"card": card, "icon_circle": icon_circle, "checkmark": checkmark, "avail_label": avail_label}
        self._update_model_card_styles()

    def _update_model_card_styles(self):
        for key, w in self.model_card_widgets.items():
            selected = (self.model_var.get() == key)
            w["card"].configure(fg_color=ACCENT_SOFT if selected else BG_CARD_ALT,
                                 border_color=ACCENT if selected else BORDER)
            w["icon_circle"].configure(fg_color=ACCENT if selected else BG_CANVAS)
            if selected:
                w["checkmark"].pack(side="left", padx=(8, 0))
            else:
                w["checkmark"].pack_forget()

    def _select_model(self, key):
        if self.model_var.get() == key:
            return
        self.model_var.set(key)
        self._update_model_card_styles()

    def _recheck_environment(self):
        self.anomalib_status_var.set("Anomalib: Checking...")
        self.anomalib_label.configure(text_color=TEXT_MUTED)
        for widgets in self.model_card_widgets.values():
            widgets["avail_label"].configure(text="Checking availability...", text_color=TEXT_MUTED)
        threading.Thread(target=self._check_environment_background, daemon=True).start()

    # -------------------------------------------- environment detection
    def _check_environment_background(self):
        """Runs off the main thread -- actually importing anomalib (and
        each registered model class) takes a few seconds since it pulls
        in torch. Reports back via root.after so the badge/cards update
        without blocking the GUI at launch."""
        available, version, error = detect_anomalib()
        per_model = {}
        if available:
            for key, plugin in MODEL_REGISTRY.items():
                per_model[key] = plugin.check_available()
        self.root.after(0, self._apply_environment_status, available, version, error, per_model)

    def _apply_environment_status(self, available, version, error, per_model):
        if available:
            self.anomalib_status_var.set(f"Anomalib: Ready (v{version})")
            self.anomalib_label.configure(text_color=SUCCESS)
            if not self.training_running:
                self.train_btn.configure(state="normal")
        else:
            self.anomalib_status_var.set("Anomalib: Not found")
            self.anomalib_label.configure(text_color=DANGER)
            self._log(f"Anomalib import failed at startup: {error}")
            self._log("Start Training is disabled until this is fixed -- "
                       "see the setup steps for installing torch + anomalib.")
            self.train_btn.configure(state="disabled")

        for key, widgets in self.model_card_widgets.items():
            label = widgets["avail_label"]
            if not available:
                label.configure(text="Unavailable -- anomalib not installed", text_color=DANGER)
                continue
            ok, err = per_model.get(key, (False, "not checked"))
            if ok:
                label.configure(text="Ready", text_color=SUCCESS)
            else:
                short_err = (err or "").splitlines()[-1][:80] if err else "unknown error"
                label.configure(text=f"Unavailable: {short_err}", text_color=DANGER)

    # ---------------------------------------------------------- profiles
    def _refresh_profile_list(self, select_first=False):
        profiles = self.profile_mgr.list_profiles()
        values = profiles if profiles else ["(no profiles yet)"]
        self.profile_menu.configure(values=values)
        if select_first and profiles:
            self.profile_var.set(profiles[0])
            self._on_profile_change(profiles[0])
        elif not profiles:
            self.profile_var.set(values[0])
            self._refresh_camera_list()

    def _new_profile(self):
        name = simpledialog.askstring("New Profile", "Phone model name (e.g. S26, A36):", parent=self.root)
        if not name:
            return
        try:
            self.profile_mgr.create_profile(name)
        except ValueError as e:
            messagebox.showerror("Training Studio", str(e))
            return
        self._refresh_profile_list()
        self.profile_var.set(name)
        self._on_profile_change(name)

    def _on_profile_change(self, name):
        if not name or name.startswith("("):
            return
        # Switching profile always re-picks a camera under the NEW profile
        # -- a camera label only means something within its own profile
        # (S26's cam1 and A36's cam1 are unrelated models), so nothing
        # about the previously-selected camera carries over.
        self._refresh_camera_list(select_first=True)

    # ----------------------------------------------------------- cameras
    def _refresh_camera_list(self, select_first=False):
        profile = self.profile_var.get()
        cameras = self.profile_mgr.list_cameras(profile) if profile and not profile.startswith("(") else []
        values = cameras if cameras else ["(no cameras yet)"]
        self.camera_menu.configure(values=values)
        if select_first and cameras:
            self.camera_var.set(cameras[0])
            self._on_camera_change(cameras[0])
        else:
            self.camera_var.set(values[0])
            self._clear_dataset_and_history()

    def _new_camera(self):
        profile = self.profile_var.get()
        if not profile or profile.startswith("("):
            messagebox.showwarning("Training Studio", "Create or select a Model Profile first.")
            return
        name = simpledialog.askstring(
            "New Camera",
            "Camera label (e.g. cam1, cam2) -- match the label assigned to this "
            "ROI in the operator app so the two line up:", parent=self.root)
        if not name:
            return
        try:
            self.profile_mgr.create_camera(profile, name)
        except ValueError as e:
            messagebox.showerror("Training Studio", str(e))
            return
        self._refresh_camera_list()
        self.camera_var.set(name)
        self._on_camera_change(name)

    def _on_camera_change(self, name):
        if not name or name.startswith("("):
            return
        profile = self.profile_var.get()
        manifest = self.profile_mgr.load_manifest(profile, name)
        good_dir = manifest.get("last_good_dir", "")
        if not good_dir:
            # First time this camera's been selected (fresh manifest, no
            # dataset folder chosen yet) -- try to auto-fill from the
            # source root rather than leaving it blank and making the
            # technician Browse to a folder the app can already compute.
            # Never overrides a folder that's already remembered (manual
            # or previously auto-detected) -- only fills a true blank.
            expected = self._expected_good_dir(profile, name)
            if expected and os.path.isdir(expected):
                good_dir = expected
                self.profile_mgr.remember_last_settings(
                    profile, name, good_dir, manifest.get("last_defect_dir", ""), manifest.get("last_model", "PatchCore"))
        self.good_dir_var.set(good_dir)
        self.defect_dir_var.set(manifest.get("last_defect_dir", ""))
        self._refresh_dataset_preview("good", self.good_dir_var.get())
        self._refresh_dataset_preview("defect", self.defect_dir_var.get())
        last_model = manifest.get("last_model")
        if last_model in MODEL_REGISTRY:
            self.model_var.set(last_model)
            self._update_model_card_styles()
        self._refresh_run_history()

    # ------------------------------------------------- source auto-detect
    def _set_source_root(self):
        path = filedialog.askdirectory(title="Select the operator app's source_images folder")
        if not path:
            return
        self.source_root_var.set(path)
        self.ts_settings["source_images_root"] = path
        save_ts_settings(self.ts_settings)
        # Re-run the same first-time-fill check for whatever camera is
        # currently selected, now that the root is known -- lets a
        # technician set the root AFTER already picking a profile/camera
        # without having to reselect the camera manually.
        camera = self.camera_var.get()
        if camera and not camera.startswith("(") and not self.good_dir_var.get():
            self._on_camera_change(camera)

    def _expected_good_dir(self, profile, camera):
        """Where this (profile, camera)'s images SHOULD be, assuming the
        source root is set and the operator app uses its standard
        <root>/<Model>/<camN>/ layout. Returns None if the root or a
        valid profile/camera isn't available yet -- doesn't check whether
        the folder actually exists (callers that care check that
        themselves, since 'not there yet' and 'root not set' need
        different messages)."""
        root = (self.source_root_var.get() or "").strip()
        if not root or not profile or profile.startswith("(") or not camera or camera.startswith("("):
            return None
        return os.path.join(root, _safe_model_folder_name(profile), camera)

    def _autodetect_good_dir(self):
        profile, camera = self.profile_var.get(), self.camera_var.get()
        if not self.source_root_var.get().strip():
            messagebox.showinfo("Auto-detect", "Set the source images root first (button above).")
            return
        if not profile or profile.startswith("(") or not camera or camera.startswith("("):
            messagebox.showinfo("Auto-detect", "Select a Model Profile and Camera first.")
            return
        expected = self._expected_good_dir(profile, camera)
        if not os.path.isdir(expected):
            messagebox.showinfo("Auto-detect", f"No folder found yet at:\n{expected}\n\n"
                                                 "Capture at least one image with this camera in the operator app "
                                                 "first, or Browse to the right folder manually.")
            return
        self.good_dir_var.set(expected)
        self._refresh_dataset_preview("good", expected)
        self.profile_mgr.remember_last_settings(profile, camera, expected, self.defect_dir_var.get(), self.model_var.get())

    def _clear_dataset_and_history(self):
        """No camera selected (new/empty profile) -- blank the dataset
        pickers and run history rather than showing stale data from
        whatever camera/profile was selected before."""
        self.good_dir_var.set("")
        self.defect_dir_var.set("")
        self._refresh_dataset_preview("good", "")
        self._refresh_dataset_preview("defect", "")
        self._refresh_run_history()

    # ------------------------------------------------------ right column
    def _build_right_column(self, parent):
        parent.grid_rowconfigure(0, weight=3)
        parent.grid_rowconfigure(1, weight=3)
        parent.grid_columnconfigure(0, weight=1)

        # ---- progress ----
        log_card = self._card(parent)
        log_card.grid(row=0, column=0, sticky="nsew", pady=(0, 14))
        log_head = ctk.CTkFrame(log_card, fg_color="transparent")
        log_head.pack(fill="x", padx=18, pady=(16, 8))
        ctk.CTkLabel(log_head, text="", image=self.ic("barchart_muted")).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(log_head, text="Progress", font=self.f_section, text_color=TEXT).pack(side="left")
        self.status_pill_frame, self.status_pill = self._pill(log_head, self.status_var)
        self.status_pill_frame.pack(side="right")

        self.progress_bar = ctk.CTkProgressBar(log_card, progress_color=ACCENT, fg_color=BG_CARD_ALT)
        self.progress_bar.pack(fill="x", padx=18, pady=(0, 10))
        self.progress_bar.set(0)

        wrap = ctk.CTkFrame(log_card, fg_color=BG_CANVAS, corner_radius=10)
        wrap.pack(fill="both", expand=True, padx=14, pady=(0, 16))
        self.log_box = ctk.CTkTextbox(wrap, fg_color=BG_CANVAS, text_color=TEXT_MUTED,
                                       font=ctk.CTkFont(family="Courier New", size=11), corner_radius=8, wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_box.configure(state="disabled")

        # ---- results + run history ----
        self.results_card = self._card(parent)
        self.results_card.grid(row=1, column=0, sticky="nsew")
        res_head = ctk.CTkFrame(self.results_card, fg_color="transparent")
        res_head.pack(fill="x", padx=18, pady=(16, 10))
        ctk.CTkLabel(res_head, text="", image=self.ic("target_white")).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(res_head, text="Results", font=self.f_section, text_color=TEXT).pack(side="left")

        self.results_body = ctk.CTkFrame(self.results_card, fg_color="transparent")
        self.results_body.pack(fill="x", padx=18, pady=(0, 8))
        ctk.CTkLabel(self.results_body, text="Train a model to see results here.",
                     font=self.f_body, text_color=TEXT_MUTED).pack(anchor="w")

        history_head = ctk.CTkFrame(self.results_card, fg_color="transparent")
        history_head.pack(fill="x", padx=18, pady=(8, 6))
        ctk.CTkLabel(history_head, text="Run history for this profile", font=self.f_body_bold, text_color=TEXT).pack(side="left")
        self._btn_secondary(history_head, "Compare Selected", self._open_compare_window, width=150, height=28).pack(side="right")
        self.history_body = ctk.CTkScrollableFrame(self.results_card, fg_color="transparent")
        self.history_body.pack(fill="both", expand=True, padx=10, pady=(0, 14))

    def _refresh_run_history(self):
        for w in self.history_body.winfo_children():
            w.destroy()
        profile = self.profile_var.get()
        camera = self.camera_var.get()
        if not profile or profile.startswith("(") or not camera or camera.startswith("("):
            ctk.CTkLabel(self.history_body, text="Select (or create) a camera for this profile first.",
                         font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=8, pady=4)
            return
        manifest = self.profile_mgr.load_manifest(profile, camera)
        runs = list(reversed(manifest.get("runs", [])))
        active_id = manifest.get("active_run_id")
        self._history_runs_by_id = {r["run_id"]: r for r in runs}
        # drop any stale selections from a previous profile/camera/run set
        self._compare_selection = [rid for rid in self._compare_selection if rid in self._history_runs_by_id]
        if not runs:
            ctk.CTkLabel(self.history_body, text=f"No runs yet for {profile} / {camera}.",
                         font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=8, pady=4)
            return
        for run in runs:
            row = ctk.CTkFrame(self.history_body, fg_color=BG_CARD_ALT, corner_radius=8)
            row.pack(fill="x", pady=3, padx=4)
            is_active = run.get("run_id") == active_id
            compare_var = tk.BooleanVar(value=run["run_id"] in self._compare_selection)
            ctk.CTkCheckBox(row, text="", variable=compare_var, width=20, checkbox_width=18, checkbox_height=18,
                            command=lambda rid=run["run_id"], v=compare_var: self._toggle_compare(rid, v)).pack(side="left", padx=(8, 2))
            left_txt = f"{run.get('model_key', '?')}  --  {run.get('timestamp', '?')}"
            ctk.CTkLabel(row, text=left_txt, font=self.f_small, text_color=TEXT).pack(side="left", padx=4, pady=8)
            if is_active:
                ctk.CTkLabel(row, text="ACTIVE", font=self.f_small, text_color=SUCCESS).pack(side="right", padx=10)
            else:
                self._btn_secondary(row, "Set Active", lambda r=run: self._set_active_run(r), width=100, height=26).pack(side="right", padx=8, pady=4)

    def _toggle_compare(self, run_id, var):
        if var.get():
            if len(self._compare_selection) >= 2:
                var.set(False)
                messagebox.showinfo("Compare Runs", "You can compare 2 runs at a time -- uncheck one first.")
                return
            self._compare_selection.append(run_id)
        elif run_id in self._compare_selection:
            self._compare_selection.remove(run_id)

    def _open_compare_window(self):
        if len(self._compare_selection) != 2:
            messagebox.showinfo("Compare Runs", "Tick exactly 2 runs (checkboxes in the history list) to compare.")
            return
        run_a = self._history_runs_by_id.get(self._compare_selection[0])
        run_b = self._history_runs_by_id.get(self._compare_selection[1])
        if not run_a or not run_b:
            return

        win = ctk.CTkToplevel(self.root)
        win.title("Compare Runs")
        win.geometry("720x440")
        win.configure(fg_color=BG)

        body = ctk.CTkFrame(win, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=20)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        all_metric_keys = sorted(set((run_a.get("metrics") or {}).keys()) | set((run_b.get("metrics") or {}).keys()))

        for col, run in enumerate((run_a, run_b)):
            card = self._card(body)
            card.grid(row=0, column=col, sticky="nsew", padx=8)
            ctk.CTkLabel(card, text=run.get("model_key", "?"), font=self.f_section, text_color=TEXT).pack(anchor="w", padx=16, pady=(14, 2))
            ctk.CTkLabel(card, text=run.get("timestamp", "?"), font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=16, pady=(0, 12))
            metrics = run.get("metrics") or {}
            if all_metric_keys:
                for k in all_metric_keys:
                    v = metrics.get(k)
                    try:
                        vs = f"{float(v):.4f}"
                    except (TypeError, ValueError):
                        vs = str(v) if v is not None else "--"
                    mrow = ctk.CTkFrame(card, fg_color="transparent")
                    mrow.pack(fill="x", padx=16, pady=2)
                    ctk.CTkLabel(mrow, text=k, font=self.f_small, text_color=TEXT_MUTED).pack(side="left")
                    ctk.CTkLabel(mrow, text=vs, font=self.f_body_bold, text_color=ACCENT).pack(side="right")
            else:
                ctk.CTkLabel(card, text="No metrics recorded", font=self.f_small, text_color=TEXT_MUTED).pack(anchor="w", padx=16, pady=(0, 10))
            ctk.CTkLabel(card, text=f"ONNX: {run.get('onnx_path') or 'n/a'}", font=self.f_small, text_color=TEXT_MUTED,
                         wraplength=300, justify="left").pack(anchor="w", padx=16, pady=(14, 16))

    def _set_active_run(self, run):
        profile = self.profile_var.get()
        camera = self.camera_var.get()
        self.profile_mgr.set_active_run(profile, camera, run["run_id"])
        self._refresh_run_history()
        self.status_var.set(f"Active run set: {run['run_id']}")

    # -------------------------------------------------------------- logic
    def _browse(self, var, preview_key=None):
        path = filedialog.askdirectory(title="Select folder")
        if path:
            var.set(path)
            if preview_key:
                self._refresh_dataset_preview(preview_key, path)

    def _log(self, line):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {line}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _handle_progress_line(self, line):
        """Training reports plain text for the log, and optionally lines
        shaped 'PROGRESS:<pct>:<text>' for live epoch progress (see
        _make_progress_callback -- only fires if that best-effort
        Lightning hook worked for the installed anomalib version)."""
        if line.startswith("PROGRESS:"):
            try:
                _, pct_str, text = line.split(":", 2)
                self.progress_bar.set(int(pct_str) / 100.0)
                self._log(text)
                return
            except Exception:
                pass
        self._log(line)

    def start_training(self):
        if self.training_running:
            return
        profile = self.profile_var.get()
        if not profile or profile.startswith("("):
            messagebox.showwarning("Training Studio", "Create or select a Model Profile first (e.g. S26, A36).")
            return
        camera = self.camera_var.get()
        if not camera or camera.startswith("("):
            messagebox.showwarning("Training Studio", "Create or select a Camera Position first (e.g. cam1, cam2) "
                                                        "-- each camera on this profile trains its own separate model.")
            return
        good_dir = self.good_dir_var.get().strip()
        defect_dir = self.defect_dir_var.get().strip()
        if not good_dir or not os.path.isdir(good_dir):
            messagebox.showwarning("Training Studio", "Pick a valid 'good images' folder first.")
            return
        if not os.listdir(good_dir):
            messagebox.showwarning("Training Studio", "That good-images folder is empty.")
            return
        try:
            max_epochs = int(self.epochs_var.get())
        except ValueError:
            messagebox.showerror("Training Studio", "Max epochs must be a number.")
            return

        model_key = self.model_var.get()
        plugin = MODEL_REGISTRY[model_key]
        self.profile_mgr.remember_last_settings(profile, camera, good_dir, defect_dir, model_key)

        run_id = f"{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(self.profile_mgr.camera_dir(profile, camera), run_id)

        self.training_running = True
        self.train_btn.configure(text="Training...", state="disabled", fg_color=BG_CARD_ALT)
        self.status_var.set("Running")
        self.status_pill.configure(text_color=WARNING)
        self.progress_bar.set(0)
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self._log(f"Starting {model_key} training for {profile} / {camera}.")

        threading.Thread(
            target=run_training_job,
            args=(profile, camera, good_dir, defect_dir, plugin, run_dir, max_epochs,
                  lambda msg: self.root.after(0, self._handle_progress_line, msg),
                  lambda result, err: self.root.after(0, self._finish_training, profile, camera, result, err)),
            daemon=True,
        ).start()

    def _finish_training(self, profile, camera, result, error):
        self.training_running = False
        self.train_btn.configure(text="Start Training", state="normal", fg_color=ACCENT)
        for w in self.results_body.winfo_children():
            w.destroy()

        if error:
            self.status_var.set("Failed")
            self.status_pill.configure(text_color=DANGER)
            self.progress_bar.set(0)
            self._log("ERROR:\n" + error)
            err_row = ctk.CTkFrame(self.results_body, fg_color="transparent")
            err_row.pack(fill="x", anchor="w")
            ctk.CTkLabel(err_row, text="", image=self.ic("alertTriangle")).pack(side="left", padx=(0, 8))
            ctk.CTkLabel(err_row, text="Training failed -- see the Progress log for the full traceback.",
                         font=self.f_body, text_color=DANGER).pack(side="left")
            messagebox.showerror("Training Studio",
                                  "Training failed -- see the Progress log for the full error. "
                                  "If this looks like an anomalib API mismatch, tell me the exact "
                                  "message and your `pip show anomalib` version and I'll fix the code.")
            return

        self.status_var.set("Done")
        self.status_pill.configure(text_color=SUCCESS)
        self.progress_bar.set(1.0)
        self._last_result = result
        self.profile_mgr.record_run(profile, camera, result, make_active=True)
        metrics = result.get("metrics") or {}

        done_row = ctk.CTkFrame(self.results_body, fg_color="transparent")
        done_row.pack(fill="x", anchor="w", pady=(0, 12))
        ctk.CTkLabel(done_row, text="", image=self.ic("check_success")).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(done_row, text=f"{result['model_key']} training complete for '{profile} / {camera}'",
                     font=self.f_body_bold, text_color=TEXT).pack(side="left")

        if metrics:
            tiles_row = ctk.CTkFrame(self.results_body, fg_color="transparent")
            tiles_row.pack(fill="x", pady=(0, 12))
            for i, (k, v) in enumerate(metrics.items()):
                tiles_row.grid_columnconfigure(i, weight=1)
                tile = ctk.CTkFrame(tiles_row, fg_color=BG_CARD_ALT, corner_radius=10)
                tile.grid(row=0, column=i, sticky="nsew", padx=4)
                try:
                    fv = float(v)
                    vs = f"{fv * 100:.1f}%" if 0 <= fv <= 1 else f"{fv:.3f}"
                except (TypeError, ValueError):
                    vs = str(v)
                ctk.CTkLabel(tile, text=vs, font=self.f_stat, text_color=ACCENT).pack(pady=(14, 0))
                ctk.CTkLabel(tile, text=k, font=self.f_small, text_color=TEXT_MUTED).pack(pady=(0, 14))
        else:
            ctk.CTkLabel(self.results_body, text="No validation metrics (no defect folder was given).",
                         font=self.f_small, text_color=TEXT_MUTED, wraplength=460, justify="left").pack(anchor="w", pady=(0, 12))

        examples_dir = result.get("examples_dir")
        if examples_dir and os.path.isdir(examples_dir):
            example_files = sorted(f for f in os.listdir(examples_dir) if f.lower().endswith(".png"))
            if example_files:
                ctk.CTkLabel(self.results_body, text="Example heatmaps (blue = low, red = high anomaly score -- click one to open full size):",
                             font=self.f_small, text_color=TEXT_MUTED, wraplength=460, justify="left").pack(anchor="w", pady=(0, 4))
                ex_row = ctk.CTkFrame(self.results_body, fg_color="transparent")
                ex_row.pack(fill="x", pady=(0, 12))
                self._example_photos = []
                for fname in example_files[:8]:
                    try:
                        full_path = os.path.join(examples_dir, fname)
                        img = Image.open(full_path).convert("RGB")
                        img.thumbnail((100, 100))
                        photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                        self._example_photos.append(photo)
                        thumb_lbl = ctk.CTkLabel(ex_row, text="", image=photo, cursor="hand2")
                        thumb_lbl.pack(side="left", padx=(0, 8))
                        thumb_lbl.bind("<Button-1>", lambda e, p=full_path, n=fname: self._open_heatmap_viewer(p, n))
                    except Exception:
                        continue

        onnx_txt = result.get("onnx_path") or "not completed -- check the Progress log"
        ctk.CTkLabel(self.results_body, text=f"ONNX export: {onnx_txt}", font=self.f_small, text_color=TEXT_MUTED,
                     wraplength=460, justify="left").pack(anchor="w", pady=(0, 10))
        btn_row = ctk.CTkFrame(self.results_body, fg_color="transparent")
        btn_row.pack(anchor="w")
        self._btn_secondary(btn_row, "Open Run Folder", lambda: self._open_folder(result["run_dir"]),
                             icon="hardDrive_white", width=160).pack(side="left", padx=(0, 8))
        if result.get("model_card_path"):
            self._btn_secondary(btn_row, "Open Model Card", lambda: self._open_folder(result["model_card_path"]),
                                 icon="externalLink", width=160).pack(side="left")

        self._refresh_run_history()

    def _open_heatmap_viewer(self, image_path, title="Heatmap"):
        """Opens the full-resolution heatmap in its own window with
        mouse-wheel zoom (anchored under the cursor), click+drag panning,
        and Zoom In/Out/Fit buttons -- the on-screen thumbnail is only
        100x100, this is where you actually inspect where the anomaly was
        marked."""
        try:
            full_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Training Studio", f"Couldn't open {image_path}:\n{e}")
            return

        win = ctk.CTkToplevel(self.root)
        win.title(title)
        win.geometry("800x700")
        win.configure(fg_color=BG)

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(10, 0))
        ctk.CTkLabel(btn_row, text=title, font=self.f_body_bold, text_color=TEXT).pack(side="left")

        canvas = tk.Canvas(win, bg=BG_CANVAS, highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=10, pady=10)

        state = {"scale": 1.0, "ox": 0, "oy": 0, "photo": None,
                 "drag_x": 0, "drag_y": 0, "img_w": full_img.width, "img_h": full_img.height}

        def fit_scale():
            canvas.update_idletasks()
            cw = max(canvas.winfo_width(), 100)
            ch = max(canvas.winfo_height(), 100)
            return min(cw / full_img.width, ch / full_img.height, 1.0) or 1.0

        def redraw():
            s = max(state["scale"], 0.05)
            w, h = max(int(full_img.width * s), 1), max(int(full_img.height * s), 1)
            resized = full_img.resize((w, h), Image.LANCZOS if s < 1 else Image.NEAREST)
            from PIL import ImageTk
            state["photo"] = ImageTk.PhotoImage(resized)
            canvas.delete("all")
            canvas.create_image(state["ox"], state["oy"], image=state["photo"], anchor="nw")

        def set_scale(new_scale, anchor_x=None, anchor_y=None):
            new_scale = max(0.05, min(new_scale, 10.0))
            if anchor_x is None:
                anchor_x, anchor_y = canvas.winfo_width() / 2, canvas.winfo_height() / 2
            # keep the point under the cursor fixed while zooming
            img_x = (anchor_x - state["ox"]) / max(state["scale"], 1e-6)
            img_y = (anchor_y - state["oy"]) / max(state["scale"], 1e-6)
            state["scale"] = new_scale
            state["ox"] = anchor_x - img_x * new_scale
            state["oy"] = anchor_y - img_y * new_scale
            redraw()

        def on_mousewheel(event):
            direction = 1 if (getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4) else -1
            set_scale(state["scale"] * (1.15 if direction > 0 else 1 / 1.15), event.x, event.y)

        def on_press(event):
            state["drag_x"], state["drag_y"] = event.x, event.y

        def on_drag(event):
            dx, dy = event.x - state["drag_x"], event.y - state["drag_y"]
            state["ox"] += dx
            state["oy"] += dy
            state["drag_x"], state["drag_y"] = event.x, event.y
            redraw()

        def do_fit():
            state["scale"] = fit_scale()
            canvas.update_idletasks()
            cw, ch = canvas.winfo_width(), canvas.winfo_height()
            state["ox"] = (cw - full_img.width * state["scale"]) / 2
            state["oy"] = (ch - full_img.height * state["scale"]) / 2
            redraw()

        canvas.bind("<MouseWheel>", on_mousewheel)     # Windows / macOS
        canvas.bind("<Button-4>", on_mousewheel)        # Linux scroll up
        canvas.bind("<Button-5>", on_mousewheel)        # Linux scroll down
        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)

        zoom_row = ctk.CTkFrame(win, fg_color="transparent")
        zoom_row.pack(fill="x", padx=10, pady=(0, 10))
        self._btn_secondary(zoom_row, "Zoom In", lambda: set_scale(state["scale"] * 1.25), width=90).pack(side="left", padx=(0, 6))
        self._btn_secondary(zoom_row, "Zoom Out", lambda: set_scale(state["scale"] / 1.25), width=90).pack(side="left", padx=(0, 6))
        self._btn_secondary(zoom_row, "Fit", do_fit, width=90).pack(side="left")

        win.after(50, do_fit)

    def _open_folder(self, path):
        if not path:
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception:
            messagebox.showinfo("Training Studio", f"Folder: {path}")


def main():
    root = ctk.CTk()
    TrainingStudioApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
