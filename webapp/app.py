"""
WhiteCoat.dev — Professional Skin Lesion AI Platform v2
========================================================
Run: python webapp/app.py
Open: http://localhost:7860

Features:
- Single model + Ensemble inference
- GradCAM explainability
- Segmentation with ABCDE analysis
- Multi-model comparison with voting
- Batch processing
- Clinical risk dashboard
- Professional medical UI
"""
import os
import sys
import json
from pathlib import Path
from collections import Counter

import cv2
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

# ============================================================
# PATHS — adjust if needed
# ============================================================
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "all_models"
KFOLD_DIR = MODELS_DIR / "kfold"

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cpu")

FOLDER_ORDER = ['0', '1', '10', '11', '2', '3', '4', '5', '6', '7', '8', '9']
MODEL_IDX_TO_REAL = {i: int(FOLDER_ORDER[i]) for i in range(12)}
REAL_TO_MODEL_IDX = {v: k for k, v in MODEL_IDX_TO_REAL.items()}

CLASS_INFO = {
    0:  {"name": "Actinic Keratosis",       "short": "AK",   "type": "Pre-malignant", "emoji": "🟡"},
    1:  {"name": "Basal Cell Carcinoma",     "short": "BCC",  "type": "Malignant",     "emoji": "🔴"},
    2:  {"name": "Dermatofibroma",           "short": "DF",   "type": "Benign",        "emoji": "🟢"},
    3:  {"name": "Hemangioma",               "short": "HEM",  "type": "Benign",        "emoji": "🟢"},
    4:  {"name": "Intraepithelial Carcinoma","short": "IEC",  "type": "Malignant",     "emoji": "🔴"},
    5:  {"name": "Lentigo",                  "short": "LEN",  "type": "Benign",        "emoji": "🟢"},
    6:  {"name": "Melanoma",                 "short": "MEL",  "type": "Malignant",     "emoji": "🔴"},
    7:  {"name": "Melanocytic Nevus",        "short": "NV",   "type": "Benign",        "emoji": "🟢"},
    8:  {"name": "Pyogenic Granuloma",       "short": "PG",   "type": "Benign",        "emoji": "🟢"},
    9:  {"name": "Seborrheic Keratosis",     "short": "SK",   "type": "Benign",        "emoji": "🟢"},
    10: {"name": "Squamous Cell Carcinoma",  "short": "SCC",  "type": "Malignant",     "emoji": "🔴"},
    11: {"name": "Wart",                     "short": "WART", "type": "Benign",        "emoji": "🟢"},
}

CLASS_DESCRIPTIONS = {
    0: "Rough scaly patch on sun-damaged skin. Precursor to SCC (5-10% risk).",
    1: "Most common skin cancer. Pearly papule, locally invasive, rarely metastasizes.",
    2: "Firm brown nodule. Positive dimple sign. Completely benign.",
    3: "Red-purple vascular growth. Most common tumor of infancy.",
    4: "SCC in situ (Bowen's disease). Can progress to invasive SCC if untreated.",
    5: "Flat brown macule. Must distinguish from lentigo maligna melanoma.",
    6: "DEADLIEST skin cancer. Early detection: 95% vs 20% survival rate.",
    7: "Common mole. Critical to distinguish from melanoma.",
    8: "Rapidly growing red nodule. Must exclude amelanotic melanoma.",
    9: "Waxy stuck-on papule. Often biopsied to exclude melanoma.",
    10: "Second most common skin cancer. Can metastasize if untreated.",
    11: "HPV-induced. Must differentiate from verrucous carcinoma.",
}

IMAGE_SIZE = 224

# ============================================================
# MODEL CACHE
# ============================================================
_model_cache = {}


def _load_single(path, model_name=None):
    key = str(path)
    if key in _model_cache:
        return _model_cache[key]
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if model_name is None:
        model_name = ckpt.get("config", {}).get("model_name", "tf_efficientnetv2_s")
    m = timm.create_model(model_name, pretrained=False, num_classes=12)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(DEVICE)
    m.train(False)
    _model_cache[key] = (m, model_name, ckpt.get("val_acc", 0))
    return _model_cache[key]


def _load_kfold(base_name):
    key = f"kfold_{base_name}"
    if key in _model_cache:
        return _model_cache[key]
    model_name = base_name.replace("kfold_", "")
    models = []
    for fold in range(5):
        path = KFOLD_DIR / f"{base_name}_fold{fold}.pth"
        if path.exists():
            m, _, _ = _load_single(path, model_name)
            models.append(m)
    _model_cache[key] = models
    return models


def _load_seg():
    if "seg" in _model_cache:
        return _model_cache["seg"]
    if not HAS_SMP:
        return None
    path = MODELS_DIR / "best_UnetPlusPlus_efficientnet_b4.pth"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    m = smp.UnetPlusPlus(
        encoder_name=cfg.get("encoder", "efficientnet-b4"),
        encoder_weights=None, classes=1, activation=None
    )
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(DEVICE)
    m.train(False)
    _model_cache["seg"] = m
    return m


# ============================================================
# TRANSFORMS
# ============================================================
val_tf = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensorV2(),
])

seg_tf = A.Compose([
    A.Resize(256, 256),
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ============================================================
# INFERENCE HELPERS
# ============================================================
def _to_tensor(img_rgb):
    return val_tf(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)


def _predict_single(model, tensor):
    with torch.no_grad():
        return F.softmax(model(tensor), dim=1)[0].cpu().numpy()


def _predict_ensemble(models, tensor):
    probs = np.zeros(12)
    for m in models:
        probs += _predict_single(m, tensor)
    return probs / len(models)


def _map_probs(raw_probs):
    """Map model-index probabilities to real class probabilities."""
    real = {}
    for mi in range(12):
        rc = MODEL_IDX_TO_REAL[mi]
        real[rc] = float(raw_probs[mi])
    return real


def _top_prediction(raw_probs):
    top_mi = int(np.argmax(raw_probs))
    top_rc = MODEL_IDX_TO_REAL[top_mi]
    return top_rc, float(raw_probs[top_mi])


# ============================================================
# GRADCAM
# ============================================================
def _get_target_layer(model, name):
    if "efficientnet" in name:
        return [model.blocks[-1]] if hasattr(model, "blocks") else [model.features[-1]]
    if "convnext" in name:
        return [model.stages[-1].blocks[-1]]
    if "swin" in name:
        return [model.layers[-1].blocks[-1].norm1]
    if "resnet" in name:
        return [model.layer4[-1]]
    return None


def _gradcam(model, model_name, img_rgb, target_mi):
    if not HAS_GRADCAM:
        return None
    tl = _get_target_layer(model, model_name)
    if not tl:
        return None
    try:
        resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        flt = resized.astype(np.float32) / 255.0
        tensor = _to_tensor(img_rgb)
        cam = GradCAM(model=model, target_layers=tl)
        gc = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target_mi)])[0]
        return show_cam_on_image(flt, gc, use_rgb=True)
    except Exception:
        return None


# ============================================================
# ABCDE
# ============================================================
def _abcde(mask_bin, img_256):
    if mask_bin.sum() < 10:
        return {"A": 0, "B": 0, "C": 0, "D": 0, "risk": 0, "level": "N/A"}
    h, w = mask_bin.shape
    left, right = mask_bin[:, :w//2], np.flip(mask_bin[:, w//2:w//2*2], axis=1)
    mw = min(left.shape[1], right.shape[1])
    asym = np.abs(left[:, :mw].astype(float) - right[:, :mw].astype(float)).sum()
    asym /= max(left[:, :mw].sum() + right[:, :mw].sum(), 1)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    irreg, diam = 0, 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        irreg = 1 - (4 * np.pi * area) / (perim * perim + 1e-6)
        pts = c.reshape(-1, 2)
        if len(pts) > 2:
            from scipy.spatial.distance import pdist
            diam = pdist(pts).max()

    masked = img_256[mask_bin == 1]
    color_var = min(masked.std(axis=0).mean() / 80, 1.0) if len(masked) > 0 else 0

    risk = asym * 0.3 + irreg * 0.3 + color_var * 0.2 + min(diam / 100, 1) * 0.2
    level = "LOW" if risk < 0.3 else "MODERATE" if risk < 0.6 else "HIGH"
    return {"A": asym, "B": irreg, "C": color_var, "D": diam, "risk": risk, "level": level}


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================
def analyze(image, mode):
    if image is None:
        return {}, None, None, "", ""

    img_rgb = np.array(image)
    tensor = _to_tensor(img_rgb)

    # Decide which models to use
    if mode == "Single Best (EfficientNetV2-S)":
        m, mn, _ = _load_single(MODELS_DIR / "best_tf_efficientnetv2_s.pth")
        raw = _predict_single(m, tensor)
        gradcam_model, gradcam_name = m, mn
    elif mode == "K-Fold Ensemble (5x EfficientNetV2)":
        models = _load_kfold("kfold_tf_efficientnetv2_s")
        raw = _predict_ensemble(models, tensor)
        gradcam_model, gradcam_name = models[0], "tf_efficientnetv2_s"
    elif mode == "Multi-Architecture Ensemble":
        all_probs = []
        # Load all available kfold models
        for base in ["kfold_tf_efficientnetv2_s", "kfold_swin_tiny_patch4_window7_224", "kfold_convnext_tiny"]:
            kf_models = _load_kfold(base)
            if kf_models:
                for m in kf_models:
                    all_probs.append(_predict_single(m, tensor))
        # Also add single models
        for fname in ["best_tf_efficientnetv2_s.pth", "best_swin_tiny_patch4_window7_224.pth"]:
            p = MODELS_DIR / fname
            if p.exists():
                m, _, _ = _load_single(p)
                all_probs.append(_predict_single(m, tensor))
        raw = np.mean(all_probs, axis=0) if all_probs else np.zeros(12)
        m0, mn0, _ = _load_single(MODELS_DIR / "best_tf_efficientnetv2_s.pth")
        gradcam_model, gradcam_name = m0, mn0
    else:
        m, mn, _ = _load_single(MODELS_DIR / "best_tf_efficientnetv2_s.pth")
        raw = _predict_single(m, tensor)
        gradcam_model, gradcam_name = m, mn

    # Map to real classes
    real_probs = _map_probs(raw)
    label_dict = {CLASS_INFO[rc]["name"]: prob for rc, prob in real_probs.items()}

    top_rc, top_conf = _top_prediction(raw)
    top_mi = int(np.argmax(raw))
    info = CLASS_INFO[top_rc]

    # GradCAM
    gc_img = _gradcam(gradcam_model, gradcam_name, img_rgb, top_mi)

    # Segmentation + ABCDE
    seg_model = _load_seg()
    abcde_html = ""
    seg_img = None
    if seg_model is not None:
        seg_tensor = seg_tf(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = torch.sigmoid(seg_model(seg_tensor)).squeeze().cpu().numpy()
        mask_bin = (pred > 0.5).astype(np.uint8)
        img_256 = cv2.resize(img_rgb, (256, 256))
        overlay = img_256.copy()
        overlay[mask_bin == 1] = [255, 50, 50]
        seg_img = cv2.addWeighted(img_256, 0.6, overlay, 0.4, 0)

        abcde = _abcde(mask_bin, img_256)
        rc = "#44BB44" if abcde["level"] == "LOW" else "#FFA500" if abcde["level"] == "MODERATE" else "#FF4444"
        abcde_html = f"""
        <div style="font-family:system-ui;padding:16px;border-radius:12px;background:#f9fafb;border:1px solid #e5e7eb;">
            <h3 style="margin:0 0 12px;color:#1f2937;">ABCDE Clinical Features</h3>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div style="padding:8px;background:white;border-radius:8px;border:1px solid #e5e7eb;">
                    <b>A</b> Asymmetry: <span style="float:right;">{abcde['A']:.2f}</span>
                    <div style="background:#e5e7eb;border-radius:4px;height:6px;margin-top:4px;">
                        <div style="background:{'#ef4444' if abcde['A']>0.3 else '#22c55e'};width:{min(abcde['A']*100,100):.0f}%;height:6px;border-radius:4px;"></div>
                    </div>
                </div>
                <div style="padding:8px;background:white;border-radius:8px;border:1px solid #e5e7eb;">
                    <b>B</b> Border: <span style="float:right;">{abcde['B']:.2f}</span>
                    <div style="background:#e5e7eb;border-radius:4px;height:6px;margin-top:4px;">
                        <div style="background:{'#ef4444' if abcde['B']>0.4 else '#22c55e'};width:{min(abcde['B']*100,100):.0f}%;height:6px;border-radius:4px;"></div>
                    </div>
                </div>
                <div style="padding:8px;background:white;border-radius:8px;border:1px solid #e5e7eb;">
                    <b>C</b> Color: <span style="float:right;">{abcde['C']:.2f}</span>
                    <div style="background:#e5e7eb;border-radius:4px;height:6px;margin-top:4px;">
                        <div style="background:{'#ef4444' if abcde['C']>0.5 else '#22c55e'};width:{min(abcde['C']*100,100):.0f}%;height:6px;border-radius:4px;"></div>
                    </div>
                </div>
                <div style="padding:8px;background:white;border-radius:8px;border:1px solid #e5e7eb;">
                    <b>D</b> Diameter: <span style="float:right;">{abcde['D']:.0f}px</span>
                    <div style="background:#e5e7eb;border-radius:4px;height:6px;margin-top:4px;">
                        <div style="background:{'#ef4444' if abcde['D']>60 else '#22c55e'};width:{min(abcde['D']/1.5,100):.0f}%;height:6px;border-radius:4px;"></div>
                    </div>
                </div>
            </div>
            <div style="margin-top:12px;padding:10px;background:{rc}15;border:1px solid {rc};border-radius:8px;text-align:center;">
                <b style="color:{rc};font-size:16px;">Risk: {abcde['level']} ({abcde['risk']:.2f})</b>
            </div>
        </div>"""

    # Risk card
    tc = "#ef4444" if info["type"] == "Malignant" else "#f59e0b" if info["type"] == "Pre-malignant" else "#22c55e"
    risk_html = f"""
    <div style="font-family:system-ui;padding:20px;border-radius:16px;background:linear-gradient(135deg, {tc}08, {tc}15);border:2px solid {tc};margin:0;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
            <span style="font-size:36px;">{info['emoji']}</span>
            <div>
                <h2 style="margin:0;color:{tc};font-size:22px;">{info['name']}</h2>
                <span style="background:{tc};color:white;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;">{info['type'].upper()}</span>
            </div>
        </div>
        <div style="background:white;border-radius:12px;padding:12px;margin:8px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:14px;color:#6b7280;">Confidence</span>
                <span style="font-size:24px;font-weight:700;color:{tc};">{top_conf:.1%}</span>
            </div>
            <div style="background:#e5e7eb;border-radius:6px;height:8px;margin-top:6px;">
                <div style="background:{tc};width:{top_conf*100:.0f}%;height:8px;border-radius:6px;transition:width 0.5s;"></div>
            </div>
        </div>
        <p style="color:#6b7280;font-size:13px;margin:8px 0 0;line-height:1.4;">{CLASS_DESCRIPTIONS[top_rc]}</p>
        <p style="color:#9ca3af;font-size:10px;margin:12px 0 0;text-align:center;">
            For research and demonstration purposes only. Not for clinical use.
        </p>
    </div>"""

    return label_dict, gc_img, seg_img, risk_html, abcde_html


# ============================================================
# COMPARISON TAB
# ============================================================
def compare_all(image):
    if image is None:
        return ""
    img_rgb = np.array(image)
    tensor = _to_tensor(img_rgb)

    rows = []

    # Single models
    for fname in sorted(MODELS_DIR.glob("best_*.pth")):
        if "UnetPlusPlus" in fname.name or "DeepLab" in fname.name or "Unet_" in fname.name:
            continue
        try:
            m, mn, vacc = _load_single(fname)
            raw = _predict_single(m, tensor)
            rc, conf = _top_prediction(raw)
            rows.append({"model": mn, "type": "Single", "val_acc": vacc,
                         "pred_rc": rc, "conf": conf})
        except Exception:
            pass

    # K-fold ensembles
    for base in ["kfold_tf_efficientnetv2_s", "kfold_swin_tiny_patch4_window7_224", "kfold_convnext_tiny"]:
        models = _load_kfold(base)
        if models:
            raw = _predict_ensemble(models, tensor)
            rc, conf = _top_prediction(raw)
            mname = base.replace("kfold_", "")
            rows.append({"model": f"{mname} (5-fold)", "type": "Ensemble",
                         "val_acc": 0, "pred_rc": rc, "conf": conf})

    if not rows:
        return "<p>No models available</p>"

    # Sort by confidence
    rows.sort(key=lambda x: -x["conf"])

    html = """<table style="width:100%;border-collapse:collapse;font-family:system-ui;font-size:14px;">
    <tr style="background:#f3f4f6;"><th style="padding:10px;text-align:left;">Model</th>
    <th style="padding:10px;">Type</th><th style="padding:10px;">Prediction</th>
    <th style="padding:10px;">Confidence</th><th style="padding:10px;">Risk</th></tr>"""

    for r in rows:
        info = CLASS_INFO[r["pred_rc"]]
        tc = "#ef4444" if info["type"] == "Malignant" else "#f59e0b" if info["type"] == "Pre-malignant" else "#22c55e"
        bar = f'<div style="background:#e5e7eb;border-radius:4px;width:120px;height:14px;display:inline-block;vertical-align:middle;"><div style="background:{tc};width:{r["conf"]*100:.0f}%;height:14px;border-radius:4px;"></div></div>'
        html += f"""<tr style="border-bottom:1px solid #e5e7eb;">
            <td style="padding:8px;font-weight:500;">{r['model']}</td>
            <td style="padding:8px;text-align:center;"><span style="background:{'#dbeafe' if r['type']=='Ensemble' else '#f3f4f6'};padding:2px 8px;border-radius:8px;font-size:12px;">{r['type']}</span></td>
            <td style="padding:8px;font-weight:600;">{info['emoji']} {info['name']}</td>
            <td style="padding:8px;text-align:center;">{bar} {r['conf']:.1%}</td>
            <td style="padding:8px;text-align:center;color:{tc};font-weight:600;">{info['type']}</td>
        </tr>"""
    html += "</table>"

    # Consensus
    preds = [r["pred_rc"] for r in rows]
    consensus = Counter(preds).most_common(1)[0]
    cinfo = CLASS_INFO[consensus[0]]
    agreement = consensus[1] / len(preds)
    ctc = "#ef4444" if cinfo["type"] == "Malignant" else "#f59e0b" if cinfo["type"] == "Pre-malignant" else "#22c55e"
    html += f"""<div style="margin-top:12px;padding:14px;background:{ctc}10;border:2px solid {ctc};border-radius:12px;text-align:center;">
        <span style="font-size:20px;">{cinfo['emoji']}</span>
        <b style="font-size:16px;color:{ctc};"> Consensus: {cinfo['name']}</b>
        <span style="color:#6b7280;"> ({agreement:.0%} agreement, {len(preds)} models)</span>
    </div>"""

    return html


# ============================================================
# GRADIO APP
# ============================================================
def build():
    modes = [
        "Single Best (EfficientNetV2-S)",
        "K-Fold Ensemble (5x EfficientNetV2)",
        "Multi-Architecture Ensemble",
    ]

    css = """
    .gradio-container {max-width:1400px !important; margin:auto;}
    footer {display:none !important;}
    """

    with gr.Blocks(title="WhiteCoat.dev", css=css, theme=gr.themes.Soft(
        primary_hue="blue", secondary_hue="red",
        font=gr.themes.GoogleFont("Inter"),
    )) as app:

        gr.HTML("""
        <div style="text-align:center;padding:20px 0 10px;">
            <h1 style="font-family:system-ui;font-size:28px;font-weight:700;color:#1e3a5f;margin:0;">
                WhiteCoat.dev
            </h1>
            <p style="color:#6b7280;font-size:14px;margin:4px 0 0;">
                AI-Powered Skin Lesion Analysis | 12-Class Classification + Segmentation + Explainability
            </p>
        </div>""")

        with gr.Tabs():
            # ========== TAB 1: ANALYSIS ==========
            with gr.Tab("Analysis"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        img_in = gr.Image(type="numpy", label="Upload Skin Lesion", height=300)
                        mode_in = gr.Radio(modes, value=modes[0], label="Inference Mode")
                        btn = gr.Button("Analyze", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        risk_out = gr.HTML()
                        probs_out = gr.Label(num_top_classes=5, label="All Predictions")

                with gr.Row():
                    with gr.Column():
                        gc_out = gr.Image(type="numpy", label="GradCAM — Where the Model Looks", height=280)
                    with gr.Column():
                        seg_out = gr.Image(type="numpy", label="Lesion Segmentation", height=280)

                abcde_out = gr.HTML()

                btn.click(analyze, [img_in, mode_in], [probs_out, gc_out, seg_out, risk_out, abcde_out])

                gr.Examples(
                    examples=[
                        [str(f)] for f in sorted((BASE_DIR / "data/classification/train").glob("*/0.png"))[:6]
                    ] if (BASE_DIR / "data/classification/train").exists() else [],
                    inputs=[img_in],
                    label="Example Images (one per class)",
                )

            # ========== TAB 2: COMPARISON ==========
            with gr.Tab("Model Comparison"):
                gr.Markdown("### Compare all models on the same image — see consensus")
                with gr.Row():
                    cmp_img = gr.Image(type="numpy", label="Upload Image", height=250)
                    cmp_btn = gr.Button("Compare All Models", variant="primary", size="lg")
                cmp_out = gr.HTML()
                cmp_btn.click(compare_all, [cmp_img], [cmp_out])

            # ========== TAB 3: ABOUT ==========
            with gr.Tab("About"):
                gr.Markdown(f"""
### WhiteCoat.dev | AI in Healthcare Hackathon 2026
**Team #37** | Central Asian University, Tashkent

#### Trained Models
| Architecture | K-Fold AVG Accuracy |
|---|---|
| EfficientNetV2-S | **88.3%** |
| Swin-Tiny | **87.9%** |
| ConvNeXt-Tiny | **85.1%** |
| U-Net++ (Segmentation) | **81.0% IoU** |

**Total: 25+ models** (3 architectures x 5 folds + single models)

#### Pipeline
ImageNet pretrained → Fine-tuned on 11,411 clinical skin photos → 5-fold cross-validation → Ensemble

#### Stack
PyTorch + timm + segmentation_models_pytorch + albumentations + GradCAM + Gradio

#### Features
- Classification (12 skin conditions)
- Segmentation (lesion boundary)
- GradCAM explainability
- ABCDE clinical feature extraction
- Multi-model ensemble with consensus

---
*For research and demonstration purposes only. Not for clinical use.*

*Models: {len(list(MODELS_DIR.glob('best_*.pth')))} single + {len(list(KFOLD_DIR.glob('kfold_*_fold*.pth')))} K-fold*
""")

        gr.HTML("""<p style="text-align:center;color:#9ca3af;font-size:11px;padding:8px;">
            For research and demonstration purposes only. Not for clinical use. | WhiteCoat.dev 2026
        </p>""")

    return app


if __name__ == "__main__":
    print("WhiteCoat.dev Platform v2")
    print(f"Device: {DEVICE}")
    print(f"Models: {list(MODELS_DIR.glob('best_*.pth'))}")
    print(f"K-fold: {list(KFOLD_DIR.glob('kfold_*_fold*.pth'))}")
    print(f"GradCAM: {HAS_GRADCAM} | Segmentation: {HAS_SMP}")
    build().launch(server_name="0.0.0.0", server_port=7860, share=False)
