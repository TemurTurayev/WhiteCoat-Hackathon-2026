"""
WhiteCoat.dev — Skin Lesion AI Platform v3
============================================
Run: python webapp/app.py
Open: http://localhost:7860

Tabs: Classification | Segmentation | Model Comparison | About
Each tab has its own model selection and settings.
"""
import os
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
# PATHS
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

CLASS_INFO = {
    0:  {"name": "Actinic Keratosis",        "short": "AK",   "type": "Pre-malignant", "emoji": "🟡"},
    1:  {"name": "Basal Cell Carcinoma",      "short": "BCC",  "type": "Malignant",     "emoji": "🔴"},
    2:  {"name": "Dermatofibroma",            "short": "DF",   "type": "Benign",        "emoji": "🟢"},
    3:  {"name": "Hemangioma",                "short": "HEM",  "type": "Benign",        "emoji": "🟢"},
    4:  {"name": "Intraepithelial Carcinoma", "short": "IEC",  "type": "Malignant",     "emoji": "🔴"},
    5:  {"name": "Lentigo",                   "short": "LEN",  "type": "Benign",        "emoji": "🟢"},
    6:  {"name": "Melanoma",                  "short": "MEL",  "type": "Malignant",     "emoji": "🔴"},
    7:  {"name": "Melanocytic Nevus",         "short": "NV",   "type": "Benign",        "emoji": "🟢"},
    8:  {"name": "Pyogenic Granuloma",        "short": "PG",   "type": "Benign",        "emoji": "🟢"},
    9:  {"name": "Seborrheic Keratosis",      "short": "SK",   "type": "Benign",        "emoji": "🟢"},
    10: {"name": "Squamous Cell Carcinoma",   "short": "SCC",  "type": "Malignant",     "emoji": "🔴"},
    11: {"name": "Wart",                      "short": "WART", "type": "Benign",        "emoji": "🟢"},
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

RISK_COLORS = {"Malignant": "#ef4444", "Pre-malignant": "#f59e0b", "Benign": "#22c55e"}
IMAGE_SIZE = 224

# ============================================================
# MODEL CACHE
# ============================================================
_cache = {}


def _load_single(path, model_name=None):
    key = str(path)
    if key in _cache:
        return _cache[key]
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    # Handle both formats: checkpoint dict with model_state_dict, or raw state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        if model_name is None:
            model_name = ckpt.get("config", {}).get("model_name", "tf_efficientnetv2_s")
        acc = ckpt.get("val_acc", 0)
    else:
        sd = ckpt
        acc = 0
    if model_name is None:
        model_name = "tf_efficientnetv2_s"
    m = timm.create_model(model_name, pretrained=False, num_classes=12)
    m.load_state_dict(sd)
    m.to(DEVICE).train(False)
    _cache[key] = (m, model_name, acc)
    return _cache[key]


def _load_kfold(base_name):
    key = f"kf_{base_name}"
    if key in _cache:
        return _cache[key]
    model_name = base_name.replace("kfold_", "")
    models = []
    for fold in range(5):
        p = KFOLD_DIR / f"{base_name}_fold{fold}.pth"
        if p.exists():
            m, _, _ = _load_single(p, model_name)
            models.append(m)
    _cache[key] = models
    return models


def _load_seg(arch_key):
    if arch_key in _cache:
        return _cache[arch_key]
    if not HAS_SMP:
        return None
    # Try multiple path patterns
    path = MODELS_DIR / f"{arch_key}.pth"
    if not path.exists():
        path = MODELS_DIR / f"best_{arch_key}.pth"
    if not path.exists():
        return None

    print(f"Loading segmentation model: {path.name}")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    # Handle both formats: raw state_dict or checkpoint with config
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        cfg = ckpt.get("config", {})
        arch = cfg.get("model_name", cfg.get("arch", "UnetPlusPlus"))
        encoder = cfg.get("encoder_name", cfg.get("encoder", "efficientnet-b4"))
        sd = ckpt["model_state_dict"]
    else:
        # Raw state_dict — all our best seg models use UnetPlusPlus + EfficientNetV2-S
        arch = "UnetPlusPlus"
        encoder = "tu-tf_efficientnetv2_s"
        sd = ckpt

    m = getattr(smp, arch)(encoder_name=encoder, encoder_weights=None, classes=1, activation=None)
    m.load_state_dict(sd)
    m.to(DEVICE).train(False)
    _cache[arch_key] = m
    print(f"  Loaded: {arch} / {encoder}")
    return m


# ============================================================
# AVAILABLE MODELS
# ============================================================
def get_cls_models():
    """Top 3 classification models only."""
    top3 = [
        "tf_efficientnetv2_s",
        "swin_tiny_patch4_window7_224",
        "convnext_tiny",
    ]
    models = []
    for name in top3:
        if (MODELS_DIR / f"best_{name}.pth").exists():
            models.append(f"Single: {name}")
    models.append("Ensemble: Top 3")
    return models


def get_seg_models():
    """Only the 2 best models for testing."""
    models = []
    # 1. Mega512 Lovász — best single model (IoU=0.8960 on holdout)
    if (MODELS_DIR / "best_seg_mega512_lovasz.pth").exists():
        models.append("best_seg_mega512_lovasz")
    # 2. DGX Soup 5-fold — best ensemble soup (IoU=0.8949 on holdout)
    if (MODELS_DIR / "dgx_soup_5fold.pth").exists():
        models.append("dgx_soup_5fold")
    return models if models else ["best_seg_mega512_lovasz"]


# ============================================================
# TRANSFORMS
# ============================================================
val_tf = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), ToTensorV2()])
seg_tf = A.Compose([A.Resize(512, 512), A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]), ToTensorV2()])


def _to_tensor(img, tf=val_tf):
    return tf(image=img)["image"].unsqueeze(0).to(DEVICE)


def _predict_single(model, tensor):
    with torch.no_grad():
        return F.softmax(model(tensor), dim=1)[0].cpu().numpy()


def _predict_ensemble(models, tensor):
    probs = np.zeros(12)
    for m in models:
        probs += _predict_single(m, tensor)
    return probs / len(models)


def _map_probs(raw):
    return {rc: float(raw[mi]) for mi, rc in MODEL_IDX_TO_REAL.items()}


def _top_pred(raw):
    mi = int(np.argmax(raw))
    return MODEL_IDX_TO_REAL[mi], float(raw[mi]), mi


# ============================================================
# GRADCAM
# ============================================================
def _get_tl(model, name):
    if "efficientnet" in name:
        return [model.blocks[-1]] if hasattr(model, "blocks") else [model.features[-1]]
    if "convnext" in name:
        return [model.stages[-1].blocks[-1]]
    if "swin" in name:
        return [model.layers[-1].blocks[-1].norm1]
    if "resnet" in name:
        return [model.layer4[-1]]
    return None


def _gradcam(model, name, img, target_mi):
    if not HAS_GRADCAM:
        return None
    tl = _get_tl(model, name)
    if not tl:
        return None
    try:
        r = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        f = r.astype(np.float32) / 255.0
        t = _to_tensor(img)
        cam = GradCAM(model=model, target_layers=tl)
        gc = cam(input_tensor=t, targets=[ClassifierOutputTarget(target_mi)])[0]
        return show_cam_on_image(f, gc, use_rgb=True)
    except Exception:
        return None


# ============================================================
# ABCDE
# ============================================================
def _abcde(mask, img):
    if mask.sum() < 10:
        return {"A": 0, "B": 0, "C": 0, "D": 0, "risk": 0, "level": "N/A"}
    h, w = mask.shape
    left, right = mask[:, :w//2], np.flip(mask[:, w//2:w//2*2], axis=1)
    mw = min(left.shape[1], right.shape[1])
    asym = np.abs(left[:,:mw].astype(float) - right[:,:mw].astype(float)).sum()
    asym /= max(left[:,:mw].sum() + right[:,:mw].sum(), 1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    irreg, diam = 0, 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
        irreg = 1 - (4*np.pi*area)/(perim*perim+1e-6)
        pts = c.reshape(-1,2)
        if len(pts) > 2:
            from scipy.spatial.distance import pdist
            diam = pdist(pts).max()

    masked = img[mask==1]
    cvar = min(masked.std(axis=0).mean()/80, 1.0) if len(masked) > 0 else 0
    risk = asym*0.3 + irreg*0.3 + cvar*0.2 + min(diam/100,1)*0.2
    level = "LOW" if risk < 0.3 else "MODERATE" if risk < 0.6 else "HIGH"
    return {"A": asym, "B": irreg, "C": cvar, "D": diam, "risk": risk, "level": level}


def _abcde_html(ab):
    rc = "#22c55e" if ab["level"]=="LOW" else "#f59e0b" if ab["level"]=="MODERATE" else "#ef4444"
    def bar(val, thresh, mx=100):
        c = "#ef4444" if val > thresh else "#22c55e"
        w = min(val/mx*100 if mx != 100 else val*100, 100)
        return f'<div style="background:#e5e7eb;border-radius:4px;height:6px;margin-top:4px;"><div style="background:{c};width:{w:.0f}%;height:6px;border-radius:4px;"></div></div>'

    return f"""
    <div style="font-family:system-ui;padding:16px;border-radius:12px;background:#f9fafb;border:1px solid #e5e7eb;">
        <h3 style="margin:0 0 12px;color:#1f2937;">ABCDE Clinical Features</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
            <div style="padding:8px;background:white;border-radius:8px;border:1px solid #e5e7eb;">
                <b>A</b> Asymmetry: <span style="float:right;">{ab['A']:.2f}</span>{bar(ab['A'], 0.3, 1)}
            </div>
            <div style="padding:8px;background:white;border-radius:8px;border:1px solid #e5e7eb;">
                <b>B</b> Border Irregularity: <span style="float:right;">{ab['B']:.2f}</span>{bar(ab['B'], 0.4, 1)}
            </div>
            <div style="padding:8px;background:white;border-radius:8px;border:1px solid #e5e7eb;">
                <b>C</b> Color Variation: <span style="float:right;">{ab['C']:.2f}</span>{bar(ab['C'], 0.5, 1)}
            </div>
            <div style="padding:8px;background:white;border-radius:8px;border:1px solid #e5e7eb;">
                <b>D</b> Diameter: <span style="float:right;">{ab['D']:.0f}px</span>{bar(ab['D'], 60, 150)}
            </div>
        </div>
        <div style="margin-top:12px;padding:10px;background:{rc}15;border:1px solid {rc};border-radius:8px;text-align:center;">
            <b style="color:{rc};font-size:16px;">Risk: {ab['level']} ({ab['risk']:.2f})</b>
        </div>
    </div>"""


def _risk_html(rc, conf, info):
    tc = RISK_COLORS[info["type"]]
    return f"""
    <div style="font-family:system-ui;padding:20px;border-radius:16px;background:linear-gradient(135deg,{tc}08,{tc}15);border:2px solid {tc};">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
            <span style="font-size:36px;">{info['emoji']}</span>
            <div>
                <h2 style="margin:0;color:{tc};font-size:22px;">{info['name']}</h2>
                <span style="background:{tc};color:white;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;">{info['type'].upper()}</span>
            </div>
        </div>
        <div style="background:white;border-radius:12px;padding:12px;margin:8px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:#6b7280;">Confidence</span>
                <span style="font-size:24px;font-weight:700;color:{tc};">{conf:.1%}</span>
            </div>
            <div style="background:#e5e7eb;border-radius:6px;height:8px;margin-top:6px;">
                <div style="background:{tc};width:{conf*100:.0f}%;height:8px;border-radius:6px;"></div>
            </div>
        </div>
        <p style="color:#6b7280;font-size:13px;margin:8px 0 0;">{CLASS_DESCRIPTIONS[rc]}</p>
        <p style="color:#9ca3af;font-size:10px;margin:12px 0 0;text-align:center;">For research and demonstration purposes only. Not for clinical use.</p>
    </div>"""


# ============================================================
# CLASSIFICATION TAB
# ============================================================
def classify(image, model_choice, show_gradcam, top_k):
    if image is None:
        return {}, None, ""

    img = np.array(image)
    tensor = _to_tensor(img)
    top_k = int(top_k)

    if model_choice.startswith("Single:"):
        name = model_choice.replace("Single: ", "")
        m, mn, _ = _load_single(MODELS_DIR / f"best_{name}.pth")
        raw = _predict_single(m, tensor)
        gcm, gcn = m, mn
    elif model_choice.startswith("K-Fold"):
        name = model_choice.split(": ")[1]
        models = _load_kfold(f"kfold_{name}")
        raw = _predict_ensemble(models, tensor) if models else np.zeros(12)
        gcm, gcn = (models[0], name) if models else (None, "")
    elif model_choice == "Ensemble: Top 3":
        all_p = []
        for name in ["tf_efficientnetv2_s", "swin_tiny_patch4_window7_224", "convnext_tiny"]:
            p = MODELS_DIR / f"best_{name}.pth"
            if p.exists():
                m, _, _ = _load_single(p)
                all_p.append(_predict_single(m, tensor))
        raw = np.mean(all_p, axis=0) if all_p else np.zeros(12)
        m0, mn0, _ = _load_single(MODELS_DIR / "best_tf_efficientnetv2_s.pth")
        gcm, gcn = m0, mn0
    else:
        return {}, None, ""

    real_probs = _map_probs(raw)
    sorted_probs = sorted(real_probs.items(), key=lambda x: -x[1])[:top_k]
    label_dict = {f"[{rc}] {CLASS_INFO[rc]['name']}": prob for rc, prob in sorted_probs}

    rc, conf, mi = _top_pred(raw)
    info = CLASS_INFO[rc]

    gc_img = _gradcam(gcm, gcn, img, mi) if show_gradcam and gcm else None
    risk = _risk_html(rc, conf, info)

    return label_dict, gc_img, risk


# ============================================================
# SEGMENTATION TAB
# ============================================================
def segment(image, seg_model_choice, threshold, overlay_opacity, show_probability):
    if image is None:
        return None, None, None, ""

    arch_key = seg_model_choice
    model = _load_seg(arch_key)
    if model is None:
        return None, None, None, "<p>Model not found</p>"

    img = np.array(image)
    orig_h, orig_w = img.shape[:2]
    tensor = _to_tensor(img, seg_tf)

    with torch.no_grad():
        pred = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

    # Resize prediction back to original image size for accurate display
    pred_full = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask_bin = (pred_full > threshold).astype(np.uint8)

    # Overlay on original image
    overlay = img.copy()
    overlay[mask_bin == 1] = [255, 50, 50]
    alpha = overlay_opacity / 100.0
    blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

    # B&W binary mask at original resolution
    bw_mask = (mask_bin * 255).astype(np.uint8)
    bw_mask_rgb = cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2RGB)

    # Probability heatmap
    prob_img = None
    if show_probability:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(pred_full, cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"Probability Map (threshold={threshold})", fontsize=11)
        ax.axis("off")
        plt.colorbar(ax.images[0], ax=ax, fraction=0.046)
        plt.tight_layout()
        fig.canvas.draw()
        buf = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        prob_img = buf.copy()
        plt.close()

    ab = _abcde(mask_bin, img)
    ab_html = _abcde_html(ab)

    # Add IoU placeholder and mask stats
    mask_area = mask_bin.sum()
    total_area = mask_bin.shape[0] * mask_bin.shape[1]
    pct = mask_area / total_area * 100
    ab_html += f"""
    <div style="font-family:system-ui;padding:10px;margin-top:8px;background:#f0f9ff;border-radius:8px;border:1px solid #bae6fd;">
        <b>Mask Stats:</b> {mask_area:,} px ({pct:.1f}% of image) | Threshold: {threshold}
    </div>"""

    return blended, bw_mask_rgb, prob_img, ab_html


# ============================================================
# COMPARISON TAB
# ============================================================
def compare_all(image):
    if image is None:
        return ""
    img = np.array(image)
    tensor = _to_tensor(img)

    rows = []
    for f in sorted(MODELS_DIR.glob("best_*.pth")):
        n = f.stem.replace("best_", "")
        if any(s in n for s in ["Unet", "DeepLab"]):
            continue
        try:
            m, mn, vacc = _load_single(f)
            raw = _predict_single(m, tensor)
            rc, conf, _ = _top_pred(raw)
            rows.append({"model": mn, "type": "Single", "pred_rc": rc, "conf": conf, "vacc": vacc})
        except Exception:
            pass

    for base in ["kfold_tf_efficientnetv2_s", "kfold_swin_tiny_patch4_window7_224", "kfold_convnext_tiny"]:
        models = _load_kfold(base)
        if models:
            raw = _predict_ensemble(models, tensor)
            rc, conf, _ = _top_pred(raw)
            rows.append({"model": f"{base.replace('kfold_','')}", "type": f"K-Fold ({len(models)}x)", "pred_rc": rc, "conf": conf, "vacc": 0})

    if not rows:
        return "<p>No models</p>"

    rows.sort(key=lambda x: -x["conf"])
    html = '<table style="width:100%;border-collapse:collapse;font-family:system-ui;font-size:14px;">'
    html += '<tr style="background:#f3f4f6;"><th style="padding:10px;text-align:left;">Model</th><th>Type</th><th>Prediction</th><th>Confidence</th><th>Risk</th></tr>'

    for r in rows:
        info = CLASS_INFO[r["pred_rc"]]
        tc = RISK_COLORS[info["type"]]
        bar = f'<div style="background:#e5e7eb;border-radius:4px;width:120px;height:14px;display:inline-block;vertical-align:middle;"><div style="background:{tc};width:{r["conf"]*100:.0f}%;height:14px;border-radius:4px;"></div></div>'
        tb = "K-Fold" if "K-Fold" in r["type"] else "Single"
        bg = "#dbeafe" if tb == "K-Fold" else "#f3f4f6"
        html += f'<tr style="border-bottom:1px solid #e5e7eb;"><td style="padding:8px;font-weight:500;">{r["model"]}</td>'
        html += f'<td style="padding:8px;text-align:center;"><span style="background:{bg};padding:2px 8px;border-radius:8px;font-size:12px;">{r["type"]}</span></td>'
        html += f'<td style="padding:8px;font-weight:600;">{info["emoji"]} {info["name"]}</td>'
        html += f'<td style="padding:8px;text-align:center;">{bar} {r["conf"]:.1%}</td>'
        html += f'<td style="padding:8px;text-align:center;color:{tc};font-weight:600;">{info["type"]}</td></tr>'
    html += "</table>"

    preds = [r["pred_rc"] for r in rows]
    top = Counter(preds).most_common(1)[0]
    ci = CLASS_INFO[top[0]]
    tc = RISK_COLORS[ci["type"]]
    agr = top[1] / len(preds)
    html += f'<div style="margin-top:12px;padding:14px;background:{tc}10;border:2px solid {tc};border-radius:12px;text-align:center;">'
    html += f'<span style="font-size:20px;">{ci["emoji"]}</span> <b style="font-size:16px;color:{tc};">Consensus: {ci["name"]}</b>'
    html += f' <span style="color:#6b7280;">({agr:.0%} agreement, {len(preds)} models)</span></div>'
    return html


# ============================================================
# BUILD APP
# ============================================================
def build():
    cls_models = get_cls_models()
    seg_models = get_seg_models()

    with gr.Blocks(title="WhiteCoat.dev") as app:

        gr.HTML("""<div style="text-align:center;padding:20px 0 10px;">
            <h1 style="font-family:system-ui;font-size:28px;font-weight:700;color:#1e3a5f;margin:0;">WhiteCoat.dev</h1>
            <p style="color:#6b7280;font-size:14px;margin:4px 0 0;">AI-Powered Skin Lesion Analysis | Classification + Segmentation + Explainability</p>
        </div>""")

        with gr.Tabs():
            # ========== CLASSIFICATION ==========
            with gr.Tab("Classification"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        cls_img = gr.Image(type="numpy", label="Upload Skin Lesion", height=300)
                        cls_model = gr.Dropdown(cls_models, value=cls_models[-1] if cls_models else "", label="Model")
                        with gr.Row():
                            cls_gradcam = gr.Checkbox(value=True, label="Show GradCAM")
                            cls_topk = gr.Slider(3, 12, value=5, step=1, label="Top-K Classes")
                        cls_btn = gr.Button("Classify", variant="primary", size="lg")
                    with gr.Column(scale=3):
                        cls_risk = gr.HTML()
                        cls_probs = gr.Label(num_top_classes=5, label="Predictions")
                with gr.Row():
                    cls_gc = gr.Image(type="numpy", label="GradCAM - Where the Model Looks", height=300)

                cls_btn.click(classify, [cls_img, cls_model, cls_gradcam, cls_topk], [cls_probs, cls_gc, cls_risk])

                if (BASE_DIR / "data/classification/train").exists():
                    examples = [str(f) for f in sorted((BASE_DIR / "data/classification/train").glob("*/0.png"))[:6]]
                    if examples:
                        gr.Examples(examples=[[e] for e in examples], inputs=[cls_img], label="Examples")

            # ========== SEGMENTATION ==========
            with gr.Tab("Segmentation"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        seg_img = gr.Image(type="numpy", label="Upload Skin Lesion", height=300)
                        seg_labels = {
                            "best_seg_mega512_lovasz": "Mega512 Lovász (IoU=0.8960, BEST)",
                            "dgx_soup_5fold": "DGX Soup 5-Fold (IoU=0.8949)",
                        }
                        seg_choices = [(seg_labels.get(m, m), m) for m in seg_models]
                        seg_model = gr.Dropdown(choices=seg_choices, value=seg_models[0] if seg_models else "", label="Segmentation Model")
                        seg_thresh = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Threshold")
                        seg_opacity = gr.Slider(10, 90, value=40, step=5, label="Overlay Opacity %")
                        seg_showprob = gr.Checkbox(value=True, label="Show Probability Heatmap")
                        seg_btn = gr.Button("Segment", variant="primary", size="lg")
                    with gr.Column(scale=3):
                        seg_abcde = gr.HTML()
                with gr.Row():
                    seg_overlay = gr.Image(type="numpy", label="Segmentation Overlay", height=280)
                    seg_bwmask = gr.Image(type="numpy", label="B&W Binary Mask", height=280)
                    seg_heatmap = gr.Image(type="numpy", label="Probability Heatmap", height=280)

                seg_btn.click(segment, [seg_img, seg_model, seg_thresh, seg_opacity, seg_showprob],
                              [seg_overlay, seg_bwmask, seg_heatmap, seg_abcde])

                if (BASE_DIR / "data/Segmentation/testing/images").exists():
                    seg_ex = [str(f) for f in sorted((BASE_DIR / "data/Segmentation/testing/images").glob("*.*"))[:4]]
                    if seg_ex:
                        gr.Examples(examples=[[e] for e in seg_ex], inputs=[seg_img], label="Examples")

            # ========== COMPARISON ==========
            with gr.Tab("Model Comparison"):
                gr.Markdown("### Compare ALL models on one image — see consensus voting")
                with gr.Row():
                    cmp_img = gr.Image(type="numpy", label="Upload Image", height=250)
                    cmp_btn = gr.Button("Compare All Models", variant="primary", size="lg")
                cmp_out = gr.HTML()
                cmp_btn.click(compare_all, [cmp_img], [cmp_out])

            # ========== ABOUT ==========
            with gr.Tab("About"):
                gr.Markdown("""
### WhiteCoat.dev | AI in Healthcare Hackathon 2026
**Team #37** | Central Asian University, Tashkent

| Architecture | Task | K-Fold AVG |
|---|---|---|
| EfficientNetV2-S | Classification | **88.3%** |
| Swin-Tiny | Classification | **87.9%** |
| ConvNeXt-Tiny | Classification | **85.1%** |
| U-Net++ / EfficientNet-B4 | Segmentation | **81.0% IoU** |
| DeepLabV3+ / ResNet50 | Segmentation | **80.5% IoU** |

**Pipeline:** ImageNet pretrained -> Fine-tuned on 11,411 clinical skin photos -> 5-fold CV -> Ensemble

**Stack:** PyTorch + timm + segmentation_models_pytorch + albumentations + GradCAM + Gradio

---
*For research and demonstration purposes only. Not for clinical use.*
""")

        gr.HTML('<p style="text-align:center;color:#9ca3af;font-size:11px;padding:8px;">For research and demonstration purposes only. Not for clinical use. | WhiteCoat.dev 2026</p>')
    return app


if __name__ == "__main__":
    print("WhiteCoat.dev Platform v3")
    print(f"Device: {DEVICE}")
    print(f"Classification models: {get_cls_models()}")
    print(f"Segmentation models: {get_seg_models()}")
    print(f"GradCAM: {HAS_GRADCAM} | SMP: {HAS_SMP}")
    build().launch(server_name="0.0.0.0", server_port=7860, share=False,
                    css=".gradio-container{max-width:1400px!important;margin:auto;}footer{display:none!important;}")
