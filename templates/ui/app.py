"""
WhiteCoat.dev — Medical AI Dashboard
Streamlit UI for medical radiograph AI analysis demo.

Run: streamlit run ui/app.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

st.set_page_config(page_title="WhiteCoat.dev — Medical AI", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem;
        background: linear-gradient(135deg, #1a365d, #2d5a87);
        color: white; border-radius: 10px; margin-bottom: 2rem; }
    .disclaimer { background-color: #fff3cd; border: 1px solid #ffc107;
        padding: 0.75rem; border-radius: 5px; text-align: center;
        font-weight: bold; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>WhiteCoat.dev</h1>
    <p>AI-Powered Medical Radiograph Analysis System</p>
    <p><small>AI in Healthcare Hackathon 2026 — Team #37</small></p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    For research and demonstration purposes only. Not for clinical use.
</div>
""", unsafe_allow_html=True)


def create_overlay(image, mask, alpha=0.4):
    """Create red overlay of segmentation mask on image."""
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = 255
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(
        image[mask_bool], 1 - alpha, colored_mask[mask_bool], alpha, 0
    )
    return overlay


with st.sidebar:
    st.header("Settings")
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Classification (Problem A)", "Segmentation (Problem B)", "Both"]
    )
    if "Segmentation" in analysis_mode or analysis_mode == "Both":
        seg_threshold = st.slider("Segmentation Threshold", 0.1, 0.9, 0.5, 0.05)
    else:
        seg_threshold = 0.5

    st.markdown("---")
    st.header("Team WhiteCoat.dev")
    st.markdown("- **Temur** — ML Lead + Medical Expert")
    st.markdown("- **Dilshoda** — ML Engineer")
    st.markdown("- **Muhammad** — Data Engineer + UI")
    st.markdown("- **Saida** — Presenter + Analyst")
    st.caption("AI in Healthcare Hackathon 2026\nCentral Asian University, Tashkent")


uploaded_file = st.file_uploader(
    "Upload a medical radiograph image",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image_rgb, caption=uploaded_file.name, use_container_width=True)
        st.caption(f"Size: {w}x{h} px")

    with col2:
        st.subheader("AI Analysis Results")

        if "Classification" in analysis_mode or analysis_mode == "Both":
            st.markdown("#### Problem A: Classification")
            model_path = ROOT / "checkpoints" / "best_classification.pth"
            if model_path.exists():
                try:
                    import torch
                    import timm
                    import albumentations as album
                    from albumentations.pytorch import ToTensorV2

                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                    cfg = checkpoint["config"]
                    model = timm.create_model(cfg["model_name"], pretrained=False, num_classes=cfg["num_classes"])
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.eval()

                    transform = album.Compose([
                        album.Resize(cfg.get("image_size", 384), cfg.get("image_size", 384)),
                        album.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ])
                    tensor = transform(image=image_rgb)["image"].unsqueeze(0)

                    with torch.no_grad():
                        probs = torch.softmax(model(tensor), dim=1).numpy()[0]

                    pred_class = int(np.argmax(probs))
                    confidence = float(probs[pred_class])
                    st.metric("Predicted Class", f"Class {pred_class}")
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.progress(confidence)
                    if confidence < 0.7:
                        st.warning("Low confidence — manual review recommended")
                except Exception as e:
                    st.error(f"Classification error: {e}")
            else:
                st.info("Classification model not found. Train the model first.")

        if "Segmentation" in analysis_mode or analysis_mode == "Both":
            st.markdown("#### Problem B: Segmentation")
            model_path = ROOT / "checkpoints" / "best_segmentation.pth"
            if model_path.exists():
                try:
                    import torch
                    import segmentation_models_pytorch as smp_lib
                    import albumentations as album
                    from albumentations.pytorch import ToTensorV2

                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                    cfg = checkpoint["config"]
                    model_registry = {"Unet": smp_lib.Unet, "UnetPlusPlus": smp_lib.UnetPlusPlus,
                                      "DeepLabV3Plus": smp_lib.DeepLabV3Plus, "FPN": smp_lib.FPN}
                    model = model_registry[cfg["model_name"]](
                        encoder_name=cfg["encoder_name"], encoder_weights=None,
                        classes=cfg["num_classes"],
                        activation="sigmoid" if cfg["num_classes"] == 1 else None,
                    )
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.eval()

                    img_size = cfg.get("image_size", 512)
                    transform = album.Compose([
                        album.Resize(img_size, img_size),
                        album.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(),
                    ])
                    tensor = transform(image=image_rgb)["image"].unsqueeze(0)

                    with torch.no_grad():
                        prob_mask = model(tensor).squeeze().numpy()

                    prob_mask = cv2.resize(prob_mask, (w, h))
                    binary_mask = (prob_mask > seg_threshold).astype(np.uint8) * 255
                    overlay = create_overlay(image_rgb, binary_mask)
                    st.image(overlay, caption="Segmentation Overlay", use_container_width=True)

                    coverage = (binary_mask > 0).sum() / binary_mask.size * 100
                    st.metric("Mask Coverage", f"{coverage:.1f}%")
                except Exception as e:
                    st.error(f"Segmentation error: {e}")
            else:
                st.info("Segmentation model not found. Train the model first.")
else:
    st.markdown("""
    ### How to use:
    1. Upload a medical radiograph image
    2. Select analysis mode from the sidebar
    3. View AI predictions and segmentation results

    **Technical Stack:** PyTorch, timm, segmentation_models_pytorch, albumentations, Streamlit
    """)

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#666;font-size:0.8rem;">'
    '<b>For research and demonstration purposes only. Not for clinical use.</b></p>',
    unsafe_allow_html=True,
)
