# pages/2_Object Detection Model.py
import io
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models.detection import retinanet_resnet50_fpn

# Grad-CAM imports (ipynb-style)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.base_cam import BaseCAM  # <-- add this import


# --- PATCH: make BaseCAM.__del__ safe so it doesn't crash on cleanup ---
def _safe_basecam_del(self):
    # Only call release() if the attribute actually exists
    if hasattr(self, "activations_and_grads") and self.activations_and_grads is not None:
        self.activations_and_grads.release()

BaseCAM.__del__ = _safe_basecam_del
# ----------------------------------------------------------------------


st.set_page_config(page_title="Object Detection Model", page_icon="🛴", layout="wide")

# ----- Simple blue section title style (same vibe as previous project) -----
st.markdown(
    """
<style>
.section-title-blue{
  color:#2d6cdf;
  font-weight:700;
  font-size:1.15rem;
  margin: 6px 0 10px 0;
}
hr{border-color:rgba(255,255,255,0.12);}
</style>
""",
    unsafe_allow_html=True,
)

CLASSES = [
    "Person_riding_bycycle",
    "Person_riding_kickboard",
    "Person_riding_motorcycle",
]
CLASS_COLORS = {
    "Person_riding_bycycle": "#3ddc97",
    "Person_riding_kickboard": "#ff9f1c",
    "Person_riding_motorcycle": "#2ec4b6",
}
HF_REPO_ID = "y-yingg/pedestrian_detection"
HF_FILENAME = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 512  # same as in Colab

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# validation directory for examples section
VAL_DIR = Path("data-object_detection/validation")  # corresponds to ...\\data-object_detection\\validation


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


@st.cache_resource(show_spinner=False)
def load_detector() -> torch.nn.Module:
    ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    model = retinanet_resnet50_fpn(
        num_classes=len(CLASSES) + 1,  # +1 background, matches training script
        weights=None,                  # no pretrained detection weights
        weights_backbone=None,
    )
    model.load_state_dict(_load_state_dict(ckpt_path))
    model.to(DEVICE)
    model.eval()
    return model


def predict(image: Image.Image, score_threshold: float):
    model = load_detector()
    tensor = TRANSFORM(image).to(DEVICE)
    with torch.no_grad():
        outputs = model([tensor])[0]

    keep = outputs["scores"] >= score_threshold
    boxes = outputs["boxes"][keep].cpu().numpy()
    labels = outputs["labels"][keep].cpu().tolist()
    scores = outputs["scores"][keep].cpu().numpy()
    return boxes, labels, scores


def draw_detections(image: Image.Image, boxes, labels, scores):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    rows = []
    for (xmin, ymin, xmax, ymax), label_idx, score in zip(boxes, labels, scores):
        class_idx = max(label_idx - 1, 0)
        class_name = CLASSES[class_idx] if class_idx < len(CLASSES) else f"class {label_idx}"
        color = CLASS_COLORS.get(class_name, "#ff595e")
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        caption = f"{class_name} {score:.2f}"

        left, top, right, bottom = draw.textbbox((xmin, ymin), caption, font=font)
        text_w, text_h = right - left, bottom - top
        draw.rectangle(
            [xmin, ymin - text_h - 4, xmin + text_w + 4, ymin],
            fill=color,
        )
        draw.text((xmin + 2, ymin - text_h - 2), caption, fill="#000000", font=font)

        rows.append(
            {
                "Class": class_name,
                "Confidence": f"{score:.3f}",
                "x_min": int(xmin),
                "y_min": int(ymin),
                "x_max": int(xmax),
                "y_max": int(ymax),
                "Box Area": int((xmax - xmin) * (ymax - ymin)),
            }
        )
    df = pd.DataFrame(rows)
    return annotated, df

# -------------------------------------------------------------------
# Grad-CAM helpers (ipynb-style)
# -------------------------------------------------------------------
class DetectionIndexTarget:
    """Target that returns a specific detection score scalar."""
    def __init__(self, score_tensor: torch.Tensor):
        self.score_tensor = score_tensor

    def __call__(self, model_output):
        return self.score_tensor


@st.cache_resource(show_spinner=False)
def get_cam() -> GradCAM:
    """
    Create and cache a single GradCAM object.
    This avoids repeated construction and reduces destructor warnings.
    """
    model = load_detector()
    target_layers = [model.backbone.body.layer4]  # same as in ipynb

    # IMPORTANT: no 'use_cuda' kw here – GradCAM will use whatever device
    # the model and input tensors are on.
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
    )
    return cam



def generate_cam_for_image(pil_img: Image.Image, score_threshold: float = 0.5) -> Image.Image:
    """
    True Grad-CAM (feature-map based), matching the ipynb logic:
    - Hook layer4 of the ResNet50 backbone.
    - Use the top detection's score as the CAM target.
    - Overlay heatmap on the original image with show_cam_on_image.
    """
    try:
        # Re-use cached model & CAM object
        model = load_detector()
        cam = get_cam()

        # Resize to training size
        img_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))

        # For overlay: normalised to 0-1 float RGB
        rgb_img = np.array(img_resized).astype(np.float32) / 255.0

        # For model: normalised tensor with grad
        img_tensor = TRANSFORM(img_resized).to(DEVICE)
        img_tensor.requires_grad_(True)
        input_batch = img_tensor.unsqueeze(0)  # (1,3,H,W)

        # Forward WITH gradients (no torch.no_grad here)
        preds = model(input_batch)
        pred = preds[0]
        scores = pred["scores"]
        boxes = pred["boxes"]
        labels = pred["labels"]

        if scores.numel() == 0:
            return img_resized

        # Prefer highest-scoring detection above threshold, else top-1
        keep = scores >= score_threshold
        if keep.any():
            kept_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
            local_top = torch.argmax(scores[keep]).item()
            top_idx = int(kept_idx[local_top].item())
        else:
            top_idx = int(torch.argmax(scores).item())

        score_tensor = scores[top_idx]
        box = boxes[top_idx].detach().cpu().numpy()
        label_idx = int(labels[top_idx].item())

        # Build Grad-CAM target
        target = [DetectionIndexTarget(score_tensor)]
        grayscale_cam = cam(input_tensor=input_batch, targets=target)[0]

        # --- Manual re-normalisation so map actually spans [0,1] ---
        grayscale_cam = grayscale_cam - grayscale_cam.min()
        max_val = grayscale_cam.max()
        if max_val > 0:
            grayscale_cam = grayscale_cam / max_val
        else:
            # completely flat -> just show original image
            return img_resized

        # Overlay CAM on image. image_weight small -> heatmap very obvious.
        grayscale_cam = 1.0 - grayscale_cam
        cam_image = show_cam_on_image(
            rgb_img,
            grayscale_cam,
            use_rgb=True,
            image_weight=0.15,   # 0.15 image + 0.85 heatmap
        )
        cam_pil = Image.fromarray((cam_image * 255).astype(np.uint8))

        # Draw same detection box + label on CAM image
        class_idx = max(label_idx - 1, 0)
        class_name = CLASSES[class_idx] if class_idx < len(CLASSES) else f"class {label_idx}"
        color = CLASS_COLORS.get(class_name, "#ff595e")

        xmin, ymin, xmax, ymax = box
        draw = ImageDraw.Draw(cam_pil)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        caption = f"{class_name} {float(score_tensor):.2f}"
        font = ImageFont.load_default()
        left, top, right, bottom = draw.textbbox((xmin, ymin), caption, font=font)
        text_w, text_h = right - left, bottom - top
        draw.rectangle([xmin, ymin - text_h - 4, xmin + text_w + 4, ymin], fill=color)
        draw.text((xmin + 2, ymin - text_h - 2), caption, fill="#000000", font=font)

        return cam_pil

    except Exception as e:
        # If anything goes wrong, fall back to just the resized input
        # (and optionally print the error in terminal for debugging)
        print("Grad-CAM error:", e)
        return pil_img.resize((IMG_SIZE, IMG_SIZE))


# -------------------------------------------------------------------
# Helper functions for the "View Examples" section
# -------------------------------------------------------------------
def list_validation_images(max_samples: int = 6):
    """Return up to max_samples JPG images from the validation folder."""
    if not VAL_DIR.exists():
        return []
    all_imgs = sorted(VAL_DIR.glob("*.jpg"))
    if len(all_imgs) <= max_samples:
        return all_imgs
    return random.sample(all_imgs, max_samples)


def get_random_detection_examples(score_threshold: float = 0.5):
    """
    Pick random validation images, run detection, and return list of
    (filename, annotated_image, cam_image, df) tuples.
    """
    paths = list_validation_images(6)
    examples = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            boxes, labels, scores = predict(img, score_threshold)
            annotated, df = draw_detections(img, boxes, labels, scores)
            cam_img = generate_cam_for_image(img, score_threshold)
            examples.append((path.name, annotated, cam_img, df))
        except Exception:
            continue
    return examples


# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------
st.title("Object Detection")
st.write(
    "Upload an image to detect riders on bicycles, kickboards, or motorcycles "
    "using the RetinaNet model stored on Hugging Face."
)

uploaded_file = st.file_uploader("Drag & drop an image", type=["jpg", "jpeg", "png"])
score_threshold = st.slider("Score threshold", 0.1, 0.9, value=0.5, step=0.05)

if uploaded_file:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    # resize to the same size used in training / validation
    image = image.resize((IMG_SIZE, IMG_SIZE))

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Model predictions")
        boxes, labels, scores = predict(image, score_threshold)
        annotated, table = draw_detections(image, boxes, labels, scores)
        st.image(annotated, caption="Annotated output", width=500)

    with c2:
        st.subheader("Detection details")
        if table.empty:
            st.info("No detections above the selected threshold.")
        else:
            st.dataframe(table, hide_index=True, use_container_width=True)

else:
    st.info("Upload an image to start running inference.")

# -------------------------------------------------------------------
# View Examples section (similar to previous classification project)
# -------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div class="section-title-blue">View Examples (RetinaNet Object Detection)</div>',
    unsafe_allow_html=True,
)

example_threshold = st.slider(
    "Score Threshold for Example Detections",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Higher threshold = fewer, more confident boxes",
    key="example_threshold",
)

if "det_examples" not in st.session_state:
    st.session_state.det_examples = None

col_run, col_refresh = st.columns([1, 3])
with col_run:
    if st.button("🚀 Run Model", type="primary"):
        with st.spinner("Running model on validation images..."):
            st.session_state.det_examples = get_random_detection_examples(example_threshold)
        st.rerun()

with col_refresh:
    if st.session_state.det_examples is not None:
        if st.button("🔄 Refresh Examples"):
            with st.spinner("Getting new images..."):
                st.session_state.det_examples = get_random_detection_examples(example_threshold)
            st.rerun()

if st.session_state.det_examples:
    # ONE example per row: left = annotated, right = CAM, below = detection table
    for fname, annotated_img, cam_img, df in st.session_state.det_examples:
        with st.container():
            cols = st.columns(2)
            with cols[0]:
                st.image(annotated_img, caption=f"{fname} — Annotated", use_column_width=True)
            with cols[1]:
                st.image(cam_img, caption="Grad-CAM visualization", use_column_width=True)

            if df is None or df.empty:
                st.caption("No detections above the selected threshold.")
            else:
                st.dataframe(df, hide_index=True, use_container_width=True)

        st.markdown("---")
else:
    st.info("Click 'Run Model' to see detection examples from the validation set!")
