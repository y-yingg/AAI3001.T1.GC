
# pages/2_Object Detection Model.py
import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights

st.set_page_config(page_title="Object Detection Model", page_icon="🛴", layout="wide")

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
# TRANSFORM = transforms.Compose([transforms.ToTensor()])

IMG_SIZE = 512  # same as in Colab

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


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
        # pretrained_backbone=False,
        weights=None,  # means no pretrained weights on the full model
        weights_backbone=None,  # replaces pretrained_backbone=False
        
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
        # text_w, text_h = draw.textbbox((xmin, ymin), caption, font=font)[2:]
        # draw.rectangle([xmin, ymin - text_h - 4, xmin + text_w + 4, ymin], fill=color)
        # draw.text((xmin + 2, ymin - text_h - 2), caption, fill="#000000", font=font)
        left, top, right, bottom = draw.textbbox((xmin, ymin), caption, font=font)
        text_w, text_h = right - left, bottom - top
        draw.rectangle(
            [xmin, ymin - text_h - 4, xmin + text_w + 4, ymin],
            fill=color
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

st.title("Object Detection")
st.write("Upload an image to detect riders on bicycles, kickboards, or motorcycles using the RetinaNet model stored on Hugging Face.")

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
        st.image(annotated, caption="Annotated output", use_column_width=True)

    with c2:
        st.subheader("Detection details")
        if table.empty:
            st.info("No detections above the selected threshold.")
        else:
            st.dataframe(table, hide_index=True, use_container_width=True)
else:
    st.info("Upload an image to start running inference.")
