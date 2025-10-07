from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision import transforms
import random
from utils import denormalize, CsvImageDataset
import requests
import io
import smtplib
from email.message import EmailMessage
from typing import Optional

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Classification (Best Resnet Model)", page_icon="🖼️", layout="wide")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
VAL_DIR   = Path("data/validation")
VAL_CSV   = VAL_DIR / "labels.csv"
MODEL_PATH = "models/best_resnet18_pedestrian.pt"   # <- Best model here

DATALOADER_KW = dict(
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)

# ---------- Email (Gmail) config ----------
SENDER_EMAIL = "sitprojects2024@gmail.com"
SENDER_APP_PASSWORD = "hglikztdmngldhvf"

# ---------- Page CSS ----------
st.markdown("""
<style>
.section-title-blue{
  color:#2d6cdf;
  font-weight:700;
  font-size:1.15rem;
  margin: 6px 0 10px 0;
}
hr{border-color:rgba(255,255,255,0.12);}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Telegram helpers
# -----------------------------
def send_telegram_message(bot_token: str, chat_id: str, message: str):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        requests.post(url, data=payload, timeout=15)
        return True
    except Exception:
        return False

def send_telegram_photo(bot_token: str, chat_id: str, caption: str, image: Image.Image):
    apology_msg = caption + " Sorry, the file size is too large for telegram so it could not be send here."
    try:
        import sys
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        size_bytes = sys.getsizeof(buf.getvalue())
        max_size = 5 * 1024 * 1024
        if size_bytes > max_size:
            send_telegram_message(bot_token, chat_id, apology_msg)
            return True
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        data = {"chat_id": chat_id, "caption": caption}
        files = {"photo": ("image.jpg", buf, "image/jpeg")}
        resp = requests.post(url, data=data, files=files, timeout=30)
        if resp.status_code != 200:
            send_telegram_message(bot_token, chat_id, apology_msg)
        return True
    except Exception:
        send_telegram_message(bot_token, chat_id, apology_msg)
        return True

# -----------------------------
# Email helpers
# -----------------------------
def send_email_alert(to_email: str, subject: str, body: str, image: Optional[Image.Image], attach_image: bool) -> bool:
    try:
        msg = EmailMessage()
        msg["From"] = SENDER_EMAIL
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)
        if attach_image and image is not None:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=90)
            img_bytes = buf.getvalue()
            msg.add_attachment(img_bytes, maintype="image", subtype="jpeg", filename="detected.jpg")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print(f"Email error: {e}")
    return False

def label_is_pedestrian(label_text: str) -> bool:
    return "pedestrian" in label_text.lower()

# -----------------------------
# Inference helpers
# -----------------------------

def get_model_predictions(model, val_loader, class_names, samples_per_class=3, device=None,
                          use_tta=False, threshold=0.5, pedestrian_class_idx=1):
    """
    Get model predictions with optional TTA and threshold tuning.

    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        class_names: List of class names
        samples_per_class: Number of samples to collect per class
        device: Device to run inference on
        use_tta: Whether to use Test Time Augmentation
        threshold: Decision threshold for pedestrian class (only used when use_tta=True)
        pedestrian_class_idx: Index of pedestrian class (default=1)

    Returns:
        Dictionary with class indices as keys and lists of (image_tensor, true_label, prediction) tuples
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    collected = {cls: [] for cls in range(len(class_names))}
    all_examples = []

    def predict_tta_logits(model, x):
        """TTA prediction with horizontal flip"""
        logits = model(x)
        logits += model(torch.flip(x, dims=[3]))  # H-flip
        return logits / 2.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            if use_tta:
                # Use TTA and threshold logic
                logits = predict_tta_logits(model, images)
                probs = torch.softmax(logits, dim=1)
                pedestrian_probs = probs[:, pedestrian_class_idx]

                # Apply threshold for pedestrian class
                preds = torch.where(
                    pedestrian_probs > threshold,
                    torch.tensor(pedestrian_class_idx, device=device),
                    torch.tensor(1 - pedestrian_class_idx, device=device)  # other class
                )
            else:
                # Standard argmax prediction (original behavior)
                outputs = model(images)
                preds = outputs.argmax(dim=1)

            for i in range(len(labels)):
                # Maintain EXACT same structure: (image, label, prediction)
                all_examples.append((images[i], labels[i], preds[i]))

    # Shuffle and collect balanced samples (same logic as original)
    random.shuffle(all_examples)

    for img, label, pred in all_examples:
        cls = label.item()
        if len(collected[cls]) < samples_per_class:
            collected[cls].append((img, label, pred))

        if all(len(v) >= samples_per_class for v in collected.values()):
            break

    return collected

def display_classification_examples(collected_examples, model, class_names, samples_per_class=3):
    st.subheader("🧪 Model Performance Examples")
    for cls, examples in collected_examples.items():
        class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        with st.expander(f"Class: {class_name}", expanded=True):
            cols = st.columns(samples_per_class)
            for col_idx, (img, label, pred) in enumerate(examples):
                with cols[col_idx]:
                    pil_img = denormalize(img)
                    st.image(pil_img, use_container_width=True)
                    true_label = class_names[label.item()] if label.item() < len(class_names) else f"Class {label.item()}"
                    pred_label = class_names[pred.item()] if pred.item() < len(class_names) else f"Class {pred.item()}"
                    is_correct = pred == label
                    status = "✅ Correct" if is_correct else "❌ Wrong"
                    st.caption(status)
                    st.write(f"**True:** {true_label}")
                    st.write(f"**Predicted:** {pred_label}")
                    with torch.no_grad():
                        output = model(img.unsqueeze(0))
                        confidence = torch.nn.functional.softmax(output[0], dim=0)[pred].item()
                    st.write(f"**Confidence:** {confidence:.2%}")

@st.cache_resource
def load_model(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.load_state_dict(checkpoint["model"])
    model = model.to(DEVICE)
    class_names = checkpoint.get("class_names", ["no pedestrian", "pedestrian"])
    return model, class_names

def main():
    st.title("Classification (Best Resnet Model)")

    # Load model + data
    model, class_names = load_model(MODEL_PATH)

    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_ds = CsvImageDataset(VAL_DIR, VAL_CSV, transform=val_tfms)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, **DATALOADER_KW)

    # ── Single-image demo + alerts ─────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Upload an image and let our AI model classify it!")
        uploaded_file = st.file_uploader("Choose an image...", type=IMG_EXT)

        notify_choice = st.radio(
            "Do you want to be notified if a pedestrian is detected?",
            ["No", "Yes"], index=0
        )

        send_photo_choice = "No"
        bot_token = chat_id = None
        to_email = None
        notify_channel = None

        if notify_choice == "Yes":
            notify_channel = st.radio(
                "How would you like to be notified?",
                ["Telegram", "Email", "Both"],
                index=0
            )
            if notify_channel in ("Telegram", "Both"):
                bot_token = st.text_input("Telegram Bot Token", type="password")
                chat_id = st.text_input("Telegram Chat ID")
                st.markdown(
                    '<p style="font-size:0.85rem;color:#9aa0a6;font-style:italic;margin-top:-4px;">'
                    'Please view the <strong>Telegram Help</strong> tab if you are unsure how to get your Telegram Bot Token & Chat ID.'
                    '</p>',
                    unsafe_allow_html=True
                )
            if notify_channel in ("Email", "Both"):
                to_email = st.text_input("Notification Email Address")

            send_photo_choice = st.radio(
                "Do you want the detected file to be sent along with the message?",
                ["No", "Yes"],
                index=0
            )

        predict_btn = None
        image = None

        if uploaded_file:
            predict_btn = st.button("Predict Image")

            # Add TTA toggle
            use_tta = st.checkbox("Use TTA (Test Time Augmentation)", value=True)

            # Add threshold slider
            threshold = st.slider(
                "Decision Threshold for Pedestrian",
                min_value=0.1,
                max_value=1,
                value=0.5,
                step=0.05,
                help="Higher threshold = more conservative pedestrian detection"
            )

            if predict_btn:
                image = Image.open(uploaded_file).convert("RGB")
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225)),
                ])
                input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    if use_tta:
                        # TTA: original + horizontal flip
                        logits_original = model(input_tensor)
                        logits_flip = model(torch.flip(input_tensor, dims=[3]))
                        output = (logits_original + logits_flip) / 2.0
                    else:
                        output = model(input_tensor)

                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    predicted_class = probabilities.argmax().item()
                    confidence = probabilities[predicted_class].item()

                    # Get pedestrian probability for threshold-based decision
                    pedestrian_prob = probabilities[1].item()  # assuming class 1 is pedestrian
                    is_pedestrian = pedestrian_prob > threshold

            caption_text = "🚨 Pedestrian has been detected from the files uploaded!"
            subject_text = "AAI3001 Image Classification Alert"
            attach_image = (send_photo_choice == "Yes")

            if notify_choice == "Yes":
                pred_label_text = class_names[predicted_class] if predicted_class < len(class_names) else f"class_{predicted_class}"
                if label_is_pedestrian(pred_label_text):
                    any_sent = False
                    if notify_channel in ("Telegram", "Both"):
                        if not bot_token or not chat_id:
                            st.info("Telegram selected but Bot Token/Chat ID missing. Skipped Telegram.")
                        else:
                            ok = send_telegram_photo(bot_token, chat_id, caption_text, image) if attach_image \
                                 else send_telegram_message(bot_token, chat_id, caption_text)
                            any_sent = ok or any_sent
                    if notify_channel in ("Email", "Both"):
                        if not to_email:
                            st.info("Email selected but recipient address is missing. Skipped Email.")
                        else:
                            ok = send_email_alert(
                                to_email=to_email,
                                subject=subject_text,
                                body=caption_text,
                                image=image if attach_image else None,
                                attach_image=attach_image
                            )
                            any_sent = ok or any_sent
                    if any_sent:
                        st.success("Alert sent.")
                    else:
                        st.info("Notification enabled, but nothing was sent (missing details?).")
                else:
                    st.info("No pedestrian detected; no alert sent.")

    if uploaded_file and predict_btn:
        with col2:
            st.subheader("Uploaded Image")
            st.image(image, caption='Uploaded Image', use_container_width=True)
        with col3:
            st.subheader("Prediction Results")
            st.write(f"**Predicted Class**: {class_names[predicted_class]}")
            st.write(f"**Confidence**: {confidence:.3f}")
            st.write(f"**TTA Used**: {'Yes' if use_tta else 'No'}")

    # Visual separator
    st.markdown("<hr>", unsafe_allow_html=True)

    # >>>>>>> NEW blue subheading above the Run Model button
    st.markdown('<div class="section-title-blue">View Examples (Best Resnet Model)</div>', unsafe_allow_html=True)

    optimal_threshold = st.slider(
        "Decision Threshold for Pedestrian",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Higher threshold = more conservative pedestrian detection"
    )

    # ── Validation examples block ──────────────────────────────────────────────
    if "model_results" not in st.session_state:  # Optimal threshold determined from prior analysis
        st.session_state.model_results = None
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Run Model", type="primary"):
            with st.spinner("Running model on validation data..."):
                collected_examples = get_model_predictions(
    model, val_loader, class_names,
    samples_per_class=3,
    device=DEVICE,
    use_tta=True,
    threshold=optimal_threshold
)
                st.session_state.model_results = collected_examples
            st.rerun()
    with col2:
        if st.session_state.model_results is not None:
            if st.button("🔄 Refresh Examples"):
                with st.spinner("Getting new examples..."):
                    collected_examples = get_model_predictions(
    model, val_loader, class_names,
    samples_per_class=3,
    device=DEVICE,
    use_tta=True,
    threshold=optimal_threshold
)
                    st.session_state.model_results = collected_examples
                st.rerun()
    if st.session_state.model_results is not None:
        display_classification_examples(st.session_state.model_results, model, class_names)
    else:
        st.info("Click 'Run Model' to see classification examples!")
    st.markdown("---")

if __name__ == '__main__':
    main()
