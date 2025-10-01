from pathlib import Path

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


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ["no pedestrian", "pedestrian"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
VAL_DIR   = Path("data/validation")
VAL_CSV   = VAL_DIR / "labels.csv"

DATALOADER_KW = dict(
    num_workers=0,
    pin_memory=False,
    persistent_workers=False
)

# ---------- Email (Gmail) config ----------
SENDER_EMAIL = "sitprojects2024@gmail.com"
SENDER_APP_PASSWORD = "hglikztdmngldhvf"  # Gmail app password

# ---------- Telegram helpers ----------
def send_telegram_message(bot_token: str, chat_id: str, message: str):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        requests.post(url, data=payload, timeout=15)
        return True
    except Exception:
        return False  # fail silently

def send_telegram_photo(bot_token: str, chat_id: str, caption: str, image: Image.Image):
    """
    Try to send photo if under 5 MB.
    If too large or fails (timeout/error), fallback to text-only message with apology.
    """
    apology_msg = caption + " Sorry, the file size is too large for telegram so it could not be send here."
    try:
        import sys
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)  # smaller than PNG
        buf.seek(0)

        size_bytes = sys.getsizeof(buf.getvalue())
        max_size = 5 * 1024 * 1024  # 5 MB

        if size_bytes > max_size:
            send_telegram_message(bot_token, chat_id, apology_msg)
            return True

        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        data = {"chat_id": chat_id, "caption": caption}
        files = {"photo": ("image.jpg", buf, "image/jpeg")}
        resp = requests.post(url, data=data, files=files, timeout=30)

        if resp.status_code != 200:
            # fallback if Telegram rejects
            send_telegram_message(bot_token, chat_id, apology_msg)
        return True
    except Exception:
        # fallback if request fails (timeout, network, etc.)
        send_telegram_message(bot_token, chat_id, apology_msg)
        return True

# ---------- Email helpers ----------
def send_email_alert(
    to_email: str,
    subject: str,
    body: str,
    image: Optional[Image.Image],
    attach_image: bool
) -> bool:
    """
    Sends an email via Gmail SMTP. If attach_image=True and image provided,
    attaches the image (no size/memory limit checks here).
    """
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
            msg.add_attachment(
                img_bytes, maintype="image", subtype="jpeg", filename="detected.jpg"
            )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print(f"Email error: {e}")
    return False


def label_is_pedestrian(label_text: str) -> bool:
    return "pedestrian" in label_text.lower()

def get_model_predictions(model, val_loader, class_names, samples_per_class=3):
    model.eval()
    collected = {cls: [] for cls in range(len(class_names))}
    all_examples = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for i in range(len(labels)):
                all_examples.append((images[i], labels[i], preds[i]))
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
                    #st.image(pil_img, use_container_width=True)
                    st.image(pil_img, use_column_width=True)
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

def main():
    st.set_page_config(page_title="Image Classification Demo", page_icon="🖼️", layout="wide")
    st.title("🖼️ Image Classification Demo")

    @st.cache_resource
    def load_model(path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        model = resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        model.load_state_dict(checkpoint["model"])
        model = model.to(DEVICE)
        return model, checkpoint

    model, checkpoint = load_model('models/resnet18_pedestrian.pt')
    CLASS_NAMES = checkpoint.get("class_names", ["no pedestrian", "pedestrian"])

    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_ds = CsvImageDataset(VAL_DIR, VAL_CSV, transform=val_tfms)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, **DATALOADER_KW)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Upload an image and let our AI model classify it!")
        uploaded_file = st.file_uploader("Choose an image...", type=IMG_EXT)

        # 1) Top-level: notify or not
        notify_choice = st.radio(
            "Do you want to be notified if a pedestrian is detected?",
            ["No", "Yes"], index=0
        )

        # Defaults
        send_photo_choice = "No"
        bot_token = chat_id = None
        to_email = None
        notify_channel = None

        if notify_choice == "Yes":
            # 2) Channel selector
            notify_channel = st.radio(
                "How would you like to be notified?",
                ["Telegram", "Email", "Both"],
                index=0
            )

            # 3) Inputs per channel
            if notify_channel in ("Telegram", "Both"):
                bot_token = st.text_input("Telegram Bot Token", type="password")
                chat_id = st.text_input("Telegram Chat ID")

            if notify_channel in ("Email", "Both"):
                to_email = st.text_input("Notification Email Address")

            # 4) Always show send-file choice for all channels
            send_photo_choice = st.radio(
                "Do you want the detected file to be sent along with the message?",
                ["No", "Yes"],
                index=0
            )

        predict_btn = None
        image = None

    if uploaded_file:
        predict_btn = st.button("Predict Image")
        if predict_btn:
            image = Image.open(uploaded_file).convert("RGB")
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225)),
            ])
            input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

            if 'resnet_model' not in locals():
                resnet_model, _ = load_model('models/resnet18_pedestrian.pt')

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class = probabilities.argmax().item()
                confidence = probabilities[predicted_class].item()

            # ---------- Alert logic ----------
            caption_text = "🚨 Pedestrian has been detected from the files uploaded!"
            subject_text = "AAI3001 Image Classification Alert"
            attach_image = (send_photo_choice == "Yes")

            if notify_choice == "Yes":
                pred_label_text = CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else f"class_{predicted_class}"
                if label_is_pedestrian(pred_label_text):
                    any_sent = False

                    # Telegram path
                    if notify_channel in ("Telegram", "Both"):
                        if not bot_token or not chat_id:
                            st.info("Telegram selected but Bot Token/Chat ID missing. Skipped Telegram.")
                        else:
                            if attach_image:
                                ok = send_telegram_photo(bot_token, chat_id, caption_text, image)
                            else:
                                ok = send_telegram_message(bot_token, chat_id, caption_text)
                            any_sent = ok or any_sent

                    # Email path
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
            #st.image(image, caption='Uploaded Image', use_container_width=True)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        with col3:
            st.subheader("Prediction")
            st.write(f"**Predicted Class:** {CLASS_NAMES[predicted_class]}")
            st.write(f"**Confidence:** {confidence:.2%}")

    # st.markdown("---")
    
    # Validation demo
    if "model_results" not in st.session_state:
        st.session_state.model_results = None
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Run Model", type="primary"):
            with st.spinner("Running model on validation data..."):
                collected_examples = get_model_predictions(model, val_loader, CLASS_NAMES)
                st.session_state.model_results = collected_examples
            st.rerun()
    with col2:
        if st.session_state.model_results is not None:
            if st.button("🔄 Refresh Examples"):
                with st.spinner("Getting new examples..."):
                    collected_examples = get_model_predictions(model, val_loader, CLASS_NAMES)
                    st.session_state.model_results = collected_examples
                st.rerun()
    if st.session_state.model_results is not None:
        display_classification_examples(st.session_state.model_results, model, CLASS_NAMES)
    else:
        st.info("Click 'Run Model' to see classification examples!")
    st.markdown("---")

if __name__ == '__main__':
    main()
