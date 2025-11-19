import streamlit as st
import os
import random
from pathlib import Path

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

from streamlit_image_select import image_select

from utils import CLASS_NAMES, denormalize

DATA_DIR = Path("data/validation")
SUBDIR_ALIASES = {
    "pedestrian":      ["pedestrian", "pedestrain"],
    "no pedestrian":   ["no pedestrian", "no pedestrain"],
}
MODEL_PATH = 'models/best_resnet18_pedestrian.pt'
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
CLASS_NAMES = ["no pedestrian", "pedestrian"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
# index 0 -> no pedestrian, 1 -> pedestrian
def main():

    st.set_page_config(
        page_title="Saliency Maps (Using Best Model)",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Saliency Maps")

    st.write("Here are some random images from the dataset:")

    model, checkpoint = load_model(MODEL_PATH)
    random_images = get_random_images(10)
    uploaded_file = st.file_uploader("Upload an image...", type=IMG_EXT)

    img = None
    if st.button("Get New Random Images"):
        get_random_images.clear()
        st.rerun()

    st.write("Select an image to see its saliency map:")
    selected_img_path = image_select("Random Images: ", random_images)
    label = extract_label(str(selected_img_path))
    st.write(f"Selected: {label}")

    if selected_img_path:
        img = Image.open(selected_img_path).convert("RGB")

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")


    if img:
        st.image(img, caption="Selected Image", use_container_width=True)
        use_tta = st.checkbox("Use TTA (Test Time Augmentation)", value=True)
        threshold = st.slider(
            "Decision Threshold for Pedestrian",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Higher threshold = more conservative pedestrian detection"
        )
        if st.button("Show Saliency Map"):
            show_image_with_saliency(model, img, true_label=label, use_tta=use_tta, threshold=threshold)

        if st.button("Run Occlusion Analysis"):
            with st.spinner("Performing occlusion analysis..."):

                # Preprocess your image here (convert to tensor, normalize, etc.)
                # This depends on your specific model preprocessing
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])

                input_tensor = preprocess(img).unsqueeze(0)

                # Run occlusion analysis
                result = saliency_via_occlusion_single(
                    model=model,  # Your model here
                    image=input_tensor,
                    true_label=label,
                    mask_size=32,
                    stride=8
                )

                # Visualize results
                visualize_occlusion_results_streamlit(result, CLASS_NAMES)



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

@st.cache_data
def get_random_images(num_images=10):
    all_images = []
    for label in SUBDIR_ALIASES.keys():
        for sub in SUBDIR_ALIASES[label]:
            dir_path = DATA_DIR / sub
            if dir_path.exists():
                for ext in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
                    all_images.extend(list(dir_path.rglob(f"*{ext}")))
    random_images = random.sample(all_images, num_images)
    return random_images


def generate_saliency_map(model, img, true_label=None, use_tta=False, threshold=0.5, pedestrian_class_idx=1):
    """
    Generate saliency map for a given image and model using gradient-based method

    Parameters:
    - model: Your trained PyTorch model
    - img: Input image (PIL image)
    - true_label: True label or target class for explanation
    - use_tta: Whether to use Test Time Augmentation
    - threshold: Decision threshold for pedestrian class
    - pedestrian_class_idx: Index of pedestrian class (default=1)

    Returns:
    - tuple: (saliency_map, predicted_class, confidence, is_pedestrian, pedestrian_prob)
              - First 3 values maintain backward compatibility
              - Additional values for threshold-based decision
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    # Set model to evaluation mode
    model.eval()

    # For prediction (with optional TTA and threshold)
    with torch.no_grad():
        if use_tta:
            # TTA: original + horizontal flip
            logits_original = model(img_tensor)
            logits_flip = model(torch.flip(img_tensor, dims=[3]))
            predictions_tta = (logits_original + logits_flip) / 2.0
            probabilities = torch.nn.functional.softmax(predictions_tta[0], dim=0)
        else:
            predictions = model(img_tensor)
            probabilities = torch.nn.functional.softmax(predictions[0], dim=0)

        # Get pedestrian probability and threshold-based decision
        pedestrian_prob = probabilities[pedestrian_class_idx].item()
        is_pedestrian = pedestrian_prob > threshold

        # Determine final predicted class
        if threshold != 0.5:  # Only override if using custom threshold
            predicted_class = pedestrian_class_idx if is_pedestrian else (1 - pedestrian_class_idx)
            confidence = probabilities[predicted_class].item()
        else:
            # Standard argmax
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

    # For saliency computation (always on original image for interpretability)
    img_tensor.requires_grad_()
    with torch.enable_grad():
        predictions_saliency = model(img_tensor)
        probabilities_saliency = torch.nn.functional.softmax(predictions_saliency[0], dim=0)

        # Use the final predicted class for saliency (respects threshold decision)
        if true_label is not None:
            # Convert string label to integer index if needed
            if isinstance(true_label, str):
                true_label = CLASS_NAMES.index(true_label)
            target_class = torch.tensor([true_label], device=predictions_saliency.device)
        else:
            target_class = torch.tensor([predicted_class], device=predictions_saliency.device)

        # Use gather to maintain gradient flow
        class_score = predictions_saliency.gather(1, target_class.unsqueeze(1)).squeeze(1)
        class_score = class_score[0]

        # Zero out any existing gradients
        model.zero_grad()
        if img_tensor.grad is not None:
            img_tensor.grad.zero_()

        # Backward pass to compute gradients
        class_score.backward()

        # Get gradients from the input image
        gradients = img_tensor.grad.data

        # Take maximum absolute value across color channels
        saliency_map = torch.max(torch.abs(gradients), dim=1)[0]

        # Remove batch dimension if present
        if len(saliency_map.shape) > 2:
            saliency_map = saliency_map[0]

        # Normalize to [0, 1]
        if torch.max(saliency_map) - torch.min(saliency_map) != 0:
            saliency_map = (saliency_map - torch.min(saliency_map)) / (
                    torch.max(saliency_map) - torch.min(saliency_map)
            )
        else:
            saliency_map = torch.zeros_like(saliency_map)

    # Return tuple: maintain backward compatibility + new values
    return (saliency_map.detach().cpu().numpy(),
            predicted_class,
            confidence)


def show_image_with_saliency(model, img, true_label=None, use_tta=False, threshold=0.5):
    """
    Display original image with saliency map overlay
    """
    # Generate saliency map with predictions
    saliency_map, predicted_class, confidence = generate_saliency_map(model, img, true_label, use_tta, threshold)

    # Convert PIL image to numpy array and resize to match saliency map
    img_resized = img.resize((224, 224))
    img_display = np.array(img_resized)

    # Get class name with error handling
    try:
        class_name = CLASS_NAMES[predicted_class]
    except (NameError, IndexError, KeyError):
        class_name = f"Class {predicted_class}"

    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Original image with prediction info
    ax1.imshow(img_display)
    ax1.set_title(f'Original Image\nPredicted: {class_name}\nConfidence: {confidence:.2%}')
    ax1.axis('off')

    # Saliency map
    im = ax2.imshow(saliency_map, cmap='jet')
    ax2.set_title(f'Saliency Map\n(for {class_name})')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2)

    # Overlay
    ax3.imshow(img_display)
    ax3.imshow(saliency_map, cmap='jet', alpha=0.5)
    ax3.set_title('Saliency Overlay')
    ax3.axis('off')

    plt.tight_layout()
    st.pyplot(fig)

    # Display prediction info
    st.write(f"**Prediction:** {class_name} | **Confidence:** {confidence:.2%}")


def saliency_via_occlusion_single(model, image, true_label, mask_size=32, stride=16):
    """
    Perform saliency analysis via occlusion for a single image - Streamlit compatible
    """
    model.eval()

    # Move image to device and ensure it has batch dimension
    original_image = image.to(DEVICE)
    if original_image.dim() == 3:
        original_image = original_image.unsqueeze(0)

    if isinstance(true_label, str):
        true_label = CLASS_TO_IDX[true_label]
    true_label = torch.tensor([true_label]).to(DEVICE)

    # Get original prediction
    with torch.no_grad():
        original_output = model(original_image)
        original_prob = torch.softmax(original_output, dim=1)
        original_pred = original_output.argmax(dim=1).item()
        original_confidence = original_prob[0, original_pred].item()

    # Get image dimensions
    _, C, H, W = original_image.shape

    # Create occlusion maps
    occlusion_map = torch.zeros((H, W)).to(DEVICE)
    confidence_change_map = torch.zeros((H, W)).to(DEVICE)
    misclassification_map = torch.zeros((H, W)).to(DEVICE)

    total_iterations = ((H - mask_size) // stride + 1) * ((W - mask_size) // stride + 1)
    current_iteration = 0

    # Progress bar for Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Slide mask over the image
    for i in range(0, H - mask_size + 1, stride):
        for j in range(0, W - mask_size + 1, stride):
            current_iteration += 1

            # Update progress
            if current_iteration % 10 == 0:  # Update every 10 iterations for performance
                progress = current_iteration / total_iterations
                progress_bar.progress(progress)
                status_text.text(f"Processing: {current_iteration}/{total_iterations} ({progress:.1%})")

            # Create masked image
            masked_image = original_image.clone()
            mean_val = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
            masked_image[0, :, i:i + mask_size, j:j + mask_size] = mean_val

            # Get prediction with mask
            with torch.no_grad():
                masked_output = model(masked_image)
                masked_prob = torch.softmax(masked_output, dim=1)
                masked_pred = masked_output.argmax(dim=1).item()
                masked_confidence = masked_prob[0, original_pred].item()

            # Calculate confidence drop
            confidence_drop = original_confidence - masked_confidence

            # Update occlusion maps
            occlusion_map[i:i + stride, j:j + stride] += confidence_drop
            confidence_change_map[i:i + stride, j:j + stride] = confidence_drop

            # Check if occlusion causes misclassification
            if masked_pred != original_pred:
                misclassification_map[i:i + stride, j:j + stride] += 1

    # Normalize occlusion map
    if occlusion_map.max() > 0:
        occlusion_map = occlusion_map / occlusion_map.max()

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    return {
        'image': image.cpu(),
        'true_label': true_label.item(),
        'original_pred': original_pred,
        'original_confidence': original_confidence,
        'occlusion_map': occlusion_map.cpu(),
        'confidence_change_map': confidence_change_map.cpu(),
        'misclassification_map': misclassification_map.cpu()
    }

def visualize_occlusion_results_streamlit(result, class_names):
    """
    Visualize occlusion results in Streamlit
    """
    # Convert tensors to numpy for visualization
    image = denormalize(result['image'].squeeze(0))
    occlusion_map = result['occlusion_map'].numpy()
    confidence_change_map = result['confidence_change_map'].numpy()
    misclassification_map = result['misclassification_map'].numpy()

    true_label = result['true_label']
    pred_label = result['original_pred']
    confidence = result['original_confidence']

    # Display prediction information
    st.subheader("Model Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("True Label", class_names[true_label])
    with col2:
        st.metric("Predicted Label", class_names[pred_label])
    with col3:
        st.metric("Confidence", f"{confidence:.3f}")

    # Create visualizations in columns
    st.subheader("Occlusion Analysis Results")

    # Row 1: Original image and occlusion heatmap
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Image**")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.imshow(image)
        ax1.set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}')
        ax1.axis('off')
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.write("**Occlusion Sensitivity Map**")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im1 = ax2.imshow(occlusion_map, cmap='hot')
        ax2.set_title('Confidence Drop Heatmap')
        ax2.axis('off')
        plt.colorbar(im1, ax=ax2, fraction=0.046, pad=0.04)
        st.pyplot(fig2)
        plt.close(fig2)

    # Row 2: Overlay and critical regions
    col3, col4 = st.columns(2)

    with col3:
        st.write("**Overlay on Image**")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.imshow(image, alpha=0.7)
        im2 = ax3.imshow(occlusion_map, cmap='hot', alpha=0.6)
        ax3.set_title('Occlusion Map Overlay')
        ax3.axis('off')
        plt.colorbar(im2, ax=ax3, fraction=0.046, pad=0.04)
        st.pyplot(fig3)
        plt.close(fig3)

    with col4:
        st.write("**Critical Regions**")
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        ax4.imshow(image, alpha=0.8)

        # Find top 10% regions with highest confidence drop
        if confidence_change_map.max() > 0:
            threshold = np.percentile(confidence_change_map[confidence_change_map > 0], 90)
            critical_regions = confidence_change_map >= threshold
            y_coords, x_coords = np.where(critical_regions)
            ax4.scatter(x_coords, y_coords, c='red', s=10, alpha=0.6)

        ax4.set_title('Top 10% Confidence Drop Regions')
        ax4.axis('off')
        st.pyplot(fig4)
        plt.close(fig4)

    # Row 3: Misclassification regions
    st.write("**Misclassification Analysis**")
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    ax5.imshow(image, alpha=0.8)

    if misclassification_map.max() > 0:
        misclassification_regions = misclassification_map > 0
        y_coords, x_coords = np.where(misclassification_regions)
        ax5.scatter(x_coords, y_coords, c='blue', s=15, alpha=0.6, label='Causes misclassification')
        ax5.legend()

    ax5.set_title('Regions Where Occlusion Causes Misclassification')
    ax5.axis('off')
    st.pyplot(fig5)
    plt.close(fig5)

    # Display statistics
    st.subheader("Analysis Statistics")

    # Calculate statistics
    max_confidence_drop = confidence_change_map.max()
    if confidence_change_map.max() > 0:
        threshold = np.percentile(confidence_change_map[confidence_change_map > 0], 90)
        critical_regions = confidence_change_map >= threshold
        avg_confidence_drop = np.mean(confidence_change_map[critical_regions])
        critical_percentage = np.sum(critical_regions) / (image.shape[0] * image.shape[1]) * 100
    else:
        avg_confidence_drop = 0
        critical_percentage = 0

    misclassification_pixels = np.sum(misclassification_map > 0)

    # Display metrics
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Max Confidence Drop", f"{max_confidence_drop:.3f}")
    with stat_col2:
        st.metric("Avg Drop in Critical Regions", f"{avg_confidence_drop:.3f}")
    with stat_col3:
        st.metric("Misclassification Pixels", f"{misclassification_pixels}")
    with stat_col4:
        st.metric("Critical Region %", f"{critical_percentage:.1f}%")



def extract_label(file_path):
    """Extract the label from file path like 'data/validation/no pedestrian/val (104).jpg'"""
    parts = file_path.split('/')
    # The label is typically the second last part in such paths
    return parts[-2]




if __name__ == '__main__':
    main()