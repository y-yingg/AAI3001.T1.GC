import streamlit as st

def main():
    st.set_page_config(page_title="Explanations of the 2 Models", layout="wide")

    # ---------- CSS Styling ----------
    st.markdown("""
    <style>
      .model-box {
        background-color: rgba(240, 240, 240, 0.12);
        border-radius: 14px;
        padding: 25px 28px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.08);
        margin-bottom: 25px;
        line-height: 1.6;
        height: 700px;
        overflow-y: auto;
      }
      .model-title {
        color: #2d6cdf;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 12px;
      }
      .section-sub {
        font-weight: 600;
        color: #e2e2e2;
        margin-top: 1em;
        margin-bottom: 0.3em;
        font-size: 1.05rem;
      }
      ul {
        margin-left: 1.3em;
        list-style-type: disc;
      }
      li {
        margin-bottom: 0.5em;
      }
      code {
        background-color: rgba(0,0,0,0.25);
        border-radius: 5px;
        padding: 2px 6px;
        font-size: 0.95rem;
        color: #a8ff9f;
      }
      strong {
        color: #ffd966;
      }
      p {
        margin-bottom: 0.8em;
      }
      .model-box::-webkit-scrollbar {
        width: 8px;
      }
      .model-box::-webkit-scrollbar-thumb {
        background-color: #888;
        border-radius: 4px;
      }
      .model-box::-webkit-scrollbar-thumb:hover {
        background-color: #555;
      }
    </style>
    """, unsafe_allow_html=True)

    st.title("Explanations of the 2 Models")
    st.caption("Overview of the two ResNet18 models used in this project")

    # ---------- Layout: 2 Columns ----------
    col1, col2 = st.columns(2)

    # ---------- Left Column: Base Model ----------
    with col1:
        st.markdown("""
        <div class="model-box">
          <div class="model-title">ResNet18 Base Version</div>

          <p><strong>resnet18 base version:</strong></p>

          <p class="section-sub">Data Prep:</p>
          <p>Transforms by:</p>
          <ul>
            <li>Using <code>resize((224, 224))</code> to fit the ResNet18 model.</li>
            <li><code>RandomHorizontalFlip</code> and <code>RandomRotation(10)</code> only on the training dataset to improve generalization by helping the model learn to recognize pedestrians in different positions.</li>
            <li><code>ToTensor()</code> transform converts images from their normal picture format into numerical tensors (arrays of numbers) that PyTorch models can understand. This allows the model to process the image mathematically and learn visual patterns during training.</li>
            <li>ImageNet normalization to normalize brightness and color levels by adjusting the image colors so that they look statistically similar to the dataset the base model ResNet18 was originally trained on.</li>
            <li>Using <code>DataLoader</code> to:
              <ul>
                <li>Load the data in mini-batches (size 32)</li>
                <li>Shuffle the training data (for better learning)</li>
                <li>Prepare the images and labels to be sent to the GPU</li>
              </ul>
            </li>
          </ul>

          <p><strong>Base Model:</strong> ResNet-18 from <code>torchvision</code> as it is a well-known convolutional neural network (CNN) architecture that has already been trained on ImageNet.</p>
          <p>The pretrained weights help the model “start smart” - Default was chosen.</p>

          <p class="section-sub">Modelling:</p>
          <ul>
            <li>Replace the final fully-connected layer with <code>nn.Linear(in_features, 2)</code> to output logits for the two classes as it helps to:
              <ul>
                <li>Keep the pretrained layers (for general image understanding)</li>
                <li>Replace & train the last layer (to specialize on pedestrian detection)</li>
              </ul>
            </li>
            <li>Criterion: <code>nn.CrossEntropyLoss()</code> which is suitable for 2 classes.</li>
            <li>Optimiser: <code>AdamW(lr=1e-4, weight_decay=1e-4)</code> updates the model’s weights after each training step so it makes fewer mistakes. It uses a small learning rate for gentle fine-tuning and applies weight decay to stop the model from memorizing the training data too much and cause overfitting.</li>
          </ul>

          <p class="section-sub">Modelling Training:</p>
          <ul>
            <li>Training runs for 8 epochs with each epochs:
              <ul>
                <li>Model sees a batch of images</li>
                <li>It predicts whether each has a pedestrian or not</li>
                <li>The loss is calculated</li>
                <li>The optimizer updates the weights</li>
                <li>Accuracy and loss are tracked for both train and validation sets</li>
              </ul>
            </li>
            <li>If the validation accuracy improves, the model is saved (checkpointed).</li>
            <li>To prevent the pretrained model from overfitting over time, I used a checkpoint method that saves the best-performing model based on validation accuracy. This ensures that the most accurate version of the model is kept, instead of only relying on the final trained model.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # ---------- Right Column: Best Model ----------
    with col2:
        st.markdown("""
        <div class="model-box">
          <div class="model-title">ResNet18 Best Version</div>

          <p><strong>best_resnet18's explaination:</strong></p>

          <p><strong>Resnet 18 Fine-Tuning with Balanced Sampling and Data Augmentation</strong></p>
          <ul>
            <li>Uses milder agumentation with smaller rotation, tighter crop ratios and added <code>RandomErasing</code> that hides small random patches that teaches the model not to rely on one single detail (like a person’s head), but to recognize pedestrians even if part of them is missing.</li>
          </ul>

          <p><strong>Balanced Sampling</strong></p>
          <ul>
            <li>Added a <code>WeightedRandomSampler</code> so that both classes appear equally often in each batch, even if one class has fewer images and it prevents the model from being biased toward the more common class.</li>
          </ul>

          <p><strong>In the CrossEntropyLoss:</strong></p>
          <ul>
            <li>Added <strong>Class weights</strong> → gives more penalty if the model misses a pedestrian (to reduce false negatives).</li>
            <li>Added <strong>Label smoothing (0.05)</strong> → slightly softens the target labels so the model doesn’t become overconfident and memorizing noise (less chance of overfitting).</li>
          </ul>

          <p><strong>Divided the model training into 2 phases:</strong></p>
          <ul>
            <li><strong>Phase A:</strong> freeze the pretrained backbone and only train the last layer (“head”) which makes the head adapt quickly to dataset without disturbing the pretrained features from ImageNet.</li>
            <li><strong>Phase B:</strong> unfreeze everything and fine-tune the whole network at a smaller learning rate that refines all weights carefully once the head has learned the task.</li>
          </ul>

          <p><strong>Outcome:</strong><br>
          This allow the model to be more stable training and better use of pretrained knowledge.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
