import streamlit as st

def main():
    st.set_page_config(page_title="Model explanation", layout="wide")

    # ---------- CSS Styling ----------
    st.markdown("""
    <style>
      .page-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
      }
      .page-subtitle {
        font-size: 0.98rem;
        opacity: 0.8;
        margin-bottom: 1.5rem;
      }
      .model-box {
        background: linear-gradient(135deg, rgba(45,108,223,0.12), rgba(0,0,0,0.4));
        border-radius: 18px;
        padding: 28px 30px 26px 30px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.35);
        margin-bottom: 25px;
        line-height: 1.6;
        border: 1px solid rgba(255,255,255,0.08);
      }
      .model-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1rem;
        margin-bottom: 0.6rem;
      }
      .model-title {
        color: #f5f7ff;
        font-weight: 800;
        font-size: 1.5rem;
      }
      .model-tag {
        font-size: 0.8rem;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.18);
        background-color: rgba(0,0,0,0.25);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #ffd966;
        white-space: nowrap;
      }
      .section-sub {
        font-weight: 700;
        color: #e2e2e2;
        margin-top: 1.2em;
        margin-bottom: 0.3em;
        font-size: 1.05rem;
        text-transform: none;
        letter-spacing: 0.03em;
      }
      .section-sub::before {
        content: "";
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #2d6cdf;
        margin-right: 8px;
        box-shadow: 0 0 12px rgba(45,108,223,0.7);
      }
      ul {
        margin-left: 1.3em;
        list-style-type: disc;
      }
      li {
        margin-bottom: 0.4em;
      }
      strong {
        color: #ffd966;
        font-weight: 700;
      }
      p {
        margin-bottom: 0.7em;
      }
      .metrics-box {
        margin-top: 0.5rem;
        padding: 14px 16px;
        border-radius: 12px;
        background-color: rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.12);
      }
      .metrics-title {
        font-weight: 700;
        margin-bottom: 0.4rem;
        color: #f5f5f5;
      }
      .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 0.4rem 0.8rem;
        margin-top: 0.3rem;
      }
      .metric-pill {
        font-size: 0.9rem;
        padding: 6px 9px;
        border-radius: 999px;
        background-color: rgba(19, 31, 68, 0.9);
        border: 1px solid rgba(255,255,255,0.12);
        display: flex;
        justify-content: space-between;
        gap: 0.35rem;
      }
      .metric-pill span:first-child {
        opacity: 0.85;
      }
      .metric-pill span:last-child {
        font-weight: 600;
      }
      .analysis-box {
        margin-top: 0.6rem;
        padding: 14px 16px;
        border-radius: 12px;
        background-color: rgba(255,255,255,0.03);
        border: 1px dashed rgba(255,255,255,0.15);
      }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Page Title ----------
    st.markdown('<div class="page-title">Model explanation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">RetinaNet (ResNet-50 FPN) final model used in this project</div>',
        unsafe_allow_html=True
    )

    # ---------- Main Content ----------
    st.markdown("""
    <div class="model-box">

      <div class="model-header">
        <div class="model-title">RetinaNet (ResNet-50 FPN) Final Model</div>
        <div class="model-tag">Final Model</div>
      </div>

      <p><strong>Model explanation</strong></p>

      <p class="section-sub">Data Preparation:</p>
      <ul>
        <li>Resize all images to 512×512 for stable training and efficient batching.</li>
        <li>Training augmentations: RandomHorizontalFlip(p=0.3) to generalize orientation.</li>
        <li>Normalization: mean/std = [0.5, 0.5, 0.5] to stabilize feature distributions.</li>
        <li>Custom collate function preserves variable number of objects per image.</li>
      </ul>

      <p class="section-sub">Model Architecture</p>
      <ul>
        <li>Backbone: Pretrained ResNet-50 providing rich spatial and semantic features.</li>
        <li>Feature Pyramid Network (FPN): Merges multi-scale features to improve medium and large object detection.</li>
        <li>Heads:
          <ul>
            <li>Classification subnet using focal loss to reduce dominance of easy background anchors.</li>
            <li>Regression subnet for bounding box refinement with Smooth L1.</li>
          </ul>
        </li>
      </ul>

      <p class="section-sub">Why RetinaNet</p>
      <ul>
        <li>One-stage detector which has faster inference than a two-stage detector.</li>
        <li>Focal Loss directly addresses anchor imbalance.</li>
        <li>FPN improves robustness to scale variation in riders VS vehicles or background.</li>
      </ul>

      <p class="section-sub">Training</p>
      <ul>
        <li>Optimizer: AdamW(lr=1e-4, weight_decay=1e-4) for smoother convergence and regularization.</li>
        <li>Scheduler: StepLR(step_size=10, gamma=0.1) to refine learning after initial fitting.</li>
        <li>Epochs: 15 total.</li>
        <li>Checkpoints saved every epoch with a rolling latest file for reliability.</li>
      </ul>

      <p class="section-sub">Evaluation</p>
      <ul>
        <li>After predictions were generated at 256×256, the boxes were rescaled back to the original image dimensions for COCO metrics.</li>
        <li>Low score threshold (0.01) used during evaluation to allow COCO’s precision–recall sweep.</li>
      </ul>

      <div class="metrics-box">
        <div class="metrics-title">Final Validation Metrics (COCO)</div>
        <div class="metrics-grid">
          <div class="metric-pill"><span>AP</span><span>0.6780</span></div>
          <div class="metric-pill"><span>AP50</span><span>0.9148</span></div>
          <div class="metric-pill"><span>AP75</span><span>0.7993</span></div>
          <div class="metric-pill"><span>AP_small</span><span>0.1832</span></div>
          <div class="metric-pill"><span>AP_med</span><span>0.5643</span></div>
          <div class="metric-pill"><span>AP_large</span><span>0.7425</span></div>
          <div class="metric-pill"><span>AR1</span><span>0.5789</span></div>
          <div class="metric-pill"><span>AR10</span><span>0.7506</span></div>
          <div class="metric-pill"><span>AR100</span><span>0.7511</span></div>
          <div class="metric-pill"><span>AR_small</span><span>0.1833</span></div>
          <div class="metric-pill"><span>AR_med</span><span>0.6830</span></div>
          <div class="metric-pill"><span>AR_large</span><span>0.7983</span></div>
        </div>
        <p style="margin-top:0.7rem; font-size:0.92rem;">Total detections written: 9375</p>
      </div>

      <p class="section-sub">Analysis</p>
      <div class="analysis-box">
        <p>High AP50 shows strong localization consistency.</p>
        <p>Medium and large object performance is strong while small object detection is lower but acceptable given input scale and task focus.</p>
        <p>The final RetinaNet (ResNet-50 FPN) model, trained with light augmentation, AdamW, and StepLR, achieves stable convergence and the best validation performance among other experimental models.</p>
      </div>

    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

