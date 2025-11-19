import streamlit as st

def main():
    st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

    # ---------- CSS Styling ----------
    st.markdown("""
    <style>
      body {
        background-color: #0e1117;
      }
      .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2d6cdf;
        text-align: center;
        margin-bottom: 10px;
      }
      .subtitle {
        font-size: 1.2rem;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 35px;
      }
      .section {
        background: rgba(240, 240, 240, 0.12);
        border-radius: 16px;
        padding: 30px 40px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        margin-bottom: 25px;
        line-height: 1.7;
        color: #e8e8e8;
      }
      .section-title {
        color: #ffd966;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 12px;
        border-left: 5px solid #2d6cdf;
        padding-left: 10px;
      }
      .section p {
        font-size: 1.05rem;
        margin-bottom: 0.8em;
      }
      .dataset-link a {
        color: #79b8ff;
        text-decoration: none;
        font-weight: 500;
      }
      .dataset-link a:hover {
        text-decoration: underline;
      }
      ul {
        margin-left: 1.4em;
      }
      li {
        margin-bottom: 0.5em;
      }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Content ----------
    st.markdown("<div class='main-title'>Pedestrian Detection Alert System - Object Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Team 11 · AAI3001 - Deep Learning and Computer Vision (Term 2 Project)</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    # ----- LEFT COLUMN -----
    with col1:
        st.markdown("""
        <div class="section">
          <div class="section-title">Project Overview</div>
          <p>
            In Term 2, we extend our pedestrian detection work from image classification
            to <strong>full object detection</strong>. Instead of simply answering
            “Is there a pedestrian in this image?”, the system now learns to 
            <strong>locate</strong> pedestrians and related riders with bounding boxes.
          </p>
          <p>
            The solution focuses on three urban rider classes:
            <strong>Person riding bicycle</strong>, 
            <strong>Person riding kickboard</strong>, and 
            <strong>Person riding motorcycle</strong>. 
            These are common in busy city environments and are important for traffic safety,
            planning and monitoring of restricted or high-risk areas.
          </p>
          <p>
            Our model is built using <strong>RetinaNet with a ResNet-50 FPN backbone</strong>,
            fine-tuned on a COCO-style dataset. The aim is to achieve robust detection quality
            that can support downstream use cases such as real-time alerts, analytics dashboards
            or integration into smart-city systems.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section">
          <div class="section-title">Objectives</div>
          <ul>
            <li>Design an end-to-end <strong>pedestrian object detection</strong> pipeline for urban scenes.</li>
            <li>Train and tune a <strong>RetinaNet (ResNet-50 FPN)</strong> detector on a COCO-style dataset with 3 rider classes.</li>
            <li>Compare RetinaNet against alternative models such as YOLOv8, Faster R-CNN, ResNet and EfficientDet.</li>
            <li>Achieve validation performance suitable for deployment in monitoring or traffic-safety tools.</li>
            <li>Analyse qualitative results (example detections, missed cases) to understand strengths and limitations.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # ----- RIGHT COLUMN -----
    with col2:
        st.markdown("""
        <div class="section">
          <div class="section-title">Dataset & Methodology</div>
          <p>
            The dataset uses <strong>COCO-style annotations</strong> with bounding boxes for
            three rider classes related to pedestrians:
          </p>
          <ul>
            <li>Person riding bicycle</li>
            <li>Person riding kickboard</li>
            <li>Person riding motorcycle</li>
          </ul>
          <p>
            Images are pre-processed by resizing to a fixed resolution 
            (e.g. <strong>256×256</strong>), normalising pixel values and applying 
            random horizontal flips for augmentation. This helps the model generalise
            better to real-world variations in pose and viewpoint.
          </p>
          <p>
            On the modelling side, we use <strong>transfer learning</strong> with a 
            pretrained ResNet-50 feature pyramid network (FPN) inside 
            <strong>RetinaNet</strong>. The detector is optimised using 
            <strong>AdamW</strong> (learning rate <code>1e-4</code>, weight decay 
            <code>1e-4</code>) and a <strong>StepLR scheduler</strong>. 
            Multiple experiments with different training settings are run and 
            the best checkpoint is chosen based on validation mAP.
          </p>
          <p class="dataset-link">
            <strong>Dataset (Roboflow):</strong><br>
            <a href="https://universe.roboflow.com/atech-witjl/pedestrian-kmhf3/dataset/3" target="_blank">
              • Pedestrian Object Detection Dataset (3 rider classes)
            </a>
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section">
          <div class="section-title">Results & Insights</div>
          <p>
            In our initial comparison, several detectors were evaluated:
          </p>
          <ul>
            <li><strong>YOLOv8</strong> - mAP ≈ 0.82</li>
            <li><strong>RetinaNet</strong> - mAP ≈ 0.79 using only ~25% of the dataset</li>
            <li><strong>Faster R-CNN</strong> - mAP ≈ 0.78</li>
            <li><strong>ResNet (classification)</strong> - mAP ≈ 0.66</li>
            <li><strong>EfficientDet</strong> - mAP ≈ 0.49</li>
          </ul>
          <p>
            Based on this, we chose RetinaNet as a strong and reasonably efficient baseline,
            then carried out three tuning runs. The final tuned model achieves:
          </p>
          <ul>
            <li><strong>COCO AP@[0.5:0.95]</strong> ≈ 0.66</li>
            <li><strong>AP@0.5</strong> ≈ 0.92</li>
            <li><strong>Overall mAP</strong> ≈ 0.91 on the validation set</li>
          </ul>
          <p>
            Qualitative examples show <strong>tight bounding boxes</strong> around riders and
            relatively <strong>few false positives</strong>. Most missed detections occur for
            very small, distant or heavily occluded pedestrians, suggesting that additional data
            or tailored augmentation for these edge cases could further improve recall.
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ----- FULL-WIDTH SECTION -----
    st.markdown("""
    <div class="section">
      <div class="section-title">Conclusion & Future Work</div>
      <p>
        This Term 2 project shows that <strong>RetinaNet with a ResNet-50 FPN backbone</strong> is a
        strong candidate for pedestrian-related object detection, even with a modest dataset.
        The tuned model reaches high mAP and produces clean detections across different rider types.
      </p>
      <p>
        Moving forward, we plan to:
      </p>
      <ul>
        <li>Collect more varied scenes (night, rain, crowds and different camera angles).</li>
        <li>Further tune confidence thresholds and non-maximum suppression settings.</li>
        <li>Optimise inference speed for deployment on edge devices or live CCTV feeds.</li>
        <li>Integrate the detector with downstream components such as alerting pipelines or analytics dashboards.</li>
      </ul>
      <p>
        Overall, this object detection extension builds on our Term 1 work and moves the
        Pedestrian Detection Alert System closer to a practical tool for monitoring and 
        safety in real urban environments.
      </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
