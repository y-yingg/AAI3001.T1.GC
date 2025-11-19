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
        font-size: 2.4rem;
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
      ul {
        margin-left: 1.4em;
      }
      li {
        margin-bottom: 0.5em;
      }
      .term-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 10px;
      }
      .term1 {
        background-color: rgba(45,108,223,0.15);
        color: #9ab6ff;
        border: 1px solid #2d6cdf;
      }
      .term2 {
        background-color: rgba(255,217,102,0.12);
        color: #ffe69a;
        border: 1px solid #ffd966;
      }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Header ----------
    st.markdown(
        "<div class='main-title'>Pedestrian Detection Projects - Term 1 vs Term 2</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='subtitle'>Team 11 · AAI3001 - Deep Learning and Computer Vision</div>",
        unsafe_allow_html=True,
    )

    # ---------- Two-column comparison ----------
    col1, col2 = st.columns([1, 1])

    # ----- TERM 1 -----
    with col1:
        st.markdown("""
        <div class="section">
          <span class="term-tag term1">Term 1 · Classification</span>
          <div class="section-title">Pedestrian Detection Alert System</div>
          <p>
            In Term 1, our focus was on <strong>image-level pedestrian detection</strong>. 
            The system takes in a single image and predicts whether a pedestrian is present 
            or not, acting as an early screening step for monitored or restricted areas.
          </p>
          <p>
            We fine-tuned a <strong>ResNet-18</strong> model using transfer learning and 
            data augmentation to handle varied lighting, backgrounds and viewpoints, 
            despite having a relatively small dataset.
          </p>
          <p>
            The project also explored how model predictions could be integrated with 
            <strong>alert mechanisms</strong> (e.g. Telegram / email notifications) 
            to support real-time safety workflows.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section">
          <div class="section-title">Term 1 – Key Points</div>
          <ul>
            <li>Binary <strong>image classification</strong>: pedestrian vs non-pedestrian.</li>
            <li>Fine-tuned <strong>ResNet-18</strong> with transfer learning.</li>
            <li>Used data augmentation to improve generalisation.</li>
            <li>Prototype design for <strong>alert pipeline</strong> (messaging / email).</li>
            <li>Main outcome: a solid baseline for pedestrian presence detection.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # ----- TERM 2 -----
    with col2:
        st.markdown("""
        <div class="section">
          <span class="term-tag term2">Term 2 · Object Detection</span>
          <div class="section-title">Pedestrian Detection - Object Detection Extension</div>
          <p>
            In Term 2, we extended the original idea from “Is there a pedestrian?” to 
            <strong>“Where are the riders and what are they riding?”</strong>. 
            Instead of a single label per image, the model now predicts 
            <strong>bounding boxes and classes</strong> for multiple objects.
          </p>
          <p>
            We trained a <strong>RetinaNet (ResNet-50 FPN backbone)</strong> model 
            on a COCO-style dataset with three rider classes:
          </p>
          <ul>
            <li>Person riding bicycle</li>
            <li>Person riding kickboard</li>
            <li>Person riding motorcycle</li>
          </ul>
          <p>
            The detector is deployed as an interactive web demo where users can upload 
            images and see visualised bounding boxes and class labels.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section">
          <div class="section-title">Term 2 – Key Points</div>
          <ul>
            <li>Shift from <strong>classification</strong> to full <strong>object detection</strong>.</li>
            <li>Model: <strong>RetinaNet</strong> with ResNet-50 FPN backbone.</li>
            <li>COCO-style dataset with 3 rider-related classes.</li>
            <li>Pre-processing with resizing, normalisation and augmentation.</li>
            <li>Deployed via <strong>Streamlit</strong> + model hosted on Hugging Face.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # ---------- Connection / Overall summary ----------
    st.markdown("""
    <div class="section">
      <div class="section-title">How Term 1 and Term 2 Fit Together</div>
      <p>
        Together, the two projects form a <strong>progressive pipeline</strong>:
      </p>
      <ul>
        <li><strong>Term 1</strong> builds the foundation with image-level classification and 
            explores how deep learning can support a Pedestrian Detection Alert System.</li>
        <li><strong>Term 2</strong> advances this by locating specific rider categories with 
            bounding boxes, making the outputs more useful for monitoring, analytics and 
            potential integration with smart city infrastructure.</li>
      </ul>
      <p>
        The work in Term 2 reuses the lessons learnt in Term 1 about dataset quality, 
        augmentation and deployment, and pushes the project closer towards a realistic, 
        end-to-end solution for urban safety and surveillance use cases.
      </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
