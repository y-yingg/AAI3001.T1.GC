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
    st.markdown("<div class='main-title'>Pedestrian Detection Alert System</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Team 11 · AAI3001 — Deep Learning and Computer Vision (Tri 1, 2024)</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="section">
          <div class="section-title">Project Overview</div>
          <p>
            The <strong>Pedestrian Detection Alert System</strong> is an AI-based solution designed to detect pedestrians in images and trigger alerts when pedestrians are present in restricted or monitored areas.
          </p>
          <p>
            Using a <strong>fine-tuned ResNet-18 model</strong>, the system classifies images as either containing pedestrians or not. 
            This model leverages <strong>transfer learning</strong> and <strong>data augmentation</strong> techniques to achieve high accuracy, even with limited training data.
          </p>
          <p>
            The project aims to enhance public and urban safety by supporting real-time monitoring and automated alert notifications through Telegram and Email.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section">
          <div class="section-title">Objectives</div>
          <ul>
            <li>Develop an image classification model to detect pedestrian presence.</li>
            <li>Use fine-tuning techniques to improve performance on a small dataset.</li>
            <li>Integrate real-time alert features through messaging and email APIs.</li>
            <li>Evaluate and compare base and fine-tuned ResNet18 performance.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section">
          <div class="section-title">Datasets & References</div>
          <p>Our model was trained using publicly available pedestrian datasets:</p>
          <p class="dataset-link">
            <a href="https://www.kaggle.com/datasets/mohamedgobara/26-class-object-detection-dataset/data" target="_blank">• 26-Class Object Detection Dataset (Kaggle)</a><br>
            <a href="https://www.kaggle.com/datasets/tejasvdante/pedestrian-no-pedestrian" target="_blank">• Pedestrian vs Non-Pedestrian Dataset (Kaggle)</a>
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section">
          <div class="section-title">Key Findings & Insights</div>
          <p>
            The <strong>fine-tuned ResNet-18 model</strong> demonstrated strong classification capability with high validation accuracy and F1-scores, reliably distinguishing pedestrian and non-pedestrian images.
          </p>
          <p>
            <strong>Key insight:</strong> transfer learning combined with data augmentation enables robust performance even with a smaller dataset, improving generalisation significantly.
          </p>
          <p>
            However, slightly lower recall for pedestrian images suggests potential areas for improvement through:
          </p>
          <ul>
            <li>Increasing dataset size and diversity (e.g., crowded or low-light conditions).</li>
            <li>Exploring ensemble models or object detection approaches.</li>
            <li>Integrating real-time video pipeline for live monitoring.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section">
      <div class="section-title">Conclusion</div>
      <p>
        Overall, this project provides a solid foundation for reliable, AI-driven pedestrian detection. 
        The system demonstrates the real-world potential of deep learning in public safety monitoring and can be further enhanced for large-scale deployment in smart city infrastructure.
      </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
