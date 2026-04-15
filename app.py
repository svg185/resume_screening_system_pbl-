import io

import pandas as pd
import streamlit as st
from model import predict_resume, predict_resume_with_confidence
from PyPDF2 import PdfReader


st.set_page_config(page_title="AI Resume Screening System", layout="wide")

LOGO_PATH = "resume logo.png"


def extract_pdf_text(uploaded_file):
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def extract_file_text(uploaded_file):
    if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
        return extract_pdf_text(uploaded_file)

    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


st.markdown(
    """
    <style>
        .stApp {
            background:
                linear-gradient(135deg, rgba(244, 248, 255, 0.96), rgba(247, 252, 249, 0.96)),
                radial-gradient(circle at top left, rgba(65, 132, 228, 0.16), transparent 28%);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero {
            display: flex;
            align-items: center;
            gap: 1.4rem;
            padding: 1.5rem;
            border: 1px solid #d8e2f0;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 14px 36px rgba(38, 62, 92, 0.10);
            margin-bottom: 1.25rem;
        }

        .hero-title {
            color: #17324d;
            font-size: 2.25rem;
            line-height: 1.1;
            font-weight: 800;
            margin: 0;
        }

        .hero-subtitle {
            color: #4d647a;
            font-size: 1.02rem;
            margin: 0.6rem 0 0;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.9rem;
            margin: 1rem 0 1.4rem;
        }

        .metric-box, .result-box {
            border: 1px solid #d8e2f0;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.86);
            padding: 1rem;
            box-shadow: 0 10px 28px rgba(38, 62, 92, 0.08);
        }

        .metric-label {
            color: #60758a;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .metric-value {
            color: #17324d;
            font-size: 1.15rem;
            font-weight: 800;
            margin-top: 0.25rem;
        }

        .result-title {
            color: #60758a;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .result-category {
            color: #0f6b58;
            font-size: 1.9rem;
            font-weight: 800;
            margin-top: 0.3rem;
        }

        .developer-section {
            margin-top: 2rem;
            padding: 1.25rem;
            border: 1px solid #d8e2f0;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.86);
            box-shadow: 0 10px 28px rgba(38, 62, 92, 0.08);
        }

        .developer-heading {
            color: #17324d;
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 0.9rem;
        }

        .developer-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.85rem;
        }

        .developer-card {
            border: 1px solid #dbe6ef;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.95rem;
        }

        .developer-name {
            color: #17324d;
            font-size: 1.02rem;
            font-weight: 800;
        }

        .developer-role {
            color: #176b5b;
            font-size: 0.92rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.75rem 1rem;
            background: #ffffff;
            border: 1px solid #d8e2f0;
        }

        .stButton > button {
            border-radius: 8px;
            border: 0;
            background: #176b5b;
            color: white;
            font-weight: 700;
            padding: 0.65rem 1.15rem;
        }

        .stButton > button:hover {
            background: #115247;
            color: white;
        }

        textarea {
            border-radius: 8px !important;
            border-color: #c9d7e6 !important;
        }

        @media (max-width: 760px) {
            .hero {
                flex-direction: column;
                align-items: flex-start;
            }

            .hero-title {
                font-size: 1.7rem;
            }

            .metric-row {
                grid-template-columns: 1fr;
            }

            .developer-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

hero_logo, hero_text = st.columns([1, 5], vertical_alignment="center")
with hero_logo:
    st.image(LOGO_PATH, width=110)
with hero_text:
    st.markdown(
        """
        <div>
            <h1 class="hero-title">AI Resume Screening System</h1>
            <p class="hero-subtitle">Paste resume text, upload files, and get fast category predictions with confidence ranking.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-label">Pipeline</div>
            <div class="metric-value">Cleaning + TF-IDF</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Model</div>
            <div class="metric-value">Naive Bayes</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Inputs</div>
            <div class="metric-value">Text, PDF, TXT</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_text, tab_files = st.tabs(["Single Resume", "Multiple Resume Ranking"])

with tab_text:
    left, right = st.columns([2, 1], gap="large")

    with left:
        resume_text = st.text_area(
            "Paste Resume Text Here",
            height=300,
            placeholder="Example: Python developer with machine learning, data analysis, pandas, model evaluation...",
        )

        analyze_clicked = st.button("Analyze Resume", use_container_width=True)

    with right:
        st.markdown(
            """
            <div class="result-box">
                <div class="result-title">Prediction Result</div>
                <div class="result-category">Ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if analyze_clicked:
        if resume_text.strip():
            category = predict_resume(resume_text)
            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-title">Predicted Job Category</div>
                    <div class="result-category">{category}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Please enter resume text.")

with tab_files:
    uploaded_files = st.file_uploader(
        "Upload resumes",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more PDF or TXT resumes.",
    )

    if st.button("Rank Uploaded Resumes", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one resume.")
        else:
            results = []

            for uploaded_file in uploaded_files:
                text = extract_file_text(uploaded_file)
                if not text:
                    results.append(
                        {
                            "Resume": uploaded_file.name,
                            "Predicted Category": "No readable text",
                            "Confidence (%)": 0.0,
                        }
                    )
                    continue

                category, confidence = predict_resume_with_confidence(text)
                results.append(
                    {
                        "Resume": uploaded_file.name,
                        "Predicted Category": category,
                        "Confidence (%)": round(confidence * 100, 2),
                    }
                )

            ranked = pd.DataFrame(results).sort_values("Confidence (%)", ascending=False)
            st.subheader("Ranked Results")
            st.dataframe(ranked, use_container_width=True, hide_index=True)

st.markdown(
    """
    <div class="developer-section">
        <div class="developer-heading">Developed By</div>
        <div class="developer-grid">
            <div class="developer-card">
                <div class="developer-name">Rahul Sammal</div>
                <div class="developer-role">Frontend Designer</div>
            </div>
            <div class="developer-card">
                <div class="developer-name">Mayank Sammal</div>
                <div class="developer-role">Backend Developer</div>
            </div>
            <div class="developer-card">
                <div class="developer-name">Kartik Kumar</div>
                <div class="developer-role">Backend Developer</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
