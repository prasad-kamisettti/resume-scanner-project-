import streamlit as st
from resume_parser import extract_text_from_pdf
from utils import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Matcher", layout="centered")

st.title("📄 Resume vs Job Description Matcher")

# Upload PDF Resume
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

# Paste Job Description
jd_input = st.text_area("Paste the Job Description")

if uploaded_file and jd_input:
    if st.button("🔍 Match Resume with JD"):
        # Save uploaded file
        with open("temp_resume.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Extract and preprocess the resume
        resume_text = extract_text_from_pdf("temp_resume.pdf")
        cleaned_resume = preprocess_text(resume_text)

        # Preprocess the jd 
        cleaned_jd = preprocess_text(jd_input)

        # TF-IDF + Cosine Similarity
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([cleaned_resume, cleaned_jd])
        similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]
        percentage = round(similarity_score * 100, 2)

        # Show match score
        st.subheader(f"✅ Match Score: {percentage}%")

        # Show missing keywords
        resume_words = set(cleaned_resume.split())
        jd_words = set(cleaned_jd.split())
        missing_keywords = sorted(jd_words - resume_words)

        if missing_keywords:
            st.subheader("❌ Missing Keywords from Resume:")
            st.write(", ".join(missing_keywords))
        else:
            st.success("Your resume contains all keywords from the JD!")

else:
    st.info("Please upload a resume and paste a job description to get started.")
