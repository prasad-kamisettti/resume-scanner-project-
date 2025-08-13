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

from resume_parser import extract_text_from_pdf_bytes

@st.cache_resource
def get_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1)

@st.cache_data
def fit_and_transform(_vect, a, b):
    return _vect.fit_transform([a, b])



if uploaded_file and jd_input:
    if st.button("🔍 Match Resume with JD"):
        # Read uploaded file directly in memory
        resume_bytes = uploaded_file.read()
        resume_text = extract_text_from_pdf_bytes(resume_bytes)
        cleaned_resume = preprocess_text(resume_text)


        # Preprocess the jd 
        cleaned_jd = preprocess_text(jd_input)

        # TF-IDF (bigrams) + Cosine Similarity, with impact-sorted phrases
        vect = get_vectorizer()
        vectors = fit_and_transform(vect, cleaned_resume, cleaned_jd)

        similarity_score = cosine_similarity(vectors[0], vectors[1])[0, 0]
        percentage = round(similarity_score * 100, 2)
        st.subheader(f"✅ Match Score: {percentage}%")

        # Surface top matched phrases and high-impact missing terms
        vocab = vect.get_feature_names_out()
        arr = vectors.toarray()
        r, j = arr[0], arr[1]           # resume weights, JD weights
        impact = r * j                  # importance of overlap

        # Top matches (phrases that both contain, sorted by impact)
        top_idx = impact.argsort()[::-1]
        top_matches = [vocab[i] for i in top_idx if j[i] > 0 and r[i] > 0][:15]

        # Missing terms (present in JD, absent in resume), sorted by JD weight
        missing_idx = [i for i in range(len(vocab)) if j[i] > 0 and r[i] == 0]
        missing_terms = [vocab[i] for i in sorted(missing_idx, key=lambda i: j[i], reverse=True)][:20]

        if top_matches:
            st.markdown("**Top matched phrases:** " + ", ".join(f"`{t}`" for t in top_matches))

        if missing_terms:
            st.markdown("**High‑impact missing terms:** " + ", ".join(f"`{m}`" for m in missing_terms))
        else:
            st.success("Your resume covers the JD’s key phrases!")


else:
    st.info("Please upload a resume and paste a job description to get started.")
