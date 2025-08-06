from resume_parser import extract_text_from_pdf
from utils import preprocess_text

# Extract raw text from the resume PDF
pdf_path = "sample_resume.pdf"  # make sure this file is in the same folder
text = extract_text_from_pdf(pdf_path)

# Print the raw resume text
print("\nRAW RESUME TEXT:\n")
print(text)

#Preprocess (clean) the resume text
cleaned_text = preprocess_text(text)

#Print the cleaned resume text
print("\nCLEANED RESUME TEXT:\n")
print(cleaned_text)

from resume_parser import extract_text_from_pdf
from utils import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Paste a sample job description
job_description = """
We are looking for a Data Engineer with experience in Python, SQL, ETL pipelines, and cloud platforms like AWS or Azure.
Experience with data modeling, analytics, and tools like Spark or Hadoop is a plus.
"""

#Clean the job description
cleaned_jd = preprocess_text(job_description)

#Compare resume and JD using TF-IDF + Cosine Similarity
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([cleaned_text, cleaned_jd])

#Calculate similarity
similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]  # value between 0 and 1
percentage = round(similarity_score * 100, 2)  # convert to percentage

#Show match score
print("\n🔍 MATCH SCORE WITH JD: ", percentage, "%")

#Show keywords in JD not in Resume
jd_words = set(cleaned_jd.split())
resume_words = set(cleaned_text.split())

missing_keywords = jd_words - resume_words

print("\n Missing Keywords from Resume: ")
print(", ".join(missing_keywords))

