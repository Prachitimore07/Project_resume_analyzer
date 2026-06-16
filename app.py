import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Function to extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# -----------------------------
# Function to calculate match
# -----------------------------
def calculate_match(resume_text, job_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0] * 100

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AI Resume Analyzer & Candidate Shortlisting")

job_description = st.text_area("Paste Job Description", height=200)

uploaded_resumes = st.file_uploader(
    "Upload Multiple Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

shortlist_threshold = st.slider(
    "Shortlist Threshold (%)",
    0, 100, 60
)

if st.button("Analyze Resumes"):
    if uploaded_resumes and job_description:
        results = []

        for resume in uploaded_resumes:
            resume_text = extract_text_from_pdf(resume)
            match_percentage = calculate_match(resume_text, job_description)

            results.append({
                "Candidate": resume.name,
                "Match Percentage": round(match_percentage, 2)
            })

        # Sort by match percentage (highest first)
        results = sorted(results, key=lambda x: x["Match Percentage"], reverse=True)

        st.subheader("📊 Resume Match Results")
        for r in results:
            st.write(f"📄 {r['Candidate']} → {r['Match Percentage']}%")

        # Shortlisted candidates
        st.subheader("✅ Shortlisted Candidates")
        shortlisted = [r for r in results if r["Match Percentage"] >= shortlist_threshold]

        if shortlisted:
            for s in shortlisted:
                st.success(f"{s['Candidate']} → {s['Match Percentage']}%")
        else:
            st.warning("No candidates meet the shortlist criteria.")

    else:
        st.warning("Please upload resumes and enter job description.")
