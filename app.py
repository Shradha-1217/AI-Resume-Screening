import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract text from PDF
def extract_text_from_pdf(file):
    """Extract text content from uploaded PDF file"""
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + " "
        return text.strip()
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}. Please check the file format and try again.")
        return ""

# Function to extract skills using spaCy (if available)
def extract_skills(text):
    """Extract relevant skills from resume text using NLP"""
    
    
    try:
        doc = nlp(text.lower())
        skills = set()
        
        # Common technical skills and keywords to look for
        skill_keywords = [
            "python", "java", "javascript", "sql", "machine learning", "data analysis", 
            "communication", "teamwork", "leadership", "project management", "agile",
            "react", "angular", "node.js", "django", "flask", "aws", "azure", "docker",
            "kubernetes", "git", "html", "css", "c++", "c#", "php", "ruby", "go",
            "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn", "tableau",
            "power bi", "excel", "powerpoint", "word", "linux", "windows", "macos"
        ]
        
        # Extract skills based on noun chunks and keywords
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            if any(keyword in chunk_text for keyword in skill_keywords):
                skills.add(chunk.text.title())
        
        # Also look for direct keyword matches
        for keyword in skill_keywords:
            if keyword in text.lower():
                skills.add(keyword.title())
        
        return list(skills)[:10]  # Limit to top 10 skills
    except Exception as e:
        st.warning(f"Error extracting skills: {str(e)}")
        return ["Skill extraction error"]

# Function to rank resumes
def rank_resumes(job_description, resumes):
    """Rank resumes based on similarity to job description using TF-IDF and cosine similarity"""
    if not job_description.strip() or not resumes:
        return [], []
    
    try:
        documents = [job_description] + resumes
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1, 2))
        vectors = vectorizer.fit_transform(documents)
        job_description_vector = vectors[0]
        resume_vectors = vectors[1:]
        cosine_similarities = cosine_similarity(job_description_vector, resume_vectors).flatten()
        return cosine_similarities, vectorizer.get_feature_names_out()
    except Exception as e:
        st.error(f"Error ranking resumes: {str(e)}")
        return [], []

# Function to highlight keywords in text
def highlight_keywords(text, keywords):
    """Highlight important keywords in resume text"""
    if not text or not keywords:
        return text
    
    highlighted_text = text
    # Use top 5 keywords to avoid over-highlighting
    for keyword in keywords[:5]:
        if len(keyword) > 2:  # Avoid highlighting very short words
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted_text = pattern.sub(f"<mark>{keyword}</mark>", highlighted_text)
    return highlighted_text

# Function to create downloadable CSV
def get_csv_download_link(df):
    """Generate download link for CSV export"""
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="resume_ranking.csv">Download CSV</a>'
    except Exception as e:
        st.error(f"Error creating CSV download: {str(e)}")
        return ""



# Custom CSS styling
st.markdown("""
    <style>
    .big-font {font-size: 40px !important; color: #1f77b4;}
    .subheader {color: #ff7f0e;}
    .highlight {background-color: #ffeb3b; padding: 2px 5px; border-radius: 3px;}
    .sidebar .sidebar-content {background-color: #f0f2f6;}
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<p class="big-font">AI-Powered Resume Screening & Ranking</p>', unsafe_allow_html=True)

# Sidebar for customization options
with st.sidebar:
    st.subheader("Customization Options")
    min_score = st.slider("Minimum Similarity Score", 0.0, 1.0, 0.1, 0.05)
    top_n = st.number_input("Show Top N Candidates", min_value=1, max_value=50, value=10)
    show_skills = st.checkbox("Extract and Show Skills")
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
    
    st.subheader("Instructions")
    st.info("""
    1. Enter a detailed job description
    2. Upload PDF resumes (multiple files supported)
    3. Adjust similarity threshold if needed
    4. View ranked results and export data
    """)

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<p class="subheader">Job Description</p>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Enter the job description here", 
        height=200,
        placeholder="Paste the complete job description including required skills, experience, and qualifications..."
    )
    
    st.markdown('<p class="subheader">Upload Resumes</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop your PDF resumes here", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload multiple PDF resumes to analyze and rank against the job description"
    )

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135768.png", width=150)
    st.info("Upload resumes and enter a job description to rank candidates instantly!")
    
    # Display some metrics if files are uploaded
    if uploaded_files:
        st.metric("Resumes Uploaded", len(uploaded_files))
    if job_description:
        st.metric("Job Description Length", f"{len(job_description.split())} words")

# Main processing logic
if uploaded_files and job_description.strip():
    with st.spinner("Analyzing resumes... This may take a moment."):
        resumes = []
        file_names = []
        processing_errors = []
        
        # Process uploaded files
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            if text and len(text.strip()) > 50:  # Ensure meaningful content
                resumes.append(text)
                file_names.append(file.name)
            else:
                processing_errors.append(file.name)

        # Display processing errors if any
        if processing_errors:
            st.warning(f"Could not extract meaningful text from: {', '.join(processing_errors)}")

        if resumes:
            # Rank resumes
            scores, feature_names = rank_resumes(job_description, resumes)
            
            if len(scores) > 0:
                # Create results dataframe
                results = pd.DataFrame({
                    "Resume": file_names, 
                    "Similarity Score": scores
                })
                
                # Filter by minimum score and sort
                filtered_results = results[results["Similarity Score"] >= min_score].sort_values(
                    by="Similarity Score", ascending=False
                )

                # Extract skills if enabled
                if show_skills and len(resumes) > 0:
                    with st.spinner("Extracting skills..."):
                        skills_list = []
                        for i, resume in enumerate(resumes):
                            if file_names[i] in filtered_results["Resume"].values:
                                skills = extract_skills(resume)
                                skills_list.append(", ".join(skills) if skills else "No skills extracted")
                            else:
                                skills_list.append("")
                        
                        # Add skills to filtered results only
                        skills_for_filtered = []
                        for resume_name in filtered_results["Resume"]:
                            idx = file_names.index(resume_name)
                            skills = extract_skills(resumes[idx])
                            skills_for_filtered.append(", ".join(skills) if skills else "No skills extracted")
                        
                        filtered_results = filtered_results.copy()
                        filtered_results["Key Skills"] = skills_for_filtered

                # Display results
                if not filtered_results.empty:
                    st.markdown('<p class="subheader">Top Candidates</p>', unsafe_allow_html=True)
                    top_results = filtered_results.head(top_n)
                    
                    # Display formatted dataframe
                    st.dataframe(
                        top_results.style.format({"Similarity Score": "{:.2%}"}),
                        use_container_width=True
                    )

                    # Visualization
                    if len(top_results) > 1:
                        st.markdown('<p class="subheader">Score Distribution</p>', unsafe_allow_html=True)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Create horizontal bar chart
                        bars = ax.barh(range(len(top_results)), top_results["Similarity Score"], 
                                      color=sns.color_palette("Blues_r", len(top_results)))
                        
                        # Customize chart
                        ax.set_yticks(range(len(top_results)))
                        ax.set_yticklabels([name.replace('.pdf', '') for name in top_results["Resume"]])
                        ax.set_xlabel("Similarity Score")
                        ax.set_title("Resume Ranking by Similarity Score")
                        
                        # Add value labels on bars
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                   f'{width:.2%}', ha='left', va='center')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                    # Detailed resume matches
                    with st.expander("View Detailed Resume Matches"):
                        for idx, row in top_results.iterrows():
                            resume_idx = file_names.index(row["Resume"])
                            resume_text = resumes[resume_idx]
                            
                            # Get relevant keywords from job description
                            job_keywords = job_description.split()[:15]  # Top 15 words
                            highlighted_text = highlight_keywords(resume_text[:800], job_keywords)
                            
                            st.markdown(f"**{row['Resume']} (Score: {row['Similarity Score']:.2%})**")
                            st.markdown(f'<div style="border-left: 3px solid #1f77b4; padding-left: 10px; margin: 10px 0;">{highlighted_text}...</div>', 
                                       unsafe_allow_html=True)
                            st.markdown("---")

                    # Export functionality
                    st.markdown('<p class="subheader">Export Results</p>', unsafe_allow_html=True)
                    export_data = filtered_results if not filtered_results.empty else results
                    
                    col_export1, col_export2, col_export3 = st.columns(3)
                    
                    with col_export1:
                        if export_format == "CSV" or st.button("Export CSV"):
                            csv_link = get_csv_download_link(export_data)
                            if csv_link:
                                st.markdown(csv_link, unsafe_allow_html=True)
                    
                    with col_export2:
                        if export_format == "Excel" or st.button("Export Excel"):
                            try:
                                buffer = BytesIO()
                                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                    export_data.to_excel(writer, index=False, sheet_name='Resume Rankings')
                                
                                st.download_button(
                                    label="Download Excel",
                                    data=buffer.getvalue(),
                                    file_name="resume_ranking.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            except Exception as e:
                                st.error(f"Error creating Excel file: {str(e)}")
                    
                    with col_export3:
                        if export_format == "JSON" or st.button("Export JSON"):
                            try:
                                json_str = export_data.to_json(orient="records", indent=2)
                                st.download_button(
                                    label="Download JSON",
                                    data=json_str,
                                    file_name="resume_ranking.json",
                                    mime="application/json"
                                )
                            except Exception as e:
                                st.error(f"Error creating JSON file: {str(e)}")

                else:
                    st.warning(f"No resumes meet the minimum similarity score of {min_score:.1%}. Try lowering the threshold.")
            else:
                st.error("Error occurred during resume ranking. Please check your input and try again.")
        else:
            st.error("No valid text extracted from uploaded PDFs. Please ensure your files are readable PDF documents.")

elif uploaded_files and not job_description.strip():
    st.warning("Please enter a job description to start ranking resumes.")
elif job_description.strip() and not uploaded_files:
    st.warning("Please upload PDF resumes to analyze.")
else:
    st.info("👆 Upload resumes and enter a job description to get started!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>AI Resume Ranker - Powered by Machine Learning & Natural Language Processing</p>
</div>
""", unsafe_allow_html=True)
