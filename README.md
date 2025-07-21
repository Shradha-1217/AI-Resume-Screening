# 📄 AI Resume Screening

The **AI Resume Screening** app is an intelligent web tool that automatically evaluates the relevance of resumes to a given job description using **Natural Language Processing (NLP)**. Whether you're an HR professional or a developer building hiring tools, this app simplifies initial resume shortlisting using AI.

Built with **Streamlit**, the application provides an intuitive interface to upload resumes (PDF), input job descriptions, and visualize similarity scores via charts and heatmaps.

---

## 🚀 Features

* 📄 **PDF Resume Upload**: Upload multiple resumes in PDF format for evaluation.
* 📝 **Job Description Input**: Paste a job description against which all resumes will be evaluated.
* 🤖 **AI-Powered Relevance Scoring**: Uses TF-IDF vectorization and cosine similarity to calculate how well each resume matches the job description.
* 📊 **Visual Results**: Displays a bar chart and similarity matrix for easy comparison of all uploaded resumes.
* 📥 **Downloadable Results**: Optionally download results or copy for reporting (can be extended).
* 🔍 **Skill Extraction (NLP)**: Extracts potential skill keywords (optional, depending on spaCy usage).
* ⚠️ **Error Handling**: Displays clear errors for unreadable files or missing inputs.

---

## 🧠 Technologies Used

* **Python** – Core language for logic and processing.
* **Streamlit** – UI framework to create a web app quickly.
* **PyPDF2** – For reading text from uploaded PDF files.
* **scikit-learn** – For TF-IDF vectorization and cosine similarity calculations.
* **Pandas** – For data handling and score tabulation.
* **Matplotlib & Seaborn** – For visual charts and heatmaps.
* **spaCy (optional)** – For advanced NLP and keyword extraction (can be added manually).

---

## 🛠 Installation

### Prerequisites

* Python 3.7 or higher
* pip installed
* Resume files in PDF format

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/ai-resume-screening.git
   cd ai-resume-screening
   ```

2. **Create Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate     # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**

   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`.

---

## 🎯 Usage Instructions

1. **Launch the App**: Use the terminal command above to run the Streamlit server.
2. **Upload Resumes**: Drag and drop or select multiple PDF resumes.
3. **Paste Job Description**: Input a job description in the provided text area.
4. **Click "Analyze" or "Compare"**: The app will extract and vectorize text, then calculate similarity.
5. **View Results**:

   * **Bar Chart**: Shows similarity scores by resume filename.
   * **Heatmap**: Matrix showing score relationships (optional).
6. *(Optional)* Extend or export results for integration with ATS systems.

---

## 📁 Project Structure

```
AI-Resume-Screening/
│
├── app.py                  # Streamlit app logic
├── requirements.txt        # Dependencies list
├── pyproject.toml          # Python project build system (if used)
├── uv.lock                 # Dependency version locking (for poetry/pipenv)
└── generated-icon.png      # Optional icon for UI
```

---

## ⚠️ Troubleshooting

| Problem               | Solution                                               |
| --------------------- | ------------------------------------------------------ |
| Resume not processing | Ensure it's in PDF format and not scanned images       |
| App won’t run         | Ensure Python and dependencies are correctly installed |
| Similarity score is 0 | Resume may lack relevant keywords or content           |
| Heatmap missing       | Some visualizations may depend on installed packages   |

---

## 💡 Future Improvements

* 🌐 Export results as Excel or PDF reports
* 🧠 Integrate LLMs (like GPT/Gemini) for smarter screening
* 🎨 Tag cloud of most frequent skills per resume
* ✅ Integration with job portals or ATS
* 🌍 Multilingual resume screening

---

## 📝 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

* **Streamlit** – for the easy-to-use UI framework.
* **Google & OpenAI NLP Community** – inspiration for AI-based resume parsing.
* **scikit-learn** – for vectorization and similarity scoring.

