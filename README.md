# 📄 AI Resume Analyser

An AI-powered Resume Analyser built with Python and Streamlit that helps students and job seekers understand the strengths, weaknesses, ATS compatibility, missing skills, and job suitability of their resumes.

## 🚀 Live Demo

👉 [Open the AI Resume Analyser](https://ai-resume-analyser-csssn.streamlit.app/)

---

## ✨ Features

- 📄 Upload resumes in PDF format
- 🤖 AI-powered resume analysis
- 📊 Resume Quality Score
- 🎯 ATS Compatibility Score
- 🧠 AI-generated Professional Summary
- 🛠️ Skills Detection
- 💪 Resume Strengths
- ⚠️ Resume Weaknesses
- 📚 Missing Skills Identification
- 💼 Best-Suited Job Roles
- 💡 Personalized Resume Improvement Suggestions
- 📝 Job Description Matching
- 🔍 Identify skills missing for a specific job
- 🔑 Detect important keywords missing from the resume
- 📈 Job Match Score

---

## 🖥️ How It Works

1. Upload your resume as a PDF.
2. Optionally enter a job description.
3. Click **Analyze Resume**.
4. The application extracts and analyses the resume content.
5. AI generates:
   - Resume quality score
   - ATS compatibility estimate
   - Detected skills
   - Strengths and weaknesses
   - Missing skills
   - Suitable job roles
   - Improvement suggestions
6. If a job description is provided, the application also performs a targeted job match analysis.

---

## 🧰 Technologies Used

- **Python**
- **Streamlit**
- **Google Gemini API**
- **PDF Processing**
- **AI / Natural Language Processing**
- **Data Analysis**

---

## 📂 Project Structure

```text
ai-resume-analyser/
│
├── app.py
├── ai_analyzer.py
├── resume_parser.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml
