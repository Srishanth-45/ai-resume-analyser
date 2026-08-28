# 📄 AI Resume Analyser

An AI-powered resume analysis application built with Python and Streamlit. It analyzes PDF resumes, evaluates resume quality and ATS compatibility, identifies skills, strengths, weaknesses, missing skills, and recommends suitable career roles.

It also provides targeted job-description matching using Google's Gemini API.

## 🚀 Live Demo

[Open the AI Resume Analyser](YOUR_STREAMLIT_APP_URL)

## 📌 Project Overview

AI Resume Analyser helps students and job seekers understand how well their resume represents their skills and experience.

Users can upload a PDF resume and receive AI-generated feedback including:

- Resume Quality Score
- ATS Compatibility Score
- Professional Summary
- Detected Skills
- Strengths
- Weaknesses
- Missing Skills
- Best-Suited Roles
- Resume Improvement Suggestions

Users can also provide a job description to receive a targeted compatibility analysis.

## ✨ Features

### 📄 Resume Analysis
- Upload a resume in PDF format
- Extract resume content automatically
- Generate an AI-powered professional summary
- Evaluate overall resume quality
- Estimate ATS compatibility

### 🛠️ Skills Analysis
- Detect technical and professional skills
- Identify strengths
- Identify weaknesses
- Suggest missing skills
- Highlight areas for career development

### 💼 Career Recommendations
- Recommend suitable entry-level roles
- Match resume skills with potential career paths
- Provide actionable resume improvement suggestions

### 🎯 Job Description Matching
When a job description is provided, the application performs targeted analysis:

- Job Match Score
- Skills the candidate already has
- Skills missing from the resume
- Important missing keywords
- Job-specific improvement suggestions

If no job description is provided, the job-specific analysis is automatically skipped.

### 🤖 AI-Powered Analysis
The application uses Google's Gemini API to generate contextual resume feedback rather than relying only on fixed rules.

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **Google Gemini API**
- **PyPDF2**
- **Pandas**
- **python-dotenv / Streamlit Secrets**

## 📂 Project Structure

```text
ai-resume-analyser/
│
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
│
├── ai_analyzer.py
├── app.py
├── resume_parser.py
├── requirements.txt
├── README.md
└── .gitignore
