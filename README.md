# AI Resume Analyser

## Overview
AI Resume Analyser is a small Streamlit portfolio project that reviews a PDF
resume with Google's Gemini API and returns structured, easy-to-scan feedback.

It estimates resume quality, ATS compatibility, skills, strengths, weaknesses,
missing skills, suitable roles, and practical improvement suggestions. Users can
also paste a job description to get a simple AI-based resume-to-job match.

## Features
- PDF resume upload
- Text extraction from PDF resumes using `pypdf`
- Gemini-powered structured JSON analysis
- Resume Quality Score from 0 to 100
- ATS Compatibility Score from 0 to 100
- Professional summary
- Skills detected
- Strengths and weaknesses
- Missing skills
- Best-suited roles
- Actionable improvement suggestions
- Optional job-description match analysis
- Job Match percentage
- Matching skills, missing job skills, and missing keywords
- Reset button to analyse another resume without refreshing the browser
- Friendly error handling for invalid PDFs, scanned PDFs, missing API keys, and AI failures

## Tech Stack
- Python
- Streamlit
- Gemini API through `google-genai`
- PDF text extraction with `pypdf`

## How It Works
```text
Upload Resume -> Extract Text -> Gemini Analysis -> Validate JSON -> Streamlit Results
```

For job matching:

```text
Paste Job Description -> Gemini Match Analysis -> Skill Gap Results
```

The ATS and job-match scores are AI-generated estimates for guidance. They are
not guaranteed hiring scores and are not a replacement for a commercial ATS.

## Installation
```bash
git clone <your-repo-url>
cd ai-resume-analyser
pip install -r requirements.txt
```

## Gemini API Key Setup
This app reads your Gemini API key from Streamlit secrets. Never hardcode the key.

1. Create a `.streamlit` folder in the project root if it does not already exist.
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
3. Add your key:

```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

`.streamlit/secrets.toml` is ignored by Git and should not be committed.

## Running Locally
```bash
python -m streamlit run app.py
```

If `python` is not available on your machine, use the Python command configured
for your environment.

## Deploying To Streamlit Community Cloud
1. Push the project to GitHub.
2. Create a new app on Streamlit Community Cloud.
3. Set the main file path to `app.py`.
4. Add `GEMINI_API_KEY` under **App settings -> Secrets**.
5. Deploy the app.

## Future Improvements
- Support `.docx` resume uploads
- Add a downloadable analysis report
- Add optional resume section detection details
- Improve scoring consistency with a small local rule-based pre-check
