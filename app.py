"""
AI Resume Analyser

A small Streamlit app that analyses a PDF resume with Gemini and returns
structured feedback, ATS compatibility, and optional job-description matching.
"""

import html

import streamlit as st

from ai_analyzer import AIAnalysisError, analyze_resume
from resume_parser import ResumeParseError, extract_text_from_pdf


st.set_page_config(
    page_title="AI Resume Analyser",
    page_icon="📄",
    layout="centered",
)

CUSTOM_CSS = """
<style>
    .main .block-container {
        max-width: 860px;
        padding-top: 2rem;
    }
    .app-subtitle {
        color: #6b7280;
        font-size: 1.02rem;
        margin: -0.6rem 0 1.4rem;
    }
    .score-card {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        background: #ffffff;
    }
    .score-label {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
    .score-value {
        color: #111827;
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .skill-pill {
        display: inline-block;
        background: #eef2ff;
        color: #3730a3;
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        margin: 0.2rem 0.25rem 0.2rem 0;
        font-size: 0.86rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def initialize_state() -> None:
    defaults = {
        "result": None,
        "resume_text": "",
        "upload_key": 0,
        "job_description_input": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_analysis() -> None:
    st.session_state["result"] = None
    st.session_state["resume_text"] = ""
    st.session_state["job_description_input"] = ""
    st.session_state["upload_key"] += 1
    st.rerun()


def valid_job_description(job_description: str) -> bool:
    return not job_description.strip() or len(job_description.strip()) >= 40


def render_score_card(label: str, score: int, help_text: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="score-card">
            <div class="score-label">{html.escape(label)}</div>
            <div class="score-value">{score} / 100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(score / 100, text=help_text)


def render_pills(items: list[str], empty_text: str = "No items found.") -> None:
    if not items:
        st.write(empty_text)
        return

    pills = "".join(
        f'<span class="skill-pill">{html.escape(str(item))}</span>' for item in items
    )
    st.markdown(pills, unsafe_allow_html=True)


def render_list(items: list[str], empty_text: str = "No details returned.") -> None:
    if not items:
        st.write(empty_text)
        return

    for item in items:
        st.markdown(f"- {item}")


def run_analysis() -> bool:
    job_description = st.session_state["job_description_input"]

    if job_description.strip() and not valid_job_description(job_description):
        st.warning("Please paste a more complete job description, or leave it blank.")
        return False

    with st.spinner("Analyzing with Gemini..."):
        try:
            st.session_state["result"] = analyze_resume(
                st.session_state["resume_text"],
                api_key,
                job_description=st.session_state["job_description_input"],
            )
        except AIAnalysisError as exc:
            st.error(f"Analysis failed: {exc}")
            return False

    return True


initialize_state()

st.title("📄 AI Resume Analyser")
st.markdown(
    '<div class="app-subtitle">Upload a PDF resume to get AI-powered feedback, '
    "an ATS compatibility estimate, and optional job-description matching.</div>",
    unsafe_allow_html=True,
)

api_key = st.secrets.get("GEMINI_API_KEY", None)
if not api_key:
    st.error(
        "Gemini API key is not configured. Add `GEMINI_API_KEY` to your Streamlit secrets to use this app."
    )
    st.stop()

result = st.session_state.get("result")

if not result:
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)",
        type=["pdf"],
        key=f"resume_upload_{st.session_state['upload_key']}",
    )

    st.text_area(
        "Optional job description",
        key="job_description_input",
        height=180,
        placeholder="Paste a job description here if you want a targeted match analysis.",
    )

    analyze_clicked = st.button(
        "Analyze Resume",
        type="primary",
        disabled=uploaded_file is None,
    )

    if analyze_clicked:
        with st.spinner("Reading your resume..."):
            try:
                st.session_state["resume_text"] = extract_text_from_pdf(uploaded_file)
            except ResumeParseError as exc:
                st.error(f"PDF could not be read: {exc}")
                st.stop()

        if not st.session_state["resume_text"].strip():
            st.warning("No readable resume text was found. Try uploading a text-based PDF.")
            st.stop()

        run_analysis()

    if not st.session_state.get("result"):
        st.info("Upload a resume and click **Analyze Resume** to get started.")

result = st.session_state.get("result")

if result:
    st.divider()
    st.subheader("Resume Analysis")

    st.info(
        "Scores are AI-generated estimates meant for guidance, not guaranteed hiring outcomes."
    )

    score_col, ats_col = st.columns(2)
    with score_col:
        render_score_card(
            "Resume Quality Score",
            result.get("score", 0),
            "Overall resume quality",
        )
    with ats_col:
        render_score_card(
            "ATS Compatibility Score",
            result.get("ats_score", 0),
            "AI-based ATS compatibility estimate",
        )

    st.subheader("Professional Summary")
    st.write(result.get("summary", "No summary was returned."))

    with st.expander("Skills Detected", expanded=True):
        render_pills(result.get("skills", []))

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Strengths", expanded=True):
            render_list(result.get("strengths", []))
    with col2:
        with st.expander("Weaknesses", expanded=True):
            render_list(result.get("weaknesses", []))

    with st.expander("Missing Skills", expanded=True):
        st.caption("Career-readiness skills to develop or add beyond this resume.")
        render_pills(result.get("missing_skills", []), "No missing skills were returned.")

    st.subheader("Best-Suited Roles")
    st.write(result.get("role_suitability", "Not specified."))

    with st.expander("Improvement Suggestions", expanded=True):
        suggestions = result.get("improvement_suggestions") or result.get("recommendations", [])
        render_list(suggestions)

    st.divider()
    st.subheader("Job Description Match")

    if result.get("has_job_match"):
        st.metric("Job Match Score", f"{result.get('job_match_score', 0)}%")
        st.progress(
            result.get("job_match_score", 0) / 100,
            text="AI-generated job match estimate",
        )
        st.write(result.get("job_match_summary") or "No match summary was returned.")

        match_col, gap_col = st.columns(2)
        with match_col:
            with st.expander("Skills I Have", expanded=True):
                st.caption(
                    "Detected resume skills that match the supplied job description."
                )
                render_pills(
                    result.get("matching_skills", []),
                    "No matching skills were returned.",
                )
        with gap_col:
            with st.expander("Skills Missing", expanded=True):
                st.caption("Job-description skills that were not detected in the resume.")
                render_pills(
                    result.get("missing_job_skills", []),
                    "No missing job skills were returned.",
                )

        with st.expander("Important Keywords Missing", expanded=True):
            st.caption(
                "Absent job-description keywords, tools, qualifications, or "
                "requirements that may affect ATS matching."
            )
            render_pills(result.get("missing_keywords", []), "No missing keywords were returned.")

        with st.expander("How to Improve for This Job", expanded=True):
            render_list(result.get("job_match_recommendations", []))
    else:
        st.info(
            "No job description was provided, so job-specific matching was skipped."
        )

    st.divider()
    if st.button("Analyze Another Resume", type="primary"):
        reset_analysis()
