"""
ai_analyzer.py

Sends resume text to the Gemini API and returns structured feedback.
"""

import json
import re
import traceback

from google import genai


class AIAnalysisError(Exception):
    """Raised when the Gemini API call fails or returns an unusable response."""


REQUIRED_KEYS = [
    "score",
    "summary",
    "skills",
    "strengths",
    "weaknesses",
    "missing_skills",
    "role_suitability",
    "recommendations",
]

OPTIONAL_KEYS = [
    "ats_score",
    "improvement_suggestions",
    "job_match_score",
    "matching_skills",
    "missing_job_skills",
    "missing_keywords",
    "job_match_summary",
    "job_match_recommendations",
]

BASE_PROMPT_TEMPLATE = """You are an experienced professional resume reviewer and career coach.

Carefully analyse the resume text below and return your evaluation as a single
valid JSON object ONLY - no markdown formatting, no code fences, no extra commentary.

The JSON object must include these keys:
- "score": an integer from 0 to 100 rating overall resume quality
- "ats_score": an integer from 0 to 100 estimating ATS compatibility
- "summary": a 2-3 sentence professional summary of the candidate
- "skills": a list of key skills detected in the resume (strings)
- "strengths": a list of 3-5 notable strengths (strings)
- "weaknesses": a list of 3-5 areas that are weak or missing (strings)
- "missing_skills": a list of relevant skills the candidate should consider
  developing or adding for overall career readiness, independent of any specific
  job description (strings)
- "role_suitability": a short string naming 1-3 job roles this resume is best suited for
- "recommendations": a list of 3-5 concrete, actionable recommendations (strings)
- "improvement_suggestions": a list of 4-6 specific, actionable resume improvements (strings)
{job_schema}

ATS compatibility should consider keyword relevance, skills relevance, section
completeness, readability, formatting suitability for ATS, role relevance,
quantified achievements, and overall machine-readability.

This is an AI-based estimate, not a guaranteed hiring score or a commercial ATS scan.
Keep recommendations specific and actionable.
Keep the sections internally consistent. Do not list a skill as both detected and
missing unless the resume only mentions it weakly and the explanation makes that
clear. Avoid repeating the same wording across strengths, weaknesses, missing
skills, and recommendations.
{job_instruction}

Resume text:
\"\"\"
{resume_text}
\"\"\"
{job_description_block}

Respond with ONLY the JSON object.
"""

JOB_SCHEMA = """- "job_match_score": an integer from 0 to 100 estimating match with the job description
- "matching_skills": a list of skills detected in the resume that satisfy the job description
- "missing_job_skills": a list of specific skills required or preferred in the
  job description that are not detected in the resume
- "missing_keywords": a list of important job-description keywords, tools,
  qualifications, or requirements that are absent from the resume and could
  affect ATS matching
- "job_match_summary": a short explanation of the match
- "job_match_recommendations": a list of specific recommendations to improve the resume for this job"""

JOB_INSTRUCTION = (
    "Compare the resume against the job description only because one was provided. "
    "Keep job-specific matching separate from the general resume analysis. Use "
    "\"missing_skills\" for overall career-readiness gaps, \"missing_job_skills\" "
    "for job-description skills absent from the resume, and \"missing_keywords\" "
    "for exact keywords, tools, qualifications, or requirements from the job "
    "description that may affect ATS matching. Do not copy the same list into "
    "these three fields."
)


def _build_prompt(resume_text: str, job_description: str = "") -> str:
    clean_job_description = (job_description or "").strip()
    has_job_description = bool(clean_job_description)

    return BASE_PROMPT_TEMPLATE.format(
        resume_text=resume_text[:12000],
        job_schema=JOB_SCHEMA if has_job_description else "",
        job_instruction=JOB_INSTRUCTION if has_job_description else "",
        job_description_block=(
            f'\nJob description:\n"""\n{clean_job_description[:6000]}\n"""\n'
            if has_job_description
            else ""
        ),
    )


def _extract_json(raw_text: str) -> dict:
    """Pull a JSON object out of the model's response, tolerating stray formatting."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise AIAnalysisError(
        "The AI response could not be understood. Please try analyzing again."
    )


def _score(value, default: int = 0) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = default
    return max(0, min(100, number))


def _string(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_string(item) for item in value if _string(item)]
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    return [_string(value)] if _string(value) else []


def _dedupe(items: list[str]) -> list[str]:
    unique_items = []
    seen = set()
    for item in items:
        normalized = item.casefold().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_items.append(item)
    return unique_items


def _without_exact_matches(items: list[str], excluded: list[str]) -> list[str]:
    excluded_values = {item.casefold().strip() for item in excluded}
    return [item for item in items if item.casefold().strip() not in excluded_values]


def _validate_and_normalize(result: dict, has_job_description: bool) -> dict:
    if not isinstance(result, dict):
        raise AIAnalysisError(
            "The AI response could not be understood. Please try analyzing again."
        )

    missing = [key for key in REQUIRED_KEYS if key not in result]
    if missing:
        raise AIAnalysisError(
            "The AI response was missing expected fields. Please try analyzing again."
        )

    for key in OPTIONAL_KEYS:
        result.setdefault(key, None)

    normalized = {
        "score": _score(result.get("score")),
        "ats_score": _score(result.get("ats_score", result.get("score"))),
        "summary": _string(result.get("summary"), "No summary was returned."),
        "skills": _dedupe(_list(result.get("skills"))),
        "strengths": _dedupe(_list(result.get("strengths"))),
        "weaknesses": _dedupe(_list(result.get("weaknesses"))),
        "missing_skills": _dedupe(_list(result.get("missing_skills"))),
        "role_suitability": _string(result.get("role_suitability"), "Not specified."),
        "recommendations": _dedupe(_list(result.get("recommendations"))),
        "improvement_suggestions": _dedupe(
            _list(result.get("improvement_suggestions"))
        ),
        "job_match_score": _score(result.get("job_match_score")),
        "matching_skills": _dedupe(_list(result.get("matching_skills"))),
        "missing_job_skills": _dedupe(_list(result.get("missing_job_skills"))),
        "missing_keywords": _dedupe(_list(result.get("missing_keywords"))),
        "job_match_summary": _string(result.get("job_match_summary")),
        "job_match_recommendations": _dedupe(
            _list(result.get("job_match_recommendations"))
        ),
        "has_job_match": has_job_description,
    }

    if not normalized["improvement_suggestions"]:
        normalized["improvement_suggestions"] = normalized["recommendations"]

    if normalized["has_job_match"]:
        normalized["missing_job_skills"] = _without_exact_matches(
            normalized["missing_job_skills"],
            normalized["matching_skills"],
        )
        normalized["missing_keywords"] = _without_exact_matches(
            normalized["missing_keywords"],
            normalized["missing_job_skills"] + normalized["matching_skills"],
        )
        normalized["missing_skills"] = _without_exact_matches(
            normalized["missing_skills"],
            normalized["missing_job_skills"] + normalized["missing_keywords"],
        )
    else:
        normalized["job_match_score"] = 0
        normalized["matching_skills"] = []
        normalized["missing_job_skills"] = []
        normalized["missing_keywords"] = []
        normalized["job_match_summary"] = ""
        normalized["job_match_recommendations"] = []

    return normalized


def analyze_resume(
    resume_text: str,
    api_key: str,
    model_name: str = "gemini-3.6-flash",
    job_description: str = "",
) -> dict:
    """
    Send resume text to Gemini and return a structured analysis dict.

    Args:
        resume_text: Extracted plain text of the resume.
        api_key: Gemini API key (from st.secrets).
        model_name: Gemini model to use.
        job_description: Optional job description for targeted match analysis.

    Returns:
        Dict containing resume feedback, ATS score, and optional job-match fields.

    Raises:
        AIAnalysisError: If the API call fails or the response is unusable.
    """
    if not api_key:
        raise AIAnalysisError(
            "Gemini API key is missing. Please configure GEMINI_API_KEY in Streamlit secrets."
        )

    clean_job_description = (job_description or "").strip()

    try:
        client = genai.Client(api_key=api_key)
        prompt = _build_prompt(resume_text, clean_job_description)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        raw_text = response.text

    except Exception as exc:
        print("GEMINI API ERROR")
        print(f"Exception type: {type(exc).__name__}")
        print(f"Exception message: {exc}")
        print(f"Requested model: {model_name}")
        print(f"Job description provided: {bool(clean_job_description)}")
        print("Traceback:")
        traceback.print_exc()
        raise AIAnalysisError(
            "The AI analysis service is currently unavailable. Please try again in a moment."
        ) from exc

    if not raw_text:
        raise AIAnalysisError(
            "The AI returned an empty response. Please try analyzing again."
        )

    result = _extract_json(raw_text)
    return _validate_and_normalize(result, bool(clean_job_description))
