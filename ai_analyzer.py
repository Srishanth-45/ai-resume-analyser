"""
ai_analyzer.py

Sends resume text to the Gemini API and returns structured feedback.
"""

import json
import re
import traceback
from collections import Counter

from google import genai
from google.genai import types


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

RESUME_SCORING_RUBRIC = """Use this fixed scoring rubric every time. Base scores
only on evidence in the resume. Do not add random variation,
optimism/pessimism adjustments, or style-based guesswork. For identical resume
text, the same rubric should produce the same scores. Before choosing each
score, evaluate each factor below independently, apply the exact weights, round
only the final weighted score, and keep the result within 0-100.

Resume Quality Score ("score") = weighted average of:
- Content relevance and role focus: 25%
- Clarity, structure, and readability: 20%
- Specific achievements, impact, and quantified results: 20%
- Skills, tools, and experience depth shown by resume evidence: 20%
- Professional completeness, grammar, and presentation quality: 15%

ATS Compatibility Score ("ats_score") = weighted average of:
- Clear standard sections and machine-readable organization: 25%
- Relevant keywords and skills visible in the resume: 25%
- Simple ATS-friendly formatting and readable wording: 20%
- Complete work/education/project details where applicable: 15%
- Role alignment and searchable terminology: 15%

Round each final weighted score to the nearest integer and keep it within 0-100.
Use the full range when justified, but avoid changing a score unless the evidence
or rubric factor changed."""

JOB_SCORING_RUBRIC = """When a job description is provided, use this fixed
job-match scoring rubric. Base the score only on evidence in the resume compared
with requirements and preferences in the supplied job description.

When a job description is provided, Job Match Score ("job_match_score") =
weighted average of:
- Required skills and qualifications matched by resume evidence: 45%
- Preferred skills, tools, and domain keywords matched: 20%
- Relevant experience level, responsibilities, and project evidence: 20%
- Resume keyword coverage for ATS matching against this job: 10%
- Missing critical requirements penalty: 5%

Round the final weighted score to the nearest integer and keep it within 0-100.
For identical resume text and identical job description, the same rubric should
produce the same job match score. Required or must-have criteria from the job
description should drive the score more than broad role similarity."""

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

{resume_scoring_rubric}

ATS compatibility should consider keyword relevance, skills relevance, section
completeness, readability, formatting suitability for ATS, role relevance,
quantified achievements, and overall machine-readability.

The application will verify and normalize the final scores with the same fixed
weights, so make the supporting analysis evidence-based and internally
consistent rather than trying to tune scores impressionistically.

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
    "these three fields.\n\n"
    f"{JOB_SCORING_RUBRIC}"
)

SECTION_ALIASES = {
    "summary": ("summary", "profile", "objective", "about"),
    "skills": ("skills", "technical skills", "core skills", "competencies"),
    "experience": ("experience", "work experience", "employment", "career history"),
    "education": ("education", "academic", "qualification", "qualifications"),
    "projects": ("projects", "portfolio"),
    "certifications": ("certifications", "certificates", "licenses"),
}

ACTION_VERBS = {
    "achieved",
    "automated",
    "built",
    "collaborated",
    "created",
    "delivered",
    "designed",
    "developed",
    "drove",
    "enhanced",
    "implemented",
    "improved",
    "increased",
    "launched",
    "led",
    "managed",
    "optimized",
    "reduced",
    "shipped",
    "streamlined",
}

COMMON_SKILL_TERMS = {
    "account management",
    "agile",
    "analytics",
    "aws",
    "azure",
    "business analysis",
    "c++",
    "communication",
    "content strategy",
    "crm",
    "css",
    "customer service",
    "data analysis",
    "data visualization",
    "django",
    "docker",
    "excel",
    "figma",
    "flask",
    "git",
    "google analytics",
    "html",
    "java",
    "javascript",
    "kubernetes",
    "leadership",
    "machine learning",
    "marketing",
    "node.js",
    "power bi",
    "product management",
    "project management",
    "python",
    "react",
    "sales",
    "sql",
    "tableau",
    "typescript",
    "ux",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "our",
    "the",
    "to",
    "with",
    "you",
    "your",
}

REQUIRED_MARKERS = (
    "required",
    "must",
    "need",
    "needs",
    "minimum",
    "mandatory",
    "essential",
    "qualification",
    "qualifications",
)

PREFERRED_MARKERS = (
    "preferred",
    "nice to have",
    "bonus",
    "plus",
    "desirable",
    "advantage",
)


def _build_prompt(resume_text: str, job_description: str = "") -> str:
    clean_job_description = (job_description or "").strip()
    has_job_description = bool(clean_job_description)

    return BASE_PROMPT_TEMPLATE.format(
        resume_text=resume_text[:12000],
        job_schema=JOB_SCHEMA if has_job_description else "",
        resume_scoring_rubric=RESUME_SCORING_RUBRIC,
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


def _weighted_score(factors: list[tuple[float, float]]) -> int:
    total_weight = sum(weight for weight, _ in factors) or 1
    total = sum(weight * max(0, min(100, score)) for weight, score in factors)
    return _score(total / total_weight)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9.+#-]*", text.casefold())


def _token_set(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in STOPWORDS}


def _ratio_score(count: int, excellent: int, partial_floor: int = 0) -> float:
    if count <= partial_floor:
        return 0
    return min(100, (count / excellent) * 100)


def _has_section(text: str, aliases: tuple[str, ...]) -> bool:
    for alias in aliases:
        pattern = rf"(?im)^\s*(?:[-*#]+\s*)?{re.escape(alias)}\s*:?\s*$"
        if re.search(pattern, text):
            return True
        if re.search(rf"(?i)\b{re.escape(alias)}\b", text):
            return True
    return False


def _section_score(resume_text: str) -> float:
    found = sum(
        1 for aliases in SECTION_ALIASES.values() if _has_section(resume_text, aliases)
    )
    return _ratio_score(found, excellent=5)


def _extract_known_terms(text: str) -> set[str]:
    lowered = text.casefold()
    return {term for term in COMMON_SKILL_TERMS if term in lowered}


def _keyword_richness_score(text: str) -> float:
    significant_tokens = [
        token for token in _tokens(text) if len(token) >= 3 and token not in STOPWORDS
    ]
    known_terms = _extract_known_terms(text)
    unique_keywords = set(significant_tokens) | known_terms
    return _ratio_score(len(unique_keywords), excellent=85)


def _length_score(word_count: int) -> float:
    if word_count < 120:
        return 35
    if word_count < 250:
        return 65
    if word_count <= 900:
        return 100
    if word_count <= 1300:
        return 85
    return 70


def _bullet_count(text: str) -> int:
    return len(re.findall(r"(?m)^\s*(?:[-*\u2022]|\d+[.)])\s+", text))


def _quantified_count(text: str) -> int:
    pattern = (
        r"(?i)(?:\b\d+(?:[.,]\d+)?\s*%|\$\s?\d+|\b\d+x\b|"
        r"\b\d+\+?\s*(?:users|customers|projects|clients|hours|days|weeks|"
        r"months|years|members|revenue|sales|costs|tickets|requests)\b)"
    )
    return len(re.findall(pattern, text))


def _action_verb_count(text: str) -> int:
    counts = Counter(_tokens(text))
    return sum(counts[verb] for verb in ACTION_VERBS)


def _readability_score(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0
    overly_long = sum(1 for line in lines if len(line) > 140)
    long_line_penalty = (overly_long / len(lines)) * 35
    ascii_chars = sum(1 for char in text if ord(char) < 128 or char.isspace())
    ascii_ratio = ascii_chars / max(1, len(text))
    return max(0, min(100, 100 - long_line_penalty - ((1 - ascii_ratio) * 30)))


def _resume_quality_score(resume_text: str) -> int:
    word_count = len(_tokens(resume_text))
    sections = _section_score(resume_text)
    keyword_richness = _keyword_richness_score(resume_text)
    bullets = _ratio_score(_bullet_count(resume_text), excellent=10)
    quantified = _ratio_score(_quantified_count(resume_text), excellent=6)
    action_verbs = _ratio_score(_action_verb_count(resume_text), excellent=12)
    readability = _readability_score(resume_text)
    completeness = _weighted_score(
        [(0.45, sections), (0.25, _length_score(word_count)), (0.30, readability)]
    )

    content_relevance = _weighted_score(
        [(0.45, keyword_richness), (0.35, sections), (0.20, _length_score(word_count))]
    )
    clarity_structure = _weighted_score(
        [(0.45, readability), (0.35, sections), (0.20, bullets)]
    )
    achievements = _weighted_score([(0.65, quantified), (0.35, action_verbs)])
    skills_depth = _weighted_score([(0.70, keyword_richness), (0.30, action_verbs)])

    return _weighted_score(
        [
            (0.25, content_relevance),
            (0.20, clarity_structure),
            (0.20, achievements),
            (0.20, skills_depth),
            (0.15, completeness),
        ]
    )


def _ats_score(resume_text: str) -> int:
    word_count = len(_tokens(resume_text))
    sections = _section_score(resume_text)
    keywords = _keyword_richness_score(resume_text)
    formatting = _weighted_score(
        [
            (0.45, _readability_score(resume_text)),
            (0.30, _ratio_score(_bullet_count(resume_text), excellent=8)),
            (0.25, _length_score(word_count)),
        ]
    )
    completeness = _weighted_score(
        [
            (0.60, sections),
            (
                0.25,
                100 if re.search(r"(?i)\b(?:20\d{2}|19\d{2})\b", resume_text) else 40,
            ),
            (
                0.15,
                100
                if re.search(r"(?i)\b(?:email|@|phone|linkedin|github)\b", resume_text)
                else 50,
            ),
        ]
    )
    role_terms = _weighted_score(
        [
            (0.60, keywords),
            (0.25, _ratio_score(_action_verb_count(resume_text), excellent=10)),
            (0.15, _ratio_score(_quantified_count(resume_text), excellent=4)),
        ]
    )

    return _weighted_score(
        [
            (0.25, sections),
            (0.25, keywords),
            (0.20, formatting),
            (0.15, completeness),
            (0.15, role_terms),
        ]
    )


def _sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?;])\s+|\n+", text)
        if part.strip()
    ]


def _candidate_job_terms(text: str) -> set[str]:
    terms = set(_extract_known_terms(text))
    capitalized_term_pattern = (
        r"\b[A-Z][A-Za-z0-9.+#-]{1,}"
        r"(?:\s+[A-Z][A-Za-z0-9.+#-]{1,}){0,2}\b"
    )
    for match in re.findall(capitalized_term_pattern, text):
        cleaned = match.casefold().strip()
        if cleaned not in STOPWORDS and len(cleaned) > 2:
            terms.add(cleaned)

    tokens = [
        token for token in _tokens(text) if len(token) >= 3 and token not in STOPWORDS
    ]
    for index, token in enumerate(tokens):
        if token in {"experience", "knowledge", "proficiency", "familiarity"}:
            for nearby in tokens[index + 1 : index + 5]:
                if len(nearby) >= 3 and nearby not in STOPWORDS:
                    terms.add(nearby)
    return terms


def _split_job_terms(job_description: str) -> tuple[set[str], set[str], set[str]]:
    required_terms = set()
    preferred_terms = set()
    all_terms = _candidate_job_terms(job_description)

    for sentence in _sentences(job_description):
        sentence_terms = _candidate_job_terms(sentence)
        lowered = sentence.casefold()
        if any(marker in lowered for marker in REQUIRED_MARKERS):
            required_terms.update(sentence_terms)
        elif any(marker in lowered for marker in PREFERRED_MARKERS):
            preferred_terms.update(sentence_terms)

    preferred_terms -= required_terms
    if not required_terms:
        required_terms = all_terms - preferred_terms
    return required_terms, preferred_terms, all_terms


def _term_matches_resume(term: str, resume_text: str, resume_tokens: set[str]) -> bool:
    lowered_resume = resume_text.casefold()
    if term in lowered_resume:
        return True
    term_tokens = [token for token in _tokens(term) if token not in STOPWORDS]
    return bool(term_tokens) and all(token in resume_tokens for token in term_tokens)


def _coverage_score(terms: set[str], resume_text: str, resume_tokens: set[str]) -> float:
    if not terms:
        return 100
    matched = sum(
        1 for term in terms if _term_matches_resume(term, resume_text, resume_tokens)
    )
    return (matched / len(terms)) * 100


def _job_match_score(resume_text: str, job_description: str) -> int:
    if not job_description.strip():
        return 0

    resume_tokens = _token_set(resume_text)
    required_terms, preferred_terms, all_terms = _split_job_terms(job_description)
    required = _coverage_score(required_terms, resume_text, resume_tokens)
    preferred = _coverage_score(preferred_terms, resume_text, resume_tokens)
    keyword_coverage = _coverage_score(all_terms, resume_text, resume_tokens)
    experience_overlap = _weighted_score(
        [
            (0.60, keyword_coverage),
            (0.25, _ratio_score(_quantified_count(resume_text), excellent=4)),
            (0.15, _ratio_score(_action_verb_count(resume_text), excellent=10)),
        ]
    )
    missing_required = 100 - required

    return _weighted_score(
        [
            (0.45, required),
            (0.20, preferred),
            (0.20, experience_overlap),
            (0.10, keyword_coverage),
            (0.05, 100 - missing_required),
        ]
    )


def _deterministic_scores(resume_text: str, job_description: str = "") -> dict:
    return {
        "score": _resume_quality_score(resume_text),
        "ats_score": _ats_score(resume_text),
        "job_match_score": _job_match_score(resume_text, job_description),
    }


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


def _validate_and_normalize(
    result: dict,
    has_job_description: bool,
    resume_text: str = "",
    job_description: str = "",
) -> dict:
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

    deterministic_scores = _deterministic_scores(resume_text, job_description)

    normalized = {
        "score": deterministic_scores["score"],
        "ats_score": deterministic_scores["ats_score"],
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
        "job_match_score": deterministic_scores["job_match_score"],
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
            config=types.GenerateContentConfig(
                candidate_count=1,
                response_mime_type="application/json",
                seed=0,
                temperature=0,
                top_k=1,
                top_p=1,
            ),
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
    return _validate_and_normalize(
        result,
        bool(clean_job_description),
        resume_text=resume_text,
        job_description=clean_job_description,
    )
