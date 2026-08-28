"""
resume_parser.py

Handles extracting plain text from an uploaded PDF resume.
"""

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ResumeParseError(Exception):
    """Raised when a resume PDF cannot be read or contains no usable text."""


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract text from a Streamlit UploadedFile (PDF).

    Args:
        uploaded_file: The file object returned by st.file_uploader.

    Returns:
        Extracted, cleaned text from the PDF.

    Raises:
        ResumeParseError: If the file isn't a valid/readable PDF, or no
            extractable text is found (e.g. a scanned/image-only resume).
    """
    try:
        reader = PdfReader(uploaded_file)
    except PdfReadError as exc:
        raise ResumeParseError(
            "The uploaded file could not be processed. Please make sure it's a valid PDF."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface any parse failure as a friendly error
        raise ResumeParseError(
            "The uploaded file could not be processed. Please make sure it's a valid PDF."
        ) from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise ResumeParseError(
                "This PDF is password-protected. Please upload an unprotected file."
            ) from exc

    pages_text = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - skip unreadable pages, don't crash the app
            page_text = ""
        pages_text.append(page_text)

    full_text = "\n".join(pages_text).strip()

    if not full_text:
        raise ResumeParseError(
            "We couldn't extract readable text from this resume. "
            "It may be a scanned image — try uploading a text-based PDF instead."
        )

    return full_text
