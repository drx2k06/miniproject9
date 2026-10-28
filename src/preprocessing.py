"""Text preprocessing and cleaning utilities for NewsLens AI.

Preserves domain entities, acronyms, and technical terms while
normalizing whitespace, stripping HTML tags/entities, and handling invalid input.

Author: Daniel Richard Xavier
"""

import re
import html
from typing import Optional, Union, List
import pandas as pd


# Regex patterns compiled for performance
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def clean_text(text: Union[str, float, None]) -> str:
    """Clean and normalize a single news text entry.
    
    Processing steps:
      1. Null and type safety check.
      2. HTML entity unescaping (e.g. &amp; -> &, &quot; -> ").
      3. HTML tag removal.
      4. URL normalization / stripping.
      5. Control characters removal (e.g. \b, \x00).
      6. Whitespace normalization (collapsing tabs, newlines, multiple spaces).
      7. Stripping leading/trailing spaces.
    
    Args:
        text: Raw text string or object.

    Returns:
        Cleaned, normalized text string.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Unescape HTML entities
    text = html.unescape(text)

    # 2. Strip HTML tags if present
    text = HTML_TAG_PATTERN.sub(" ", text)

    # 3. Strip URLs if present
    text = URL_PATTERN.sub(" ", text)

    # 4. Remove control characters (including literal escape glitches like \b)
    text = CONTROL_CHAR_PATTERN.sub(" ", text)
    text = text.replace(r"\b", " ").replace(r"\n", " ").replace(r"\t", " ").replace(r"\r", " ")

    # 5. Normalize whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def combine_headline_and_body(
    headline: Optional[str] = None,
    body: Optional[str] = None
) -> str:
    """Combine headline and body/description into a unified text string.
    
    Args:
        headline: News title/headline.
        body: News body or description text.

    Returns:
        Combined cleaned string.
    """
    clean_h = clean_text(headline)
    clean_b = clean_text(body)

    if clean_h and clean_b:
        # Standard format: Headline followed by description
        return f"{clean_h} - {clean_b}"
    elif clean_h:
        return clean_h
    elif clean_b:
        return clean_b
    else:
        return ""


def preprocess_corpus(
    texts: Union[pd.Series, List[str]],
    min_char_length: int = 5
) -> List[str]:
    """Clean a collection of news texts.
    
    Args:
        texts: Series or list of raw strings.
        min_char_length: Minimum character length for valid text.

    Returns:
        List of cleaned strings.
    """
    cleaned = [clean_text(t) for t in texts]
    return cleaned


Tuple_Validation = tuple[bool, str, Optional[str]]


def validate_input_text(text: str, max_chars: int = 15000) -> Tuple_Validation:
    """Validate user-entered article text for inference.
    
    Args:
        text: Input string.
        max_chars: Maximum permissible character length.

    Returns:
        Tuple of (is_valid: bool, cleaned_text: str, error_message: Optional[str]).
    """
    if not text or not isinstance(text, str):
        return False, "", "Article text is empty or missing. Please enter a headline or article body."

    cleaned = clean_text(text)
    if not cleaned or len(cleaned.strip()) == 0:
        return False, "", "Article text contains only whitespace or invalid characters. Please enter valid text."

    if len(cleaned) < 5:
        return False, cleaned, "Article text is too short (less than 5 characters). Please provide a fuller headline or snippet."

    if len(cleaned) > max_chars:
        return False, cleaned[:max_chars], f"Article text exceeds maximum length of {max_chars} characters. It has been truncated."

    return True, cleaned, None
