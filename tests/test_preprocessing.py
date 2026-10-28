"""Unit tests for text preprocessing and validation routines.

Author: Daniel Richard Xavier
"""

import pytest
from src.preprocessing import (
    clean_text,
    combine_headline_and_body,
    validate_input_text,
    preprocess_corpus
)


def test_clean_text_normalizes_whitespace():
    raw = "Breaking   news:\tStock  market \n surges!   "
    cleaned = clean_text(raw)
    assert cleaned == "Breaking news: Stock market surges!"


def test_clean_text_unescapes_html_entities():
    raw = "Apple &amp; Microsoft report &quot;record&quot; Q3 earnings &lt;tech&gt;"
    cleaned = clean_text(raw)
    assert "&amp;" not in cleaned
    assert "&quot;" not in cleaned
    assert 'Apple & Microsoft report "record" Q3 earnings' in cleaned


def test_clean_text_strips_html_tags():
    raw = "<p>Olympic <b>gold</b> medalist wins again.</p>"
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "Olympic gold medalist wins again." in cleaned


def test_clean_text_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""


def test_clean_text_removes_urls():
    raw = "Visit https://news.example.com/article?id=123 for full coverage."
    cleaned = clean_text(raw)
    assert "https://" not in cleaned
    assert "news.example.com" not in cleaned


def test_clean_text_removes_escape_glitches():
    raw = r"Oil prices rise\brapidly in Asian trading"
    cleaned = clean_text(raw)
    assert r"\b" not in cleaned
    assert "Oil prices rise rapidly in Asian trading" == cleaned


def test_combine_headline_and_body():
    h = "NASA Launches Rover"
    b = "The spacecraft successfully began its 6-month journey to Mars."
    combined = combine_headline_and_body(h, b)
    assert combined == "NASA Launches Rover - The spacecraft successfully began its 6-month journey to Mars."


def test_combine_headline_only():
    assert combine_headline_and_body("Champions League Final", None) == "Champions League Final"


def test_combine_body_only():
    assert combine_headline_and_body(None, "Economic indicators point to growth.") == "Economic indicators point to growth."


def test_combine_both_empty():
    assert combine_headline_and_body(None, "") == ""


def test_validate_input_text_valid():
    is_valid, text, err = validate_input_text("Federal Reserve maintains benchmark interest rate.")
    assert is_valid is True
    assert err is None
    assert "Federal Reserve" in text


def test_validate_input_text_too_short():
    is_valid, text, err = validate_input_text("Hi")
    assert is_valid is False
    assert "too short" in err.lower()


def test_validate_input_text_whitespace_only():
    is_valid, text, err = validate_input_text("    \n\t  ")
    assert is_valid is False
    assert "empty or missing" in err.lower() or "only whitespace" in err.lower()


def test_validate_input_text_truncates_excess_length():
    huge_text = "word " * 4000  # 20,000 characters
    is_valid, text, err = validate_input_text(huge_text, max_chars=1000)
    assert is_valid is False
    assert len(text) <= 1000
    assert "exceeds maximum length" in err.lower()


def test_preprocess_corpus():
    corpus = ["First &amp; article", "<p>Second article</p>"]
    cleaned = preprocess_corpus(corpus)
    assert len(cleaned) == 2
    assert "&amp;" not in cleaned[0]
    assert "<p>" not in cleaned[1]
