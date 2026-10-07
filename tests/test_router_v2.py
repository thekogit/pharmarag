import pytest
from src.router import RegulatoryRouter

def test_unclassified_fallback():
    """Test that text before any header is assigned to UNCLASSIFIED."""
    router = RegulatoryRouter()
    text = "Introductory text before any section.\nMore intro text."
    result = router.parse(text)
    assert "UNCLASSIFIED" in result
    assert result["UNCLASSIFIED"] == "Introductory text before any section.\nMore intro text."
    assert "GENERAL" not in result

def test_mixed_unclassified_and_headers():
    """Test that text before header is UNCLASSIFIED, then follows standard headers."""
    router = RegulatoryRouter()
    text = """This is unclassified text.
1 INDICATIONS AND USAGE
Some indications.
2 DOSAGE AND ADMINISTRATION
Some dosage.
"""
    result = router.parse(text)
    assert result["UNCLASSIFIED"] == "This is unclassified text."
    assert result["INDICATIONS"] == "Some indications."
    assert result["DOSAGE"] == "Some dosage."

def test_no_empty_sections():
    """Test that empty sections (including UNCLASSIFIED if empty) are removed."""
    router = RegulatoryRouter()
    text = """1 INDICATIONS AND USAGE
Some indications.
"""
    result = router.parse(text)
    assert "UNCLASSIFIED" not in result
    assert result["INDICATIONS"] == "Some indications."

def test_standard_headers_preserved():
    """Ensure existing header logic still works."""
    router = RegulatoryRouter()
    text = """1 INDICATIONS AND USAGE
ind
2 DOSAGE AND ADMINISTRATION
dos
4 CONTRAINDICATIONS
con
5 WARNINGS AND PRECAUTIONS
war
6 ADVERSE REACTIONS
adv
14 CLINICAL STUDIES
cli
"""
    result = router.parse(text)
    expected = {
        "INDICATIONS": "ind",
        "DOSAGE": "dos",
        "CONTRAINDICATIONS": "con",
        "WARNINGS": "war",
        "ADVERSE_REACTIONS": "adv",
        "CLINICAL_STUDIES": "cli"
    }
    assert result == expected
