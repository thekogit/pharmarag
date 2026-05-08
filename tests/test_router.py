import pytest
from src.router import RegulatoryRouter

def test_unclassified_section():
    router = RegulatoryRouter()
    text = "Some random text before any headers"
    result = router.parse(text)
    assert result == {"UNCLASSIFIED": "Some random text before any headers"}

def test_multiple_sections():
    router = RegulatoryRouter()
    text = """1 INDICATIONS AND USAGE
This drug is for testing.
2 DOSAGE AND ADMINISTRATION
Take one test daily.
6 ADVERSE REACTIONS
May cause excessive passing of tests."""
    result = router.parse(text)
    assert "INDICATIONS" in result
    assert "DOSAGE" in result
    assert "ADVERSE_REACTIONS" in result
    assert result["INDICATIONS"] == "This drug is for testing."
    assert result["DOSAGE"] == "Take one test daily."
    assert result["ADVERSE_REACTIONS"] == "May cause excessive passing of tests."

def test_all_headers():
    router = RegulatoryRouter()
    text = """
1 INDICATIONS AND USAGE
ind text
2 DOSAGE AND ADMINISTRATION
dos text
4 CONTRAINDICATIONS
con text
5 WARNINGS AND PRECAUTIONS
war text
6 ADVERSE REACTIONS
adv text
14 CLINICAL STUDIES
cli text
"""
    result = router.parse(text)
    assert result["INDICATIONS"] == "ind text"
    assert result["DOSAGE"] == "dos text"
    assert result["CONTRAINDICATIONS"] == "con text"
    assert result["WARNINGS"] == "war text"
    assert result["ADVERSE_REACTIONS"] == "adv text"
    assert result["CLINICAL_STUDIES"] == "cli text"
