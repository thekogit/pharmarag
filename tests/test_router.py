import pytest
from src.router import RegulatoryRouter

def test_general_section():
    router = RegulatoryRouter()
    text = "Some random text before any headers"
    result = router.parse(text)
    assert result == {"GENERAL": "Some random text before any headers"}
