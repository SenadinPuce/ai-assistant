from unittest.mock import MagicMock, patch

from rag.language import detect_language


@patch("rag.language._get_detector")
def test_detect_language_capitalizes_language_name(mock_get_detector: MagicMock) -> None:
    """A detected language's enum name is capitalized to match generation_chain's {language} slot."""
    detected = MagicMock()
    detected.name = "ENGLISH"
    detector = MagicMock()
    detector.detect_language_of.return_value = detected
    mock_get_detector.return_value = detector

    assert detect_language("Hello there") == "English"


@patch("rag.language._get_detector")
def test_detect_language_falls_back_when_unrecognized(mock_get_detector: MagicMock) -> None:
    """No confidently detected language falls back to English."""
    detector = MagicMock()
    detector.detect_language_of.return_value = None
    mock_get_detector.return_value = detector

    assert detect_language("???") == "English"


@patch("rag.language._get_detector")
def test_detect_language_falls_back_on_exception(mock_get_detector: MagicMock) -> None:
    """Any detector failure falls back to English rather than raising."""
    mock_get_detector.side_effect = RuntimeError("model not loaded")

    assert detect_language("anything") == "English"
