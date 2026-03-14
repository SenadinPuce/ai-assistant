from functools import lru_cache


@lru_cache(maxsize=1)
def _get_detector():
    from lingua import LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_all_languages().build()


def detect_language(text: str) -> str:
    """Return the full language name of the given text (e.g. 'English', 'Bosnian').

    Falls back to 'English' if detection fails or language is unrecognised.
    """
    try:
        language = _get_detector().detect_language_of(text)
        if language is None:
            return "English"
        return language.name.capitalize()
    except Exception:
        return "English"
