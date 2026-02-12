"""Language detection and prompt helpers for response language."""
from typing import Optional

# Language code -> full name for LLM prompts (e.g. "Respond only in Spanish")
_LANGUAGE_NAMES: dict[str, str] = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}

# Minimum length to run detection; shorter text defaults to English
_MIN_TEXT_LENGTH = 3

# Digits-only or mostly digits -> skip detection
_DIGITS_ONLY = set("0123456789 \t\n")


def detect_language(text: str) -> str:
    """
    Detect language of the given text. Returns ISO 639-1 code (e.g. 'en', 'es', 'hi').
    On very short or ambiguous text, returns 'en'.
    """
    if not text or not isinstance(text, str):
        return "en"
    cleaned = text.strip()
    if len(cleaned) < _MIN_TEXT_LENGTH:
        return "en"
    if set(cleaned.lower().replace(" ", "")) <= _DIGITS_ONLY:
        return "en"
    try:
        import langdetect
        # Use only current message for detection
        detected = langdetect.detect(cleaned)
        if detected and isinstance(detected, str):
            return detected.lower()
    except Exception:
        pass
    return "en"


def get_language_name_for_prompt(code: str) -> Optional[str]:
    """
    Map language code to full name for prompts (e.g. 'es' -> 'Spanish').
    Returns None for English so callers can skip the instruction.
    """
    if not code:
        return None
    key = (code or "").lower().strip()
    if key in ("en", "english"):
        return None
    # Unknown codes default to None so we keep English behavior
    return _LANGUAGE_NAMES.get(key)


def get_language_name(code: str) -> str:
    """
    Map language code to full name (e.g. 'ur' -> 'Urdu', 'en' -> 'English').
    Used for translation targets; returns 'English' for en.
    """
    if not code:
        return "English"
    key = (code or "").lower().strip()
    if key in ("en", "english"):
        return "English"
    return _LANGUAGE_NAMES.get(key) or key
