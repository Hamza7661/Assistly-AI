"""Text utilities for voice/TTS: strip emojis and symbols that can break playback."""
import re


# Match common emoji/symbol ranges (avoid breaking TTS or encoding)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002300-\U000023FF"  # misc technical
    "\U00002B50"
    "\U0000203C"
    "\U00002049"
    "\U000025AA-\U000025AB"
    "\U000025B6"
    "\U000025C0"
    "\U000025FB-\U000025FE"
    "\U00002614-\U00002615"
    "\U00002648-\U00002653"
    "\U0000267F"
    "\U00002693"
    "\U000026A1"
    "\U000026AA-\U000026AB"
    "\U000026BD-\U000026BE"
    "\U000026C4-\U000026C5"
    "\U000026CE"
    "\U000026D4"
    "\U000026EA"
    "\U000026F2-\U000026F3"
    "\U000026F5"
    "\U000026FA"
    "\U000026FD"
    "\U00002702"
    "\U00002705"
    "\U00002708-\U0000270D"
    "\U0000270F"
    "\U00002712-\U00002714"
    "\U00002716-\U00002717"
    "\U0000271D"
    "\U00002721"
    "\U00002728"
    "\U00002733-\U00002734"
    "\U00002744"
    "\U00002747"
    "\U0000274C"
    "\U0000274E"
    "\U00002753-\U00002755"
    "\U00002757"
    "\U00002763-\U00002764"
    "\U00002795-\U00002797"
    "\U000027A1"
    "\U000027B0"
    "\U000027BF"
    "\U00002934-\U00002935"
    "\U00003030"
    "\U0000303D"
    "\U00003297"
    "\U00003299"
    "\U0001F004"
    "\U0001F0CF"
    "\U0001F170-\U0001F171"
    "\U0001F17E-\U0001F17F"
    "\U0001F18E"
    "\U0001F191-\U0001F19A"
    "\U0001F201-\U0001F202"
    "\U0001F21A"
    "\U0001F22F"
    "\U0001F232-\U0001F23A"
    "\U0001F250-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def strip_emojis_for_voice(text: str) -> str:
    """
    Remove emojis and problematic symbols from text for voice/TTS.
    Use for greeting, closing message, and any script read aloud to avoid breaks.
    """
    if not text or not isinstance(text, str):
        return text
    cleaned = _EMOJI_PATTERN.sub("", text)
    # Collapse multiple spaces and strip
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
