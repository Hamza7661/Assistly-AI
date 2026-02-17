"""On-the-fly translation via OpenAI. Use for lead types and service names in any language."""
from typing import List, Optional, Any
import json
import logging
from .cache_utils import get_cached_translation, cache_translation

logger = logging.getLogger("assistly.translation_utils")


async def translate_batch(
    client: Optional[Any],
    model: Optional[str],
    texts: List[str],
    target_language_name: str,
    app_id: Optional[str] = None,
) -> List[str]:
    """
    Translate a list of strings to the target language. Uses one API call.
    Returns originals if client/model missing, target is English, or translation fails.
    Now with caching to reduce API calls and improve response time.
    """
    if not client or not model or not texts:
        return list(texts) if texts else []
    if not target_language_name or str(target_language_name).strip().lower() in ("en", "english"):
        return list(texts)

    cleaned = [str(t).strip() for t in texts if t is not None]
    if not cleaned:
        return list(texts)

    # Check cache first
    cached = get_cached_translation(cleaned, target_language_name, app_id)
    if cached:
        logger.info(f"Using cached translation for {len(cleaned)} texts to {target_language_name}")
        return cached

    prompt = f"""Translate each of the following to {target_language_name}. Keep the same order and meaning. Return a JSON array of strings only, no other text. Example: ["translation1", "translation2"]

List to translate:
{json.dumps(cleaned, ensure_ascii=False)}"""

    try:
        logger.info(f"Calling OpenAI to translate {len(cleaned)} texts to {target_language_name}")
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output only valid JSON arrays of strings. No markdown, no explanation."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.1,
        )
        content = (response.choices[0].message.content or "").strip()
        # Strip markdown code block if present
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        out = json.loads(content)
        if isinstance(out, list) and len(out) == len(cleaned):
            translated = [str(x) for x in out]
            # Cache successful translation for 1 hour
            cache_translation(cleaned, target_language_name, translated, app_id, ttl_seconds=3600)
            logger.info(f"Successfully translated and cached {len(cleaned)} texts")
            return translated
        # Length mismatch: return originals
        logger.warning("Translation returned wrong length, using originals")
        return list(texts)
    except Exception as e:
        logger.warning("Translation failed (%s), using originals", e)
        return list(texts)


async def translate_to_english(
    client: Optional[Any],
    model: Optional[str],
    text: str,
    app_id: Optional[str] = None,
) -> str:
    """Translate user message to English for matching against lead type text/value. Returns original if fail or no client."""
    if not client or not model or not (text and str(text).strip()):
        return str(text or "")
    try:
        result = await translate_batch(client, model, [str(text).strip()], "English", app_id)
        return result[0] if result else str(text)
    except Exception:
        return str(text)
