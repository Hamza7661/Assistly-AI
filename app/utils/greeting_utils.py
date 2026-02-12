"""Utility functions for processing greeting messages with placeholders."""

import re
from typing import Dict, Any, Optional

# Default greeting templates per language (with {assistantName}, {companyName}).
# Used when lang_code is set so the first-message language gets a localized greeting.
DEFAULT_GREETING_BY_LANG: Dict[str, str] = {
    "en": "Hi! 👋 This is {assistantName} from {companyName}. What would you like to do today?",
    "ur": "آپ کا سلام! 👋 یہ {assistantName} ہے، {companyName} سے۔ آج آپ کیا کرنا چاہیں گے؟",
    "hi": "नमस्ते! 👋 यह {assistantName} है, {companyName} से। आज आप क्या करना चाहेंगे?",
    "ar": "مرحباً! 👋 أنا {assistantName} من {companyName}. ماذا تريد أن تفعل اليوم؟",
    "es": "¡Hola! 👋 Soy {assistantName} de {companyName}. ¿Qué te gustaría hacer hoy?",
    "fr": "Bonjour ! 👋 Je suis {assistantName} de {companyName}. Que souhaitez-vous faire aujourd'hui ?",
    "de": "Hallo! 👋 Ich bin {assistantName} von {companyName}. Was möchten Sie heute tun?",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! 👋 ਇਹ {assistantName} ਹੈ, {companyName} ਤੋਂ। ਅੱਜ ਤੁਸੀਂ ਕੀ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ?",
}


def process_greeting(greeting: str, integration: Dict[str, Any]) -> str:
    """
    Process greeting message by replacing placeholders with actual values.
    If company name is not provided, gracefully removes company name phrases.
    
    Supported placeholders:
    - {assistantName} - Replaced with assistant name from integration settings
    - {companyName} - Replaced with company name from integration settings
    
    Args:
        greeting: The greeting message template (may contain placeholders)
        integration: Integration settings dict containing assistantName and companyName
        
    Returns:
        Processed greeting with placeholders replaced and company phrases removed if needed
    """
    if not greeting:
        return ""
    
    # Get values from integration settings
    assistant_name = integration.get("assistantName", "").strip() or "Assistant"
    company_name = integration.get("companyName", "").strip() or ""
    
    # Replace assistant name placeholder
    processed = greeting.replace("{assistantName}", assistant_name)
    
    # If company name is provided, replace it
    if company_name:
        processed = processed.replace("{companyName}", company_name)
    else:
        # Remove company name phrases gracefully when company name is not provided
        # Patterns to remove: "from {companyName}", "at {companyName}", etc.
        patterns_to_remove = [
            r'\s+from\s+\{companyName\}',  # " from {companyName}"
            r'\s+at\s+\{companyName\}',    # " at {companyName}"
            r'\s+of\s+\{companyName\}',    # " of {companyName}"
            r'\s+with\s+\{companyName\}',  # " with {companyName}"
            r'\{companyName\}\s+',          # "{companyName} " (at start)
            r'\s+\{companyName\}',         # " {companyName}" (anywhere)
        ]
        
        # Remove company name placeholder and associated phrases
        for pattern in patterns_to_remove:
            processed = re.sub(pattern, ' ', processed, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and trailing/leading spaces
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        # Clean up punctuation issues (e.g., "assistant ." -> "assistant.")
        processed = re.sub(r'\s+([.,!?])', r'\1', processed)
        processed = re.sub(r'([.,!?])\s+([.,!?])', r'\1\2', processed)
    
    return processed


def get_greeting_with_fallback(context: Dict[str, Any], lang_code: Optional[str] = None) -> str:
    """
    Get greeting from integration settings (database per app) with placeholder replacement.
    Always prefers the greeting stored in the database for the app's integration.
    When no integration greeting is set, falls back to a default; if lang_code is
    provided (e.g. from first message language detection), uses a localized default template.
    
    Args:
        context: Context dict containing integration settings (from backend/database)
        lang_code: Optional ISO 639-1 language code (e.g. 'ur', 'hi') for localized default when no DB greeting
        
    Returns:
        Processed greeting message
    """
    integration = context.get("integration", {}) or {}
    # Prefer greeting from database (per app integration) – never overwrite when set
    greeting = (integration.get("greeting") or "").strip()
    lang = (lang_code or "").lower().strip() if lang_code else None

    if greeting:
        # Use the app's greeting from the database; only replace placeholders
        pass
    elif lang and lang in DEFAULT_GREETING_BY_LANG:
        # No DB greeting: use localized default template for detected language
        greeting = DEFAULT_GREETING_BY_LANG[lang]
    else:
        # No DB greeting and no lang (or unsupported lang): use English default
        assistant_name = integration.get("assistantName", "").strip() or "Assistant"
        company_name = integration.get("companyName", "").strip()
        if company_name:
            greeting = f"Hi this is {assistant_name} your virtual ai assistant from {company_name}. How can I help you today?"
        elif assistant_name:
            greeting = f"Hi this is {assistant_name} your virtual ai assistant. How can I help you today?"
        else:
            greeting = "Hi this is {assistantName} your virtual ai assistant from {companyName}. How can I help you today?"

    # Process placeholders (assistantName, companyName)
    return process_greeting(greeting, integration)

