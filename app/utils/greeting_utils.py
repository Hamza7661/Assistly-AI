"""Utility functions for processing greeting messages with placeholders."""

import re
from typing import Dict, Any


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


def get_greeting_with_fallback(context: Dict[str, Any]) -> str:
    """
    Get greeting from integration settings with placeholder replacement.
    Falls back to default greeting if none is set.
    
    Args:
        context: Context dict containing integration settings
        
    Returns:
        Processed greeting message
    """
    integration = context.get("integration", {}) or {}
    greeting = integration.get("greeting", "").strip()
    
    # If no custom greeting, use Professional template as default
    if not greeting:
        assistant_name = integration.get("assistantName", "").strip() or "Assistant"
        company_name = integration.get("companyName", "").strip()
        
        # Use Professional template format
        if company_name:
            greeting = f"Hi this is {assistant_name} your virtual ai assistant from {company_name}. How can I help you today?"
        elif assistant_name:
            greeting = f"Hi this is {assistant_name} your virtual ai assistant. How can I help you today?"
        else:
            # Fallback to Professional template with placeholders
            greeting = "Hi this is {assistantName} your virtual ai assistant from {companyName}. How can I help you today?"
    
    # Process placeholders
    return process_greeting(greeting, integration)

