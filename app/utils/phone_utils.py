"""
Phone number utilities for formatting and country detection
"""
import logging
from typing import Optional

logger = logging.getLogger("assistly.phone_utils")


def detect_country_from_phone(phone: str) -> str:
    """Detect country code using Google's libphonenumber"""
    try:
        from phonenumbers import parse, is_valid_number_for_region, NumberParseException
        
        # Target regions to test
        regions = ["US", "GB", "CA", "AU", "PK"]
        
        # Try parsing against each region
        for region in regions:
            try:
                parsed_number = parse(phone, region)
                if is_valid_number_for_region(parsed_number, region):
                    logger.info(f"Detected country {region} for phone: {phone}")
                    return region
            except NumberParseException:
                continue
        
        # If no region matches, default to US
        logger.warning(f"No valid region found for phone: {phone}, defaulting to US")
        return "US"
        
    except ImportError:
        logger.warning("libphonenumber not available, using fallback pattern matching")
        # Fallback to simple pattern matching if libphonenumber not available
        digits_only = ''.join(c for c in phone if c.isdigit())
        
        if digits_only.startswith('1') and len(digits_only) == 11:
            return 'US'
        elif digits_only.startswith('44'):
            return 'UK'
        elif digits_only.startswith('92'):
            return 'PK'
        elif digits_only.startswith('61'):
            return 'AU'
        elif len(digits_only) == 10:
            return 'US'
        else:
            return 'US'


async def format_phone_number_with_gpt(phone: str, client, model: str) -> str:
    """Format phone number using GPT to detect and add country code"""
    # Clean phone number
    cleaned = ''.join(c for c in phone if c.isdigit() or c == '+')
    
    # If it already has a country code, return as is
    if cleaned.startswith('+'):
        return cleaned
    
    system_prompt = """You are a phone number formatter. Given a phone number without a country code, determine the correct country code and return the phone number in E.164 format (e.g., +1234567890).

Return ONLY the formatted phone number with country code in E.164 format. Do not include any explanation or other text. Just the phone number starting with +.

Examples:
- Input: "4155551234" → Output: "+14155551234" (US)
- Input: "07911123456" → Output: "+447911123456" (UK)
- Input: "03001234567" → Output: "+923001234567" (Pakistan)
- Input: "0412345678" → Output: "+61412345678" (Australia)
- Input: "4165551234" → Output: "+14165551234" (Canada/US)"""

    user_prompt = f"Format this phone number: {cleaned}"

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=20
    )
    
    formatted = response.choices[0].message.content.strip()
    
    # Validate and return the formatted phone number
    if formatted.startswith('+') and all(c.isdigit() or c == '+' for c in formatted[1:]):
        logger.info(f"GPT formatted phone {phone} -> {formatted}")
        return formatted
    else:
        # If GPT returns invalid format, raise an error
        raise ValueError(f"GPT returned invalid phone format: {formatted}")


def format_phone_number(phone: str) -> str:
    """Format phone number - simple passthrough if already has country code, otherwise return as-is.
    
    Note: This function is kept for backward compatibility. Actual formatting is done via GPT.
    """
    # Remove any non-digit characters except +
    cleaned = ''.join(c for c in phone if c.isdigit() or c == '+')
    
    # If it already has a country code, return as is
    if cleaned.startswith('+'):
        return cleaned
    
    # Otherwise return cleaned number (GPT will add country code)
    return cleaned


def is_valid_phone_number(phone: str) -> bool:
    """Check if phone number is valid using libphonenumber"""
    try:
        from phonenumbers import parse, is_valid_number, NumberParseException
        
        formatted_phone = format_phone_number(phone)
        parsed_number = parse(formatted_phone, None)
        return is_valid_number(parsed_number)
        
    except (ImportError, NumberParseException):
        # Fallback validation - basic length check
        digits_only = ''.join(c for c in phone if c.isdigit())
        return 7 <= len(digits_only) <= 15
