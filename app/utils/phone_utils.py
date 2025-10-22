"""
Phone number utilities for formatting and country detection
"""
import logging

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


def format_phone_number(phone: str) -> str:
    """Format phone number with proper country code"""
    # Remove any non-digit characters except +
    cleaned = ''.join(c for c in phone if c.isdigit() or c == '+')
    
    # If it already has a country code, return as is
    if cleaned.startswith('+'):
        return cleaned
    
    # Detect country and add appropriate country code
    country = detect_country_from_phone(cleaned)
    
    match country:
        case 'US':
            if len(cleaned) == 10:
                cleaned = '+1' + cleaned
            elif len(cleaned) == 11 and cleaned.startswith('1'):
                cleaned = '+' + cleaned
            else:
                cleaned = '+1' + cleaned
        case 'UK':
            cleaned = '+44' + cleaned
        case 'PK':
            cleaned = '+92' + cleaned
        case 'AU':
            cleaned = '+61' + cleaned
        case 'CA':
            cleaned = '+1' + cleaned  # Canada uses same format as US
        case _:
            # Default to US
            cleaned = '+1' + cleaned
    
    logger.info(f"Formatted phone {phone} -> {cleaned} (detected country: {country})")
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
