"""Structured data extraction for conversation data"""
import re
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger("assistly.data_extractors")


class DataExtractor:
    """Extract structured data from user messages"""
    
    @staticmethod
    def extract_email(text: str) -> Optional[str]:
        """Extract email address from text"""
        if not text:
            return None
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        if match:
            email = match.group().lower().strip()
            logger.info(f"Extracted email: {email}")
            return email
        return None
    
    @staticmethod
    def extract_phone(text: str) -> Optional[str]:
        """Extract phone number from text"""
        if not text:
            return None
        
        # Phone patterns
        phone_patterns = [
            r'\b\d{10,15}\b',  # 10-15 digits
            r'\+\d{10,15}\b',  # + followed by 10-15 digits
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US format
            r'\b\d{4}[-.\s]?\d{3}[-.\s]?\d{3}\b',  # Some international formats
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                phone = re.sub(r'[-.\s]', '', match.group())
                logger.info(f"Extracted phone: {phone}")
                return phone
        return None
    
    @staticmethod
    def extract_otp_code(text: str) -> Optional[str]:
        """Extract 6-digit OTP code from text"""
        if not text:
            return None
        
        # Look for 6-digit code
        otp_pattern = r'\b\d{6}\b'
        match = re.search(otp_pattern, text)
        if match:
            code = match.group()
            logger.info(f"Extracted OTP code: {code}")
            return code
        return None
    
    @staticmethod
    def extract_name(text: str, lead_types: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Extract name from text (heuristic) - skips lead type phrases"""
        if not text:
            return None
        
        text = text.strip()
        
        # Skip if it looks like an email or phone
        if '@' in text or re.search(r'\d{10,}', text):
            return None
        
        # Skip if it matches a lead type (CRITICAL - prevents lead type text from being extracted as name)
        if lead_types:
            if DataExtractor.match_lead_type(text, lead_types):
                return None
        
        # Skip lead type-like phrases (common patterns)
        lead_type_phrases = [
            r'^(i would like|i\'d like|i want)',
            r'(call back|appointment|further information|more info)',
            r'^(arrange|schedule|book)',
        ]
        for pattern in lead_type_phrases:
            if re.search(pattern, text, re.IGNORECASE):
                return None
        
        # Skip common non-name responses
        skip_patterns = [
            r'^(yes|no|ok|okay|sure|thanks|thank you)$',
            r'^(please|can you|could you)',
        ]
        for pattern in skip_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return None
        
        # If it's 2-50 characters and mostly letters/spaces, likely a name
        # But exclude if it's too long (likely not a name) or contains common lead type words
        if 2 <= len(text) <= 50 and re.match(r'^[A-Za-z\s\-\']+$', text):
            # Additional check: if it has more than 4 words, it's probably not a name
            if len(text.split()) > 4:
                return None
            
            name = ' '.join(word.capitalize() for word in text.split())
            logger.info(f"Extracted name: {name}")
            return name
        
        return None
    
    @staticmethod
    def match_lead_type(user_input: str, lead_types: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Match user input to a lead type - prioritizes exact matches"""
        if not user_input or not lead_types:
            return None
        
        user_input_lower = user_input.lower().strip()
        
        # Common words to ignore in keyword matching (too generic)
        common_words = {'i', 'would', 'like', 'to', 'a', 'an', 'the', 'my', 'me', 'for', 'with', 'is', 'are', 'am'}
        
        # First pass: exact matches (highest priority)
        for lt in lead_types:
            if not isinstance(lt, dict):
                continue
            
            # Check exact text match
            text = str(lt.get("text", "")).lower().strip()
            if text and text == user_input_lower:
                logger.info(f"Matched lead type by exact text: {lt.get('text')}")
                return lt
            
            # Check exact value match
            value = str(lt.get("value", "")).lower().strip()
            if value and value == user_input_lower:
                logger.info(f"Matched lead type by exact value: {lt.get('value')}")
                return lt
        
        # Second pass: substring matches (medium priority)
        for lt in lead_types:
            if not isinstance(lt, dict):
                continue
            
            # Check if text is contained in user input (or vice versa)
            text = str(lt.get("text", "")).lower().strip()
            value = str(lt.get("value", "")).lower().strip()
            
            # Normalize both for better matching (remove punctuation, extra spaces)
            import re
            text_normalized = re.sub(r'[^\w\s]', '', text)
            value_normalized = re.sub(r'[^\w\s]', '', value)
            user_input_normalized = re.sub(r'[^\w\s]', '', user_input_lower)
            
            # Check if key words from text/value are in user input
            if text_normalized:
                text_words = set(word for word in text_normalized.split() if len(word) > 2)
                user_words = set(word for word in user_input_normalized.split() if len(word) > 2)
                # If majority of meaningful words match, consider it a match
                if text_words and len(text_words.intersection(user_words)) >= min(2, len(text_words) * 0.6):
                    logger.info(f"Matched lead type by text words: {lt.get('text')}")
                    return lt
            
            if text and (text in user_input_lower or user_input_lower in text):
                logger.info(f"Matched lead type by text substring: {lt.get('text')}")
                return lt
            
            # Check value matching with word overlap
            if value_normalized:
                value_words = set(word for word in value_normalized.split() if len(word) > 2)
                user_words = set(word for word in user_input_normalized.split() if len(word) > 2)
                # If key words from value are in user input (e.g., "appointment" in "arrange an appointment")
                if value_words and value_words.intersection(user_words):
                    logger.info(f"Matched lead type by value words: {lt.get('value')}")
                    return lt
            
            if value and (value in user_input_lower or user_input_lower in value):
                logger.info(f"Matched lead type by value substring: {lt.get('value')}")
                return lt
        
        # Third pass: keyword matching (lowest priority, but more strict)
        best_match = None
        best_score = 0
        
        for lt in lead_types:
            if not isinstance(lt, dict):
                continue
            
            text = str(lt.get("text", "")).lower().strip()
            if not text:
                continue
            
            # Extract meaningful words (exclude common words)
            text_words = set(word for word in text.split() if word not in common_words and len(word) > 2)
            input_words = set(word for word in user_input_lower.split() if word not in common_words and len(word) > 2)
            
            if not text_words:
                continue
            
            # Calculate match score (meaningful words in common)
            common_meaningful = text_words.intersection(input_words)
            score = len(common_meaningful)
            
            # Require at least 2 meaningful words in common, and prefer longer matches
            if score >= 2 and score > best_score:
                best_match = lt
                best_score = score
        
        if best_match:
            logger.info(f"Matched lead type by keywords (score {best_score}): {best_match.get('text')}")
            return best_match
        
        return None
    
    @staticmethod
    def match_service(user_input: str, services: List[Any]) -> Optional[str]:
        """Match user input to a service - handles partial matches and single keywords"""
        if not user_input or not services:
            return None
        
        user_input_lower = user_input.lower().strip()
        # Remove punctuation for better matching
        import re
        user_input_normalized = re.sub(r'[^\w\s]', '', user_input_lower)
        
        # Common words to ignore
        common_words = {'i', 'would', 'like', 'to', 'a', 'an', 'the', 'my', 'me', 'for', 'with', 'is', 'are', 'am', 'well', 'can', 'you', 'tell', 'me', 'the', 'about', 'do', 'does', 'what', 'how', 'much', 'cost', 'price', 'pricing', 'information', 'info'}
        
        best_match = None
        best_score = 0
        
        for service in services:
            service_name = None
            
            if isinstance(service, dict):
                service_name = service.get("name") or service.get("title") or service.get("question", "")
            else:
                service_name = str(service)
            
            if not service_name:
                continue
            
            service_lower = service_name.lower()
            service_normalized = re.sub(r'[^\w\s]', '', service_lower)
            
            # Exact match (highest priority)
            if service_lower == user_input_lower or service_normalized == user_input_normalized:
                logger.info(f"Matched service (exact): {service_name}")
                return service_name
            
            # Contains match (high priority)
            if service_lower in user_input_lower or user_input_lower in service_lower:
                logger.info(f"Matched service (contains): {service_name}")
                return service_name
            
            # Extract meaningful words (exclude common words and short words)
            service_words = set(word for word in service_normalized.split() if word not in common_words and len(word) > 2)
            input_words = set(word for word in user_input_normalized.split() if word not in common_words and len(word) > 2)
            
            if not service_words:
                continue
            
            # Calculate overlap score
            overlap = service_words.intersection(input_words)
            score = len(overlap)
            
            # Match if:
            # 1. At least 2 words match (strong match)
            # 2. OR 1 key word matches AND it's a significant word (length >= 4) OR it's the only/main word in service name
            if score >= 2:
                if score > best_score:
                    best_match = service_name
                    best_score = score
            elif score == 1 and overlap:
                # Single word match - check if it's significant
                matched_word = list(overlap)[0]
                # If service name is 1-2 words and we matched one, or if matched word is >= 4 chars (significant)
                if len(service_words) <= 2 or len(matched_word) >= 4:
                    if score > best_score:
                        best_match = service_name
                        best_score = score
        
        if best_match:
            logger.info(f"Matched service (keyword match, score {best_score}): {best_match}")
            return best_match
        
        return None

