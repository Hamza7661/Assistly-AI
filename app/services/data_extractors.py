"""Structured data extraction for conversation data"""
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

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
        
        # If it's 2-50 characters and purely letters/spaces, treat as a full name
        # But exclude if it has more than 4 words (likely a sentence, not a name)
        if 2 <= len(text) <= 50 and re.match(r'^[A-Za-z\s\-\']+$', text):
            if len(text.split()) > 4:
                return None
            name = ' '.join(word.capitalize() for word in text.split())
            logger.info(f"Extracted name: {name}")
            return name

        # Fallback: user may have typed their name alongside digits (e.g. "John 123456").
        # Strip digit-only tokens and try again with whatever letters remain.
        alpha_only = ' '.join(w for w in text.split() if re.match(r'^[A-Za-z\-\']+$', w))
        if 2 <= len(alpha_only) <= 50 and alpha_only.strip():
            if len(alpha_only.split()) <= 4:
                name = ' '.join(word.capitalize() for word in alpha_only.split())
                logger.info(f"Extracted name (partial, stripped digits): {name}")
                return name

        return None
    
    # Optional fallback synonyms per value – only used when the app's lead type has no synonyms from DB.
    # Key = normalized value; use for any industry. Apps can override/expand via integration synonyms.
    LEAD_TYPE_SYNONYMS_FALLBACK: Dict[str, List[str]] = {
        "order": ["order", "آرڈر", "ordre", "orden", "bestellung", "ordine", "pedido"],
        "menu": ["menu", "مينو", "مینو", "mönü", "menü", "menú", "मेन्यू", "mónu", "meniu"],
        "catering": ["catering", "کیٹرنگ", "restauration"],
        "reservation": ["reservation", "رزرویشن", "reserva", "réservation", "reservierung", "आरक्षण"],
        "allergies": ["allergies", "halal", "الرجی", "حلال", "एलर्जी"],
        "halal": ["halal", "حلال", "हलाल"],
        "info": ["info", "contact", "معلومات", "رابطہ", "जानकारी", "संपर्क", "info & contact"],
        "contact": ["contact", "رابطہ", "संपर्क", "contacto", "kontakt"],
        "complaint": ["complaint", "شکایت", "शिकायत", "queja", "beschwerde", "réclamation"],
        "appointment": ["appointment", "ملاقات", "termin", "rendez-vous", "cita", "appuntamento", "अपॉइंटमेंट"],
        "callback": ["callback", "کال بیک", "rückruf", "rappel", "devolución de llamada", "कॉलबैक"],
        "information": ["information", "info", "معلومات", "जानकारी", "información", "information request"],
    }

    @staticmethod
    def normalize_lead_type_user_input(user_input: str) -> str:
        """
        NFC unicode, strip zero-width / BOM noise, NBSP → space, collapse whitespace.
        Web widget sends exact <button> labels; invisible chars break substring matching.
        """
        if not user_input:
            return ""
        s = unicodedata.normalize("NFC", str(user_input).strip())
        for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\ufe0f"):
            s = s.replace(ch, "")
        s = s.replace("\u00a0", " ")
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    @staticmethod
    def _text_after_leading_emoji(s: str) -> str:
        """First letter/digit onward (skips leading emoji/punctuation/spaces). Helps widget label vs DB emoji drift."""
        if not s:
            return ""
        i = 0
        while i < len(s):
            if s[i].isalnum():
                break
            i += 1
        return s[i:].strip()

    @staticmethod
    def match_lead_type(user_input: str, lead_types: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Match user input to a lead type. Uses per-app synonyms from DB first; fallback map only for current lead types."""
        user_stripped = DataExtractor.normalize_lead_type_user_input(user_input)
        if not user_stripped or not lead_types:
            return None

        user_input_lower = user_stripped.lower()

        # Pass 0.5: same as button label but leading emoji/symbols differ from stored combined label
        alt_user = DataExtractor._text_after_leading_emoji(user_stripped)
        if alt_user and alt_user.lower() != user_input_lower:
            alt_lower = alt_user.lower()
            for lt in lead_types:
                if not isinstance(lt, dict):
                    continue
                text = str(lt.get("text", "") or "").strip()
                if text and text.lower() == alt_lower:
                    logger.info("Matched lead type by text after leading symbols: %s", lt.get("value"))
                    return lt
                value = str(lt.get("value", "") or "").strip()
                if value and value.lower() == alt_lower:
                    logger.info("Matched lead type by value after leading symbols: %s", lt.get("value"))
                    return lt

        # Pass 0: exact match to label as shown on web buttons (emoji + text from integration)
        for lt in lead_types:
            if not isinstance(lt, dict):
                continue
            emoji = (lt.get("emoji") or "").strip()
            text = str(lt.get("text", "") or "").strip()
            if emoji and text:
                combined = f"{emoji} {text}".strip()
                if combined.lower() == user_input_lower:
                    logger.info("Matched lead type by emoji+text label (exact): %s", lt.get("value"))
                    return lt
                if (
                    DataExtractor._normalize_lead_match_phrase(combined)
                    == DataExtractor._normalize_lead_match_phrase(user_stripped)
                ):
                    logger.info("Matched lead type by emoji+text label (normalized): %s", lt.get("value"))
                    return lt
            elif text and text.lower() == user_input_lower:
                logger.info("Matched lead type by text only (exact): %s", lt.get("text"))
                return lt

        # Pass 0: synonyms from database (per-app integration) – any industry, any lead type
        for lt in lead_types:
            if not isinstance(lt, dict):
                continue
            db_synonyms = lt.get("synonyms")
            if not isinstance(db_synonyms, list):
                continue
            for syn in db_synonyms:
                if not syn:
                    continue
                s = str(syn).strip()
                if user_stripped == s or user_input_lower == s.lower():
                    logger.info(f"Matched lead type by DB synonym: '{user_stripped}' -> {lt.get('value')} / {lt.get('text')}")
                    return lt
                if user_stripped and s and (user_stripped in s or s in user_stripped):
                    logger.info(f"Matched lead type by DB synonym: '{user_stripped}' -> {lt.get('value')}")
                    return lt

        # Pass 1: fallback map only for this app's lead type values (no hardcoding for other industries)
        for lt in lead_types:
            if not isinstance(lt, dict):
                continue
            value = str(lt.get("value", "")).lower().strip()
            text = str(lt.get("text", "")).lower().strip()
            fallback = DataExtractor.LEAD_TYPE_SYNONYMS_FALLBACK.get(value) or DataExtractor.LEAD_TYPE_SYNONYMS_FALLBACK.get(text) or []
            for syn in fallback:
                syn_lower = syn.lower().strip()
                if user_stripped == syn or user_input_lower == syn_lower:
                    logger.info(f"Matched lead type by fallback synonym: user input -> {lt.get('value')} / {lt.get('text')}")
                    return lt
                if user_stripped and syn and (user_stripped in syn or syn in user_stripped):
                    logger.info(f"Matched lead type by fallback synonym: '{user_stripped}' -> {lt.get('value')}")
                    return lt

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
            text_normalized = re.sub(r'[^\w\s]', '', text)
            value_normalized = re.sub(r'[^\w\s]', '', value)
            user_input_normalized = re.sub(r'[^\w\s]', '', user_input_lower)
 
            def _meaningful_words(phrase: str) -> set[str]:
                return {
                    word
                    for word in phrase.split()
                    if len(word) > 2 and word not in common_words
                }

            # Check if key words from text/value are in user input
            if text_normalized:
                text_words = _meaningful_words(text_normalized)
                user_words = _meaningful_words(user_input_normalized)
                # If majority of meaningful words match, consider it a match
                if text_words and len(text_words.intersection(user_words)) >= min(2, len(text_words) * 0.6):
                    logger.info(f"Matched lead type by text words: {lt.get('text')}")
                    return lt
 
            if text and (text in user_input_lower or user_input_lower in text):
                logger.info(f"Matched lead type by text substring: {lt.get('text')}")
                return lt
            
            # Check value matching with word overlap
            if value_normalized:
                value_words = _meaningful_words(value_normalized)
                user_words = _meaningful_words(user_input_normalized)
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
    def _normalize_lead_match_phrase(s: str) -> str:
        """Lowercase, strip punctuation/emoji noise, collapse whitespace — for strict equality only."""
        t = (s or "").lower().strip()
        t = re.sub(r"[^\w\s]", "", t)
        return " ".join(t.split())

    @staticmethod
    def match_lead_type_strict(
        user_input: str, lead_types: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        High-confidence lead-type match for mid-flow switches: exact synonym, exact
        text/value, or normalized-exact against those. No substring containment,
        word-overlap, or keyword scoring (avoids hijacking workflow answers).
        """
        user_stripped = DataExtractor.normalize_lead_type_user_input(user_input)
        if not user_stripped or not lead_types:
            return None

        user_input_lower = user_stripped.lower()
        user_norm = DataExtractor._normalize_lead_match_phrase(user_stripped)

        # DB synonyms — exact or normalized-exact only
        for lt in lead_types:
            if not isinstance(lt, dict):
                continue
            db_synonyms = lt.get("synonyms")
            if not isinstance(db_synonyms, list):
                continue
            for syn in db_synonyms:
                if not syn:
                    continue
                s = str(syn).strip()
                if user_stripped == s or user_input_lower == s.lower():
                    logger.info(
                        "Strict matched lead type by DB synonym (exact): '%s' -> %s",
                        user_stripped,
                        lt.get("value"),
                    )
                    return lt
                if user_norm and DataExtractor._normalize_lead_match_phrase(s) == user_norm:
                    logger.info(
                        "Strict matched lead type by DB synonym (normalized): '%s' -> %s",
                        user_stripped,
                        lt.get("value"),
                    )
                    return lt

        # Fallback synonym map — exact or normalized-exact only
        for lt in lead_types:
            if not isinstance(lt, dict):
                continue
            value = str(lt.get("value", "")).lower().strip()
            text = str(lt.get("text", "")).lower().strip()
            fallback = (
                DataExtractor.LEAD_TYPE_SYNONYMS_FALLBACK.get(value)
                or DataExtractor.LEAD_TYPE_SYNONYMS_FALLBACK.get(text)
                or []
            )
            for syn in fallback:
                syn_lower = str(syn).lower().strip()
                if user_stripped == str(syn).strip() or user_input_lower == syn_lower:
                    logger.info(
                        "Strict matched lead type by fallback synonym (exact): -> %s",
                        lt.get("value"),
                    )
                    return lt
                if user_norm and DataExtractor._normalize_lead_match_phrase(str(syn)) == user_norm:
                    logger.info(
                        "Strict matched lead type by fallback synonym (normalized): -> %s",
                        lt.get("value"),
                    )
                    return lt

        # Exact text / value (display label or internal value)
        for lt in lead_types:
            if not isinstance(lt, dict):
                continue
            text = str(lt.get("text", "")).lower().strip()
            if text and text == user_input_lower:
                logger.info("Strict matched lead type by exact text: %s", lt.get("text"))
                return lt
            value = str(lt.get("value", "")).lower().strip()
            if value and value == user_input_lower:
                logger.info("Strict matched lead type by exact value: %s", lt.get("value"))
                return lt
            if user_norm:
                lt_text_raw = str(lt.get("text", "") or "")
                lt_value_raw = str(lt.get("value", "") or "")
                if lt_text_raw and DataExtractor._normalize_lead_match_phrase(lt_text_raw) == user_norm:
                    logger.info("Strict matched lead type by normalized text: %s", lt.get("text"))
                    return lt
                if lt_value_raw and DataExtractor._normalize_lead_match_phrase(lt_value_raw) == user_norm:
                    logger.info("Strict matched lead type by normalized value: %s", lt.get("value"))
                    return lt

        # Include configured emoji in label: compare "emoji + text" as shown on buttons
        for lt in lead_types:
            if not isinstance(lt, dict):
                continue
            emoji = (lt.get("emoji") or "").strip()
            base = str(lt.get("text", "") or lt.get("value", "") or "").strip()
            if emoji and base:
                combined = f"{emoji} {base}".strip()
                if combined.lower() == user_input_lower:
                    logger.info("Strict matched lead type by emoji+text: %s", lt.get("text"))
                    return lt
                if user_norm and DataExtractor._normalize_lead_match_phrase(combined) == user_norm:
                    logger.info("Strict matched lead type by normalized emoji+text: %s", lt.get("text"))
                    return lt

        return None
    
    @staticmethod
    def match_service(user_input: str, services: List[Any]) -> Optional[str]:
        """Match user input to a service - handles partial matches and single keywords"""
        if not user_input or not services:
            return None
        
        user_input_lower = user_input.lower().strip()
        # Remove punctuation for better matching
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

