"""
Single entry point for resolving user text to a lead-type dict.

- LEAD_SELECTION: used when the user is explicitly choosing a lead type (numeric
  indices allowed; full fuzzy matching via DataExtractor.match_lead_type).
- MID_FLOW_SWITCH: used when detecting a deliberate switch away from an in-progress
  branch. Digits are ignored (they are usually workflow answers), and only
  high-confidence label/synonym matches apply.
"""
from enum import Enum
from typing import Any, Dict, List, Optional

from .data_extractors import DataExtractor


class LeadTypeResolutionMode(str, Enum):
    LEAD_SELECTION = "lead_selection"
    MID_FLOW_SWITCH = "mid_flow_switch"


def resolve_lead_type(
    user_text: str,
    lead_types: List[Dict[str, Any]],
    mode: LeadTypeResolutionMode,
) -> Optional[Dict[str, Any]]:
    """
    Resolve `user_text` to one of `lead_types` according to `mode`.

    Returns the matched lead-type dict, or None.
    """
    if not user_text or not lead_types:
        return None

    raw = str(user_text).strip()
    if not raw:
        return None

    if mode == LeadTypeResolutionMode.LEAD_SELECTION:
        if raw.isdigit():
            num = int(raw)
            if 1 <= num <= len(lead_types):
                candidate = lead_types[num - 1]
                if isinstance(candidate, dict):
                    return candidate
        return DataExtractor.match_lead_type(user_text, lead_types)

    # MID_FLOW_SWITCH
    if raw.isdigit():
        return None
    return DataExtractor.match_lead_type_strict(user_text, lead_types)
