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


def _lead_types_ordered_for_index(lead_types: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Same ordering as greeting buttons (by `order` when present)."""
    items = [lt for lt in lead_types if isinstance(lt, dict)]
    if not items:
        return []
    if any(x.get("order") is not None for x in items):
        try:
            return sorted(
                items,
                key=lambda x: (x.get("order") if x.get("order") is not None else 10**9),
            )
        except Exception:
            return items
    return items


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

    normalized = DataExtractor.normalize_lead_type_user_input(str(user_text))
    if not normalized:
        return None

    if mode == LeadTypeResolutionMode.LEAD_SELECTION:
        ordered = _lead_types_ordered_for_index(lead_types)
        if normalized.isdigit():
            num = int(normalized)
            if 1 <= num <= len(ordered):
                return ordered[num - 1]
        matched = DataExtractor.match_lead_type(normalized, lead_types)
        if matched:
            return matched
        # Extra pass: some clients send labels where only the text part matches DB (emoji drift).
        alt = DataExtractor._text_after_leading_emoji(normalized)
        if alt and alt != normalized:
            matched_alt = DataExtractor.match_lead_type(alt, lead_types)
            if matched_alt:
                return matched_alt
        return None

    # MID_FLOW_SWITCH
    if normalized.isdigit():
        return None
    return DataExtractor.match_lead_type_strict(normalized, lead_types)
