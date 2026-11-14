"""Sanctions-list name screening and hit adjudication.

Two separate steps, as in a real screening operation:

1. `screen_name`  - alert generation. The party name is compared with every listed
   name and alias. A score at or above ALERT_THRESHOLD is a hit that has to be
   worked. A score in the band just below it is kept as a near miss: code takes
   no action on it, but it is shown to the model.

2. `adjudicate_hits` - disambiguation. For each hit, the identifiers on the payment
   (country, registration number) are compared with the list entry:
       TRUE_MATCH      the same party
       FALSE_POSITIVE  a different party with a similar name
       UNRESOLVED      the identifiers available do not settle it

The similarity measure is deliberately plain and explainable: the standard library's
sequence ratio on normalised names, taken on the names as written and with their
words sorted (so word order does not matter). Legal-form suffixes are ignored.
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from models import NameScreeningHit

ALERT_THRESHOLD = 0.85
NEAR_MISS_THRESHOLD = 0.70

_LEGAL_FORMS = {
    "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "gmbh", "ag", "sa", "fze", "fzco", "dmcc", "sro", "plc", "bv", "nv", "lp", "llp", "pte",
}


def normalise_name(name: str) -> str:
    """Lower-case, drop punctuation and legal-form words: 'Acme Trading F.Z.E.' -> 'acme trading'."""
    text = name.lower().replace("&", " and ")
    text = re.sub(r"(?<=\b\w)\.(?=\w\b)", "", text)      # s.r.o. -> sro, F.Z.E. -> fze
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [t for t in text.split() if t not in _LEGAL_FORMS]
    return " ".join(tokens)


def name_similarity(a: str, b: str) -> float:
    """0..1 similarity of two names after normalisation; 1.0 means identical."""
    na, nb = normalise_name(a), normalise_name(b)
    if not na or not nb:
        return 0.0
    as_written = SequenceMatcher(None, na, nb).ratio()
    words_sorted = SequenceMatcher(
        None, " ".join(sorted(na.split())), " ".join(sorted(nb.split()))
    ).ratio()
    return round(max(as_written, words_sorted), 4)


def screen_name(
    party_name: str, sanctions_list: List[Dict], party_role: str = "beneficiary"
) -> List[NameScreeningHit]:
    """Best-scoring name or alias per list entry, for entries at or above the near-miss band."""
    hits: List[NameScreeningHit] = []
    for entry in sanctions_list:
        best_score, best_name = 0.0, entry["name"]
        for candidate in [entry["name"]] + list(entry.get("aliases", [])):
            score = name_similarity(party_name, candidate)
            if score > best_score:
                best_score, best_name = score, candidate
        if best_score < NEAR_MISS_THRESHOLD:
            continue
        hits.append(
            NameScreeningHit(
                list_id=entry["list_id"],
                listed_name=entry["name"],
                matched_on=best_name,
                party_role=party_role,
                party_name=party_name,
                score=best_score,
                level="ALERT" if best_score >= ALERT_THRESHOLD else "NEAR_MISS",
                listed_country=entry.get("country"),
                listed_registration_no=entry.get("registration_no"),
                listed_program=entry.get("program"),
            )
        )
    hits.sort(key=lambda h: (-h.score, h.list_id))
    return hits


def _compare(payment_value: Optional[str], listed_value: Optional[str]) -> str:
    if not payment_value or not listed_value:
        return "unknown"
    same = re.sub(r"[^a-z0-9]", "", payment_value.lower()) == re.sub(r"[^a-z0-9]", "", listed_value.lower())
    return "match" if same else "mismatch"


def adjudicate_hits(
    hits: List[NameScreeningHit],
    party_country: Optional[str],
    party_registration_no: Optional[str],
) -> List[NameScreeningHit]:
    """Fills in the identifier comparison and the verdict for every ALERT-level hit.

    Rules, most decisive first:
      * registration numbers match                          -> TRUE_MATCH
      * name identical, country matches, no number conflict  -> TRUE_MATCH
      * registration number and country both differ          -> FALSE_POSITIVE
      * anything else                                       -> UNRESOLVED
    Near misses are returned untouched: they were never alerts.
    """
    resolved: List[NameScreeningHit] = []
    for hit in hits:
        if hit.level != "ALERT":
            resolved.append(hit)
            continue
        country = _compare(party_country, hit.listed_country)
        registration = _compare(party_registration_no, hit.listed_registration_no)
        if registration == "match":
            verdict, reason = "TRUE_MATCH", "registration number is the same"
        elif hit.score >= 1.0 and country == "match" and registration != "mismatch":
            verdict, reason = "TRUE_MATCH", "identical name and same country; no conflicting identifier"
        elif registration == "mismatch" and country == "mismatch":
            verdict, reason = "FALSE_POSITIVE", "registration number and country both differ"
        else:
            verdict, reason = "UNRESOLVED", (
                f"similar name; country {country}, registration number {registration}"
            )
        resolved.append(hit.model_copy(update={
            "country_check": country,
            "registration_check": registration,
            "verdict": verdict,
            "verdict_reason": reason,
        }))
    return resolved
