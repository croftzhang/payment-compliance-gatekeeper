"""Reference data for the ownership graph: one source for the local graph and for Spanner.

Every entity, name and identifier here is fictional. `sanction_list` values are
labels on synthetic data and do not refer to any real designation.

The same rows are used in two places:
  * the local graph backend (`SpannerGraphComplianceClient(use_mock=True)`)
  * `seed_spanner.py`, which writes them to the Entities / EntityOwnership tables

so both backends answer from identical data and can be compared row for row.

Ownership edges read "owner holds `pct` of target". The recorded direct holders of
any entity never sum to more than 100%; where they sum to less, the remainder is
simply not on record.
"""

from datetime import date
from typing import Dict, List, Tuple

# (entity_id, entity_name, entity_type, jurisdiction, is_sanctioned, sanction_list, risk_score)
ENTITIES: List[Tuple[str, str, str, str, bool, object, float]] = [
    ("ENT-ORIG-US01", "Great Lakes Industrial Supply Inc", "CORPORATE", "US", False, None, 0.10),
    # --- TX-2026-9981: sanctioned UBO behind two shell layers, 32% + 20% = 52% -> BLOCK
    ("ENT-BEN-9001", "Baltic Meridian Shipping Ltd", "CORPORATE", "CY", False, None, 0.70),
    ("ENT-SHELL-A", "Harborline Nominees Ltd", "CORPORATE", "VG", False, None, 0.80),
    ("ENT-SHELL-B", "Tidewater Holdings SA", "CORPORATE", "PA", False, None, 0.80),
    ("ENT-UBO-SDN1", "Vostok Maritime Holdings Ltd", "CORPORATE", "VG", True, "OFAC-SDN", 1.00),
    ("ENT-UBO-CLEAN1", "Alpine Industrial Partners AG", "CORPORATE", "CH", False, None, 0.15),
    # --- TX-2026-7702: 30% offshore UBO, not sanctioned -> MANUAL_ESCALATION
    ("ENT-BEN-7700", "Coral Reef Advisory Ltd", "CORPORATE", "KY", False, None, 0.55),
    ("ENT-HOLD-KY", "Seven Mile Holdings Ltd", "CORPORATE", "KY", False, None, 0.60),
    ("ENT-UBO-OFFSHORE1", "Archipelago Trade Ventures Ltd", "CORPORATE", "KY", False, None, 0.55),
    # --- TX-2026-1004: wholly owned domestic company -> APPROVE
    ("ENT-BEN-1002", "Prairie Office Supply Co", "CORPORATE", "US", False, None, 0.05),
    ("ENT-UBO-DOMESTIC", "Midwest Logistics Corp", "CORPORATE", "US", False, None, 0.05),
    # --- Extended cases ---
    # Clean cross-border beneficiary (used by the text-signal cases)
    ("ENT-BEN-2200", "Rheinwerk Precision GmbH", "CORPORATE", "DE", False, None, 0.10),
    ("ENT-UBO-CLEAN2", "Rheinland Maschinenbau Holding GmbH", "CORPORATE", "DE", False, None, 0.10),
    # 49.9% held by a sanctioned party: just under the 50% threshold -> MANUAL_ESCALATION
    ("ENT-BEN-4990", "Levant Gate Commodities DMCC", "CORPORATE", "AE", False, None, 0.65),
    ("ENT-UBO-SDN2", "Korvald Meridian Trading FZE", "CORPORATE", "AE", True, "OFAC-SDN", 1.00),
    ("ENT-UBO-CLEAN3", "Jumeirah Bay Family Office Ltd", "CORPORATE", "AE", False, None, 0.20),
    # Exactly 50%, split across two different sanctioned parties (30% + 20%) -> BLOCK
    ("ENT-BEN-5000", "Aegean Crest Trading Ltd", "CORPORATE", "MT", False, None, 0.70),
    ("ENT-SHELL-C", "Quay Street Nominees Ltd", "CORPORATE", "MT", False, None, 0.80),
    ("ENT-SHELL-D", "Bluewater Fiduciary Ltd", "CORPORATE", "SC", False, None, 0.80),
    ("ENT-UBO-SDN3", "Ostrava Dune Logistics LLC", "CORPORATE", "AE", True, "OFAC-SDN", 1.00),
    ("ENT-UBO-CLEAN4", "Valletta Harbour Partners Ltd", "CORPORATE", "MT", False, None, 0.20),
    # Three-hop chain with a cross-holding (cycle) between the two shells:
    # 90% x 90% x 70% = 56.7% -> BLOCK
    ("ENT-BEN-3300", "Northlight Bulk Carriers Ltd", "CORPORATE", "CY", False, None, 0.70),
    ("ENT-SHELL-E", "Kestrel Point Holdings Ltd", "CORPORATE", "VG", False, None, 0.85),
    ("ENT-SHELL-F", "Marlin Row Investments Ltd", "CORPORATE", "VG", False, None, 0.85),
]

_RESTRUCTURED = date(2026, 11, 13)   # the day of the 9981 wire (KYC log: split shortly before it)
_LONG_STANDING = date(2021, 3, 15)

# (owner_id, target_id, ownership_pct, effective_date)
OWNERSHIP: List[Tuple[str, str, float, date]] = [
    # ENT-BEN-9001: direct holders 40% + 25% + 35% = 100%
    ("ENT-SHELL-A", "ENT-BEN-9001", 0.40, _RESTRUCTURED),
    ("ENT-SHELL-B", "ENT-BEN-9001", 0.25, _RESTRUCTURED),
    ("ENT-UBO-CLEAN1", "ENT-BEN-9001", 0.35, _LONG_STANDING),
    ("ENT-UBO-SDN1", "ENT-SHELL-A", 0.80, _LONG_STANDING),   # 0.80 x 0.40 = 0.32
    ("ENT-UBO-SDN1", "ENT-SHELL-B", 0.80, _LONG_STANDING),   # 0.80 x 0.25 = 0.20
    # ENT-BEN-7700: 0.50 x 0.60 = 0.30 via one holding company
    ("ENT-HOLD-KY", "ENT-BEN-7700", 0.60, _LONG_STANDING),
    ("ENT-UBO-OFFSHORE1", "ENT-HOLD-KY", 0.50, _LONG_STANDING),
    # ENT-BEN-1002 / ENT-BEN-2200: wholly owned, clean
    ("ENT-UBO-DOMESTIC", "ENT-BEN-1002", 1.00, _LONG_STANDING),
    ("ENT-UBO-CLEAN2", "ENT-BEN-2200", 1.00, _LONG_STANDING),
    # ENT-BEN-4990: 49.9% sanctioned + 50.1% clean
    ("ENT-UBO-SDN2", "ENT-BEN-4990", 0.499, _LONG_STANDING),
    ("ENT-UBO-CLEAN3", "ENT-BEN-4990", 0.501, _LONG_STANDING),
    # ENT-BEN-5000: 0.60 x 0.50 = 0.30 (SDN2) plus 1.00 x 0.20 = 0.20 (SDN3), 30% clean
    ("ENT-SHELL-C", "ENT-BEN-5000", 0.50, _LONG_STANDING),
    ("ENT-SHELL-D", "ENT-BEN-5000", 0.20, _LONG_STANDING),
    ("ENT-UBO-CLEAN4", "ENT-BEN-5000", 0.30, _LONG_STANDING),
    ("ENT-UBO-SDN2", "ENT-SHELL-C", 0.60, _LONG_STANDING),
    ("ENT-UBO-SDN3", "ENT-SHELL-D", 1.00, _LONG_STANDING),
    # ENT-BEN-3300: F holds 90%, E holds 90% of F, SDN3 holds 70% of E.
    # F also holds 10% of E, which closes a cycle E -> F -> E that the traversal must not follow.
    ("ENT-SHELL-F", "ENT-BEN-3300", 0.90, _LONG_STANDING),
    ("ENT-UBO-CLEAN4", "ENT-BEN-3300", 0.10, _LONG_STANDING),
    ("ENT-SHELL-E", "ENT-SHELL-F", 0.90, _LONG_STANDING),
    ("ENT-UBO-SDN3", "ENT-SHELL-E", 0.70, _LONG_STANDING),
    ("ENT-SHELL-F", "ENT-SHELL-E", 0.10, _LONG_STANDING),
]

# Synthetic sanctions list. `entity_id` links an entry to its node in the ownership graph.
SANCTIONS_LIST: List[Dict] = [
    {
        "list_id": "SDN-90001", "entity_id": "ENT-UBO-SDN1",
        "name": "Vostok Maritime Holdings Ltd", "aliases": ["Vostok Maritime"],
        "country": "VG", "registration_no": "VG-1482207", "program": "OFAC-SDN",
    },
    {
        "list_id": "SDN-90002", "entity_id": "ENT-UBO-SDN2",
        "name": "Korvald Meridian Trading FZE", "aliases": ["Korvald Trading"],
        "country": "AE", "registration_no": "AE-JAFZ-55310", "program": "OFAC-SDN",
    },
    {
        "list_id": "SDN-90003", "entity_id": "ENT-UBO-SDN3",
        "name": "Ostrava Dune Logistics LLC", "aliases": ["Ostrava Dune"],
        "country": "AE", "registration_no": "AE-DXB-99120", "program": "OFAC-SDN",
    },
]

BENEFICIARY_IDS: List[str] = [
    "ENT-BEN-9001", "ENT-BEN-7700", "ENT-BEN-1002", "ENT-BEN-2200",
    "ENT-BEN-4990", "ENT-BEN-5000", "ENT-BEN-3300",
]

ENTITY_COLUMNS = (
    "entity_id", "entity_name", "entity_type", "jurisdiction",
    "is_sanctioned", "sanction_list", "risk_score",
)
OWNERSHIP_COLUMNS = ("owner_id", "target_id", "ownership_pct", "effective_date")


def entity_index() -> Dict[str, dict]:
    """entity_id -> row as a dict."""
    return {row[0]: dict(zip(ENTITY_COLUMNS, row)) for row in ENTITIES}


def direct_holdings_total() -> Dict[str, float]:
    """target_id -> sum of recorded direct ownership (used by a consistency test)."""
    totals: Dict[str, float] = {}
    for _owner, target, pct, _eff in OWNERSHIP:
        totals[target] = round(totals.get(target, 0.0) + pct, 6)
    return totals
