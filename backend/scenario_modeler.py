from dataclasses import dataclass

BADGE_WEIGHTS = {
    "B1_insider_transfer": 2,
    "B2_retained_control": 2,
    "B3_concealment": 3,
    "B4_absconding": 3,
    "B5_asset_removal": 2,
    "B6_insolvency_at_transfer": 3,
    "B7_timing_before_debt": 2,
    "B8_substantially_all_assets": 3,
    "B9_flight_risk": 3,
    "B10_post_transfer_insolvency": 3,
    "B11_insider_before_suit": 2,
}
MAX_SCORE = 28


@dataclass
class BadgeResult:
    badges: list
    score: float
    max_score: int
    classification: str
    claude_input: str
