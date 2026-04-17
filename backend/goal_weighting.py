import math
from dataclasses import dataclass


@dataclass
class TOPSISResult:
    ranked_strategies: list  # [{strategy_key, display_name, topsis_score, rank, reasoning}]
    goal_weights: dict
    eligible_count: int


STRATEGIES = {
    "revocable_living_trust": {
        "display": "Revocable Living Trust",
        "tax_efficiency": 0.3,
        "risk_reduction": 0.6,
        "liquidity": 0.8,
        "simplicity": 0.9,
    },
    "fl_llc_charging_order": {
        "display": "FL LLC (Charging Order Protection)",
        "tax_efficiency": 0.5,
        "risk_reduction": 0.8,
        "liquidity": 0.6,
        "simplicity": 0.6,
    },
    "slat": {
        "display": "Spousal Lifetime Access Trust (SLAT)",
        "tax_efficiency": 0.85,
        "risk_reduction": 0.7,
        "liquidity": 0.3,
        "simplicity": 0.3,
    },
    "grat": {
        "display": "Grantor Retained Annuity Trust (GRAT)",
        "tax_efficiency": 0.9,
        "risk_reduction": 0.4,
        "liquidity": 0.2,
        "simplicity": 0.2,
    },
    "idgt_installment_sale": {
        "display": "Installment Sale to IDGT",
        "tax_efficiency": 0.9,
        "risk_reduction": 0.5,
        "liquidity": 0.2,
        "simplicity": 0.15,
    },
    "ilit": {
        "display": "Irrevocable Life Insurance Trust (ILIT)",
        "tax_efficiency": 0.8,
        "risk_reduction": 0.7,
        "liquidity": 0.3,
        "simplicity": 0.4,
    },
    "qprt": {
        "display": "Qualified Personal Residence Trust (QPRT)",
        "tax_efficiency": 0.75,
        "risk_reduction": 0.5,
        "liquidity": 0.1,
        "simplicity": 0.3,
    },
    "basic_gifting": {
        "display": "Annual Exclusion Gifting Program",
        "tax_efficiency": 0.6,
        "risk_reduction": 0.4,
        "liquidity": 0.7,
        "simplicity": 0.9,
    },
}

DIMENSIONS = ["tax_efficiency", "risk_reduction", "liquidity", "simplicity"]

# Basic eligibility gates — eliminates strategies that cannot apply to this client.
# Returns True if the strategy is eligible.
_ELIGIBILITY: dict[str, callable] = {
    "slat": lambda cd: cd.get("marital_status") == "married",
    "grat": lambda cd: (cd.get("estate_size_estimate") or 0) > 500_000,
    "idgt_installment_sale": lambda cd: (cd.get("estate_size_estimate") or 0) > 1_000_000,
    "qprt": lambda cd: cd.get("homestead_declared", False),
    "ilit": lambda cd: cd.get("has_life_insurance", False) or True,  # always available
}


def _is_eligible(strategy_key: str, client_data: dict) -> bool:
    check = _ELIGIBILITY.get(strategy_key)
    if check is None:
        return True
    return bool(check(client_data))


def _normalize_weights(goal_weights: dict) -> dict:
    """Ensure weights sum to 1.0 across the four dimensions."""
    total = sum(goal_weights.get(d, 0.0) for d in DIMENSIONS)
    if total == 0:
        return {d: 0.25 for d in DIMENSIONS}
    return {d: goal_weights.get(d, 0.0) / total for d in DIMENSIONS}


def _build_reasoning(
    strategy_key: str,
    scores: dict,
    weights: dict,
    topsis_score: float,
    rank: int,
) -> str:
    """Generate a plain-English explanation of the ranking."""
    # Find the dimension where this strategy scores highest relative to its weight contribution
    weighted = {d: scores[d] * weights[d] for d in DIMENSIONS}
    top_dim = max(weighted, key=weighted.get)
    bottom_dim = min(weighted, key=weighted.get)

    dim_labels = {
        "tax_efficiency": "tax efficiency",
        "risk_reduction": "risk reduction",
        "liquidity": "liquidity",
        "simplicity": "implementation simplicity",
    }

    return (
        f"Ranked #{rank} (TOPSIS score {topsis_score:.3f}). "
        f"Strongest contribution on {dim_labels[top_dim]} "
        f"(score {scores[top_dim]:.2f}, weight {weights[top_dim]:.2f}). "
        f"Weakest on {dim_labels[bottom_dim]} "
        f"(score {scores[bottom_dim]:.2f}). "
        f"{'High' if topsis_score >= 0.6 else 'Moderate' if topsis_score >= 0.4 else 'Lower'} "
        f"overall fit for this client's goal profile."
    )


def compute_topsis(client_data: dict, goal_weights: dict) -> TOPSISResult:
    """
    AHP goal weights + TOPSIS strategy ranking.

    goal_weights example:
        {"tax_efficiency": 0.35, "risk_reduction": 0.28, "liquidity": 0.22, "simplicity": 0.15}

    Steps:
      1. Filter strategies by eligibility
      2. Normalize weights to sum to 1
      3. Build weighted normalized decision matrix
      4. Identify ideal best (max) and ideal worst (min) per dimension
      5. Compute Euclidean distance to ideal best and ideal worst
      6. Compute relative closeness = dist_worst / (dist_best + dist_worst)
      7. Rank descending by relative closeness
    """
    if not goal_weights:
        goal_weights = {d: 0.25 for d in DIMENSIONS}

    weights = _normalize_weights(goal_weights)

    # Filter to eligible strategies only
    eligible = {
        k: v for k, v in STRATEGIES.items() if _is_eligible(k, client_data)
    }
    if not eligible:
        eligible = dict(STRATEGIES)  # fallback: all strategies

    keys = list(eligible.keys())
    n = len(keys)

    # Raw score matrix: rows = strategies, cols = dimensions
    raw: list[list[float]] = [
        [eligible[k][d] for d in DIMENSIONS] for k in keys
    ]

    # Step 1: Normalize each column by its Euclidean norm
    col_norms = [
        math.sqrt(sum(raw[i][j] ** 2 for i in range(n)))
        for j in range(len(DIMENSIONS))
    ]
    normalized = [
        [
            raw[i][j] / col_norms[j] if col_norms[j] > 0 else 0.0
            for j in range(len(DIMENSIONS))
        ]
        for i in range(n)
    ]

    # Step 2: Apply weights → weighted normalized matrix
    weighted_matrix = [
        [normalized[i][j] * weights[DIMENSIONS[j]] for j in range(len(DIMENSIONS))]
        for i in range(n)
    ]

    # Step 3: Ideal best (max) and ideal worst (min) per dimension
    ideal_best = [
        max(weighted_matrix[i][j] for i in range(n)) for j in range(len(DIMENSIONS))
    ]
    ideal_worst = [
        min(weighted_matrix[i][j] for i in range(n)) for j in range(len(DIMENSIONS))
    ]

    # Step 4: Euclidean distances
    def euclidean(row: list[float], reference: list[float]) -> float:
        return math.sqrt(sum((row[j] - reference[j]) ** 2 for j in range(len(DIMENSIONS))))

    distances_best = [euclidean(weighted_matrix[i], ideal_best) for i in range(n)]
    distances_worst = [euclidean(weighted_matrix[i], ideal_worst) for i in range(n)]

    # Step 5: Relative closeness
    closeness = [
        distances_worst[i] / (distances_best[i] + distances_worst[i])
        if (distances_best[i] + distances_worst[i]) > 0
        else 0.0
        for i in range(n)
    ]

    # Step 6: Sort descending
    ranked_indices = sorted(range(n), key=lambda i: closeness[i], reverse=True)

    ranked_strategies = []
    for rank, idx in enumerate(ranked_indices, start=1):
        key = keys[idx]
        strategy_scores = {d: eligible[key][d] for d in DIMENSIONS}
        ranked_strategies.append({
            "strategy_key": key,
            "display_name": eligible[key]["display"],
            "topsis_score": round(closeness[idx], 4),
            "rank": rank,
            "dimension_scores": strategy_scores,
            "reasoning": _build_reasoning(key, strategy_scores, weights, closeness[idx], rank),
        })

    return TOPSISResult(
        ranked_strategies=ranked_strategies,
        goal_weights=weights,
        eligible_count=n,
    )
