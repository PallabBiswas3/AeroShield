"""Uncertainty-weighted counterfactual intervention ranking."""
from __future__ import annotations

import math


ACTION_LIBRARY = {
    "Heavy Industrial": {"action": "Inspect high-emitting units and verify stack controls", "efficacy": 0.35, "cost_index": 3.0, "lead_hours": 6},
    "Industrial Stack": {"action": "Run stack inspection and enforce control-equipment operation", "efficacy": 0.32, "cost_index": 2.5, "lead_hours": 4},
    "Construction": {"action": "Deploy dust suppression and pause uncovered material handling", "efficacy": 0.45, "cost_index": 1.8, "lead_hours": 2},
}


def _probability(source, field: str, fallback: float) -> float:
    value = source.get(field)
    if value is None:
        value = source.get("attribution_probability", source.get("confidence_score", fallback))
    return max(0.0, min(float(value) / 100.0, 1.0))


def rank_interventions(predicted_pm25: float, attribution_results: list, exposed_population: int = 100_000, sensitive_sites: int = 0, healthy_reference: float = 30.0) -> dict:
    """Rank verification actions using an explicitly non-causal benefit proxy."""
    if not math.isfinite(float(predicted_pm25)) or not math.isfinite(float(healthy_reference)):
        raise ValueError("PM2.5 inputs must be finite")
    if predicted_pm25 < 0 or healthy_reference < 0:
        raise ValueError("predicted_pm25 cannot be negative")
    if exposed_population < 0 or sensitive_sites < 0:
        raise ValueError("exposure inputs cannot be negative")

    excess = max(0.0, predicted_pm25 - healthy_reference)
    actions = []
    for source in attribution_results:
        template = ACTION_LIBRARY.get(source.get("type"), {"action": "Conduct source verification and apply the registered control plan", "efficacy": 0.25, "cost_index": 2.5, "lead_hours": 6})
        mean_probability = _probability(source, "attribution_probability", 0.0)
        conservative_probability = _probability(source, "probability_p10", mean_probability * 100)
        expected_reduction = excess * mean_probability * template["efficacy"]
        robust_reduction = excess * conservative_probability * template["efficacy"]
        exposure_weight = 1.0 + min(sensitive_sites, 20) * 0.03
        score = robust_reduction * (exposed_population / 100_000) * exposure_weight
        score /= template["cost_index"] * (1.0 + template["lead_hours"] / 24.0)
        actions.append({
            "source_id": source.get("source_id"), "source_name": source.get("name", "Unknown source"),
            "source_type": source.get("type", "Unknown"), "action": template["action"],
            "expected_pm25_reduction": round(expected_reduction, 2), "robust_pm25_reduction": round(robust_reduction, 2),
            "expected_benefit_proxy": round(expected_reduction, 2), "robust_benefit_proxy": round(robust_reduction, 2),
            "attribution_probability": round(mean_probability * 100, 1), "cost_index": template["cost_index"],
            "lead_hours": template["lead_hours"], "priority_score": round(score, 3), "requires_field_verification": True,
        })
    actions.sort(key=lambda action: action["priority_score"], reverse=True)
    for index, action in enumerate(actions, start=1):
        action["rank"] = index
    top = actions[:3]
    robust_total = min(excess, sum(item["robust_pm25_reduction"] for item in top))
    return {
        "baseline_pm25": round(predicted_pm25, 2), "reference_pm25": healthy_reference,
        "exposed_population": exposed_population, "sensitive_sites": sensitive_sites, "ranked_actions": actions,
        "recommended_portfolio": {"source_ids": [item["source_id"] for item in top], "robust_pm25_reduction": round(robust_total, 2), "post_action_pm25": round(max(0.0, predicted_pm25 - robust_total), 2)},
        "method": "uncertainty-weighted screening heuristic",
        "assumptions": {
            "efficacy_values": "Unvalidated scenario assumptions from ACTION_LIBRARY, not measured causal effects.",
            "attribution": "Uses the 10th percentile relative share among inventoried sources.",
            "additivity": "Portfolio proxy sums source-level values and caps them at modeled excess PM2.5.",
        },
        "disclaimer": "Decision support only. Benefit proxies are not predicted causal reductions and require field verification.",
    }
