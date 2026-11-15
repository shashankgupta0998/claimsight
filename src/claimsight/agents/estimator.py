"""Repair cost range from the damage manifest.

v0 uses a parts-and-labour table. Stretch goal: replace with a LightGBM model (HF task:
tabular regression) trained on manifest features, and keep this table as its baseline.
"""

from __future__ import annotations

from claimsight.schemas import CostEstimate, DamageItem

# Base USD cost at mid severity, by (part, damage_type). Illustrative figures only.
COST_TABLE: dict[tuple[str, str], float] = {
    ("bumper_front", "dent"): 450,
    ("bumper_front", "scratch"): 300,
    ("bumper_front", "crack"): 900,
    ("bumper_rear", "dent"): 450,
    ("bumper_rear", "scratch"): 300,
    ("bumper_rear", "crack"): 900,
    ("hood", "dent"): 600,
    ("hood", "scratch"): 400,
    ("roof", "dent"): 900,
    ("door", "dent"): 500,
    ("door", "scratch"): 350,
    ("fender", "dent"): 500,
    ("fender", "scratch"): 350,
    ("trunk", "dent"): 550,
    ("windshield", "crack"): 450,
    ("windshield", "glass_shatter"): 700,
    ("headlamp", "lamp_broken"): 400,
    ("taillamp", "lamp_broken"): 400,
    ("tire", "tire_flat"): 200,
    ("frame", "deform"): 3000,
}
DEFAULT_BASE = 500.0
SPREAD = 0.15


class Estimator:
    name = "cost-table-v0"

    def run(self, damage: list[DamageItem]) -> CostEstimate:
        items = []
        for d in damage:
            base = COST_TABLE.get((d.part, d.damage_type), DEFAULT_BASE)
            cost = base * (0.5 + 1.5 * d.severity)
            items.append({"part": d.part, "damage_type": d.damage_type, "usd": round(cost, 2)})
        total = sum(i["usd"] for i in items)
        return CostEstimate(
            low=round(total * (1 - SPREAD), 2),
            high=round(total * (1 + SPREAD), 2),
            line_items=items,
        )
