"""Pure-Python nutrition maths (no TensorFlow, no Streamlit) so it is easy to test.

Every number in the nutrition table is for ONE standard serving. Everything the
app shows is that number multiplied by the serving multiplier the user chooses.
"""
from __future__ import annotations

import re

import pandas as pd

REQUIRED_COLUMNS = [
    "class_name", "display_name", "category", "serving_description",
    "serving_grams", "calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g",
]
NUMERIC_COLUMNS = ["serving_grams", "calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugar_g"]

# Alternative spellings that different dataset versions might use.
ALIASES = {
    "kathi_roll": "kaathi_rolls", "kathi_rolls": "kaathi_rolls", "kati_roll": "kaathi_rolls",
    "kaathi_roll": "kaathi_rolls", "pani_puri": "paani_puri", "panipuri": "paani_puri",
    "pakora": "pakode", "pakoda": "pakode", "pakoras": "pakode",
    "naan": "butter_naan", "roti": "chapati", "chappati": "chapati",
    "momo": "momos", "dosa": "masala_dosa", "kadhai_paneer": "kadai_paneer",
}


def normalize_label(label: str) -> str:
    """'Butter Naan' / 'butter-naan' -> 'butter_naan', then apply known aliases."""
    cleaned = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(cleaned, cleaned)


def load_nutrition_table(path) -> pd.DataFrame:
    """Read the CSV, check the columns, and index it by normalised class name."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Nutrition table is missing columns: {missing}")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["class_name"] = df["class_name"].map(normalize_label)
    if df["class_name"].duplicated().any():
        raise ValueError("Duplicate class_name rows in nutrition table.")
    return df.set_index("class_name", drop=False)


def scale_nutrition(row: pd.Series, servings: float) -> dict:
    """Nutrition for `servings` x one standard serving."""
    s = float(servings)
    return {
        "grams": row["serving_grams"] * s,
        "calories": row["calories"] * s,
        "protein": row["protein_g"] * s,
        "carbs": row["carbs_g"] * s,
        "fat": row["fat_g"] * s,
        "fiber": row["fiber_g"] * s,
        "sugar": row["sugar_g"] * s,
    }


def meal_totals(items: list[dict], table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """items = [{"name": class_name, "servings": 1.5}, ...].

    Returns (per-item DataFrame, totals dict). Unknown foods are skipped.
    """
    rows = []
    for item in items:
        key = normalize_label(item["name"])
        if key not in table.index:
            continue
        food = table.loc[key]
        n = scale_nutrition(food, item["servings"])
        rows.append({
            "class_name": key, "Food": food["display_name"], "Category": food["category"],
            "Servings": float(item["servings"]), "Grams (est.)": n["grams"],
            "Calories (kcal)": n["calories"], "Protein (g)": n["protein"],
            "Carbs (g)": n["carbs"], "Fat (g)": n["fat"],
        })
    df = pd.DataFrame(rows, columns=[
        "class_name", "Food", "Category", "Servings", "Grams (est.)",
        "Calories (kcal)", "Protein (g)", "Carbs (g)", "Fat (g)"])
    totals = {
        "calories": float(df["Calories (kcal)"].sum()) if len(df) else 0.0,
        "protein": float(df["Protein (g)"].sum()) if len(df) else 0.0,
        "carbs": float(df["Carbs (g)"].sum()) if len(df) else 0.0,
        "fat": float(df["Fat (g)"].sum()) if len(df) else 0.0,
        "grams": float(df["Grams (est.)"].sum()) if len(df) else 0.0,
    }
    return df, totals


def add_totals(a: dict, b: dict) -> dict:
    """Add two totals dicts key by key."""
    keys = ("calories", "protein", "carbs", "fat", "grams")
    return {k: float(a.get(k, 0.0)) + float(b.get(k, 0.0)) for k in keys}


def goal_progress(totals: dict, goals: dict) -> list[dict]:
    """Percent of each daily goal reached (not capped, so >100 means over goal)."""
    out = []
    for key, label, unit in [("calories", "Calories", "kcal"), ("protein", "Protein", "g"),
                             ("carbs", "Carbohydrates", "g"), ("fat", "Fat", "g")]:
        goal = float(goals[key])
        value = float(totals.get(key, 0.0))
        pct = (value / goal * 100.0) if goal > 0 else 0.0
        out.append({"key": key, "label": label, "unit": unit, "value": value,
                    "goal": goal, "percent": pct, "remaining": goal - value})
    return out


def calorie_split(totals: dict) -> dict:
    """Share of calories from protein/carbs/fat using 4/4/9 kcal per gram."""
    p, c, f = totals["protein"] * 4, totals["carbs"] * 4, totals["fat"] * 9
    total = p + c + f
    if total <= 0:
        return {"Protein": 0.0, "Carbohydrates": 0.0, "Fat": 0.0}
    return {"Protein": p / total * 100, "Carbohydrates": c / total * 100, "Fat": f / total * 100}


# ---------------------------------------------------------------------------
# Dish-description adjustments (simple, documented assumptions)
# ---------------------------------------------------------------------------
# Each adjustment adds/removes grams per 100 g of the portion. These are rough
# rule-of-thumb values (about 1 teaspoon of oil is ~4.5 g fat), NOT measurements:
# a text description cannot tell how much oil was really used.
MODIFIERS = [
    {"id": "extra_oil", "label": "Extra oil / oily", "fat": 5.0,
     "keywords": ["extra oil", "more oil", "oily", "greasy", "lots of oil", "heavy oil"]},
    {"id": "butter_ghee", "label": "Butter / ghee", "fat": 4.0,
     "keywords": ["butter", "buttery", "ghee"]},
    {"id": "cream", "label": "Cream / malai", "fat": 3.0,
     "keywords": ["cream", "creamy", "malai"]},
    {"id": "extra_cheese", "label": "Extra cheese", "fat": 4.0, "protein": 3.0,
     "keywords": ["extra cheese", "more cheese", "cheesy", "cheese"]},
    {"id": "less_oil", "label": "Less oil", "fat": -3.0,
     "keywords": ["less oil", "low oil", "light oil", "no oil", "oil free", "oil-free", "air fried", "dry roasted"]},
    {"id": "extra_sweet", "label": "Extra sugar / syrup", "carbs": 8.0, "sugar": 8.0,
     "keywords": ["extra sugar", "more sugar", "extra sweet", "very sweet", "sweetened", "syrupy"]},
    {"id": "less_sugar", "label": "Less sugar", "carbs": -5.0, "sugar": -5.0,
     "keywords": ["less sugar", "low sugar", "no sugar", "sugar free", "sugar-free", "unsweetened", "less sweet"]},
]
MODIFIER_BY_ID = {m["id"]: m for m in MODIFIERS}
CONFLICTS = [("extra_oil", "less_oil"), ("extra_sweet", "less_sugar")]


def detect_modifiers(text: str) -> list[str]:
    """Find adjustment ids mentioned in a description. Handles 'no/without/not <keyword>'."""
    text = str(text).lower()
    found = []
    for m in MODIFIERS:
        for kw in m["keywords"]:
            if not re.search(r"\b" + re.escape(kw) + r"\b", text):
                continue
            if re.search(r"\b(?:no|without|not)\s+(?:\w+\s+)?" + re.escape(kw) + r"\b", text):
                continue
            found.append(m["id"])
            break
    for a, b in CONFLICTS:                      # contradictory words cancel out
        if a in found and b in found:
            found.remove(a)
            found.remove(b)
    return found


def estimate_nutrition(table: pd.DataFrame, food: str, grams: float,
                       modifier_ids=(), strength: float = 1.0) -> dict:
    """Estimated nutrition for `grams` of `food`, plus optional description adjustments.

    Steps: (1) scale the table's standard-serving values to the entered grams,
    (2) add each adjustment (per 100 g x grams x strength), (3) recompute calories
    from the changed macros (4 kcal/g protein and carbs, 9 kcal/g fat).
    """
    row = table.loc[normalize_label(food)]
    grams = max(float(grams), 0.0)
    servings = grams / float(row["serving_grams"])
    base = scale_nutrition(row, servings)
    base["grams"] = grams
    out = {k: base[k] for k in ("protein", "carbs", "fat", "fiber", "sugar")}
    applied = []
    per100 = grams / 100.0 * float(strength)
    for mid in modifier_ids:
        m = MODIFIER_BY_ID[mid]
        delta = {k: m.get(k, 0.0) * per100 for k in ("fat", "carbs", "protein", "sugar")}
        for k, v in delta.items():
            out[k] += v
        applied.append({"label": m["label"], **{f"{k}_g": round(v, 1) for k, v in delta.items() if v}})
    for k in ("protein", "carbs", "fat", "sugar"):     # never below zero
        out[k] = max(out[k], 0.0)
    out["sugar"] = min(out["sugar"], out["carbs"])
    out["calories"] = base["calories"] + 9 * (out["fat"] - base["fat"]) \
        + 4 * (out["carbs"] - base["carbs"]) + 4 * (out["protein"] - base["protein"])
    out.update({"grams": grams, "servings": servings, "base": base, "applied": applied})
    return out
