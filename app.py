"""FitBite — updated Streamlit app built on the separately supplied latest app.
The older artifact ZIP is used only for model and supporting assets.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps

from nutrition_calc import MODIFIERS, detect_modifiers, estimate_nutrition, load_nutrition_table, normalize_label

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "best_model.keras"
CLASS_NAMES_PATH = HERE / "class_names.json"
IMG_SIZE = 224
LOW_CONFIDENCE = 0.50
MEAL_CATEGORIES = ["Breakfast", "Lunch", "Snacks", "Dinner"]
FITNESS_GOALS = ["Select a fitness goal", "General fitness", "Lose weight", "Maintain weight", "Gain weight", "Build muscle", "Other"]
GENDERS = ["Select gender", "Prefer not to say", "Female", "Male", "Non-binary", "Other"]
ACTIVITY_LEVELS = ["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"]
ACTIVITY_FACTORS = {"Sedentary": 1.2, "Lightly active": 1.375, "Moderately active": 1.55, "Very active": 1.725, "Extra active": 1.9}

# Explicit synonym mapping only where the nutrition-table item is genuinely the same food.
# Missing entries are intentionally not replaced with unrelated dishes.
CLASS_TO_TABLE = {
    "cholebhature": "chole_bhature", "dhokla": "dhokla", "dosa": "masala_dosa",
    "jalebi": "jalebi", "kathiroll": "kaathi_rolls", "naan": "butter_naan",
    "pakora": "pakode", "panipuri": "paani_puri", "pavbhaji": "pav_bhaji",
}

st.set_page_config(page_title="FitBite | Food & Wellness", page_icon="🥗", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
:root { --fb-bg:#f7f8f5; --fb-card:#ffffff; --fb-green:#087443; --fb-green2:#e5f3eb; --fb-ink:#1f2933; --fb-muted:#68756d; --fb-line:#e7ece8; }
.stApp { background:var(--fb-bg); color:var(--fb-ink); }
[data-testid="stHeader"] { background:rgba(247,248,245,.92); }
.block-container { padding-top:1.25rem; padding-bottom:3rem; max-width:1400px; }
[data-testid="stSidebar"] { background:#f0f5f0; border-right:1px solid var(--fb-line); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#52645a; }
h1,h2,h3 { color:#172820; letter-spacing:-.025em; }
h1 { font-weight:800; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--fb-line); border-radius:18px; padding:16px 18px; box-shadow:0 4px 18px rgba(20,55,35,.035); }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--fb-line)!important; border-radius:20px!important; background:#fff; }
.stButton>button, .stFormSubmitButton>button { border-radius:14px; min-height:2.65rem; font-weight:650; border-color:#dbe7df; }
.stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primary"] { background:var(--fb-green); border-color:var(--fb-green); color:#fff; }
.stProgress > div > div > div > div { background:var(--fb-green); }
[data-testid="stFileUploader"] { background:#fff; border:1px dashed #9bb9a6; border-radius:18px; padding:12px; }
[data-testid="stTabs"] button { font-weight:650; }
.fb-eyebrow { color:var(--fb-green); text-transform:uppercase; letter-spacing:.11em; font-size:.74rem; font-weight:800; }
.fb-muted { color:var(--fb-muted); }
.fb-hero { background:linear-gradient(115deg,#e7f3eb,#f8fbf8 70%); border:1px solid #dcebe1; border-radius:24px; padding:22px 24px; margin:4px 0 18px; }
.fb-pill { display:inline-block; padding:5px 11px; border-radius:30px; background:#e6f3eb; color:#087443; font-size:.8rem; font-weight:700; }
.fb-card-title { font-weight:750; font-size:1.05rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_table() -> pd.DataFrame:
    return load_nutrition_table(HERE / "nutrition_table.csv")


@st.cache_resource(show_spinner="Loading your trained food model…")
def get_model():
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        return None, None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        return model, names
    except Exception as exc:
        return None, f"Model could not be loaded: {type(exc).__name__}: {exc}"


@st.cache_data
def get_metadata():
    p = HERE / "model_metadata.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_supabase_credentials():
    local = HERE / ".streamlit" / "secrets.toml"
    sources = []
    if local.exists():
        try:
            import tomllib
            with local.open("rb") as f:
                sources.append(tomllib.load(f).get("supabase", {}))
        except Exception:
            return None, None, "The project secrets file could not be parsed. Check its TOML syntax and [supabase] section."
    try:
        secrets = st.secrets
        sources.append(secrets.get("supabase", {}))
        sources.append({"url": secrets.get("SUPABASE_URL"), "anon_key": secrets.get("SUPABASE_ANON_KEY")})
    except Exception:
        pass
    sources.append({"url": os.environ.get("SUPABASE_URL"), "anon_key": os.environ.get("SUPABASE_ANON_KEY")})
    for source in sources:
        if not isinstance(source, dict):
            continue
        url, key = str(source.get("url") or "").strip(), str(source.get("anon_key") or "").strip()
        if not url or not key or any(x in (url + key).lower() for x in ("your_project", "placeholder", "replace_me")):
            continue
        parsed = urlsplit(url)
        if parsed.scheme in ("https", "http") and parsed.netloc:
            return url.rstrip("/"), key, None
    return None, None, f"Supabase is not configured. Add [supabase].url and [supabase].anon_key to {local}."


def get_supabase_client():
    if st.session_state.get("supabase_client") is not None:
        return st.session_state["supabase_client"], None
    url, key, err = load_supabase_credentials()
    if err:
        return None, err
    try:
        from supabase import create_client
        client = create_client(url, key)
        st.session_state["supabase_client"] = client
        return client, None
    except ImportError:
        return None, "The Supabase package is missing. Run: python -m pip install -r requirements.txt"
    except Exception:
        return None, "Could not initialise Supabase. Check the project URL and publishable/anon key."


def current_user_id(client):
    expected = st.session_state.get("supabase_user_id")
    if not expected:
        return None
    try:
        session = client.auth.get_session()
        actual = str(getattr(getattr(session, "user", None), "id", ""))
        return actual if actual == str(expected) else None
    except Exception:
        return None


def get_display_names(table):
    return {str(k): str(v) for k, v in table["display_name"].to_dict().items()}


def food_table_key(food_class: str, table: pd.DataFrame):
    key = CLASS_TO_TABLE.get(normalize_label(food_class), normalize_label(food_class))
    return key if key in table.index else None


def food_display(food_class: str, table: pd.DataFrame) -> str:
    key = food_table_key(food_class, table)
    return str(table.loc[key, "display_name"]) if key else str(food_class).replace("_", " ").title()


def default_meal_category(at: datetime | None = None) -> str:
    h = (at or datetime.now().astimezone()).hour
    if 5 <= h < 11:
        return "Breakfast"
    if 11 <= h < 16:
        return "Lunch"
    if 16 <= h < 18:
        return "Snacks"
    return "Dinner"


def local_day_bounds(day: date):
    tz = datetime.now().astimezone().tzinfo
    start = datetime.combine(day, time.min).replace(tzinfo=tz)
    end = datetime.combine(day + timedelta(days=1), time.min).replace(tzinfo=tz)
    return start.isoformat(), end.isoformat()


def summarize(logs):
    fields = {"calories": "calories_kcal", "protein": "protein_g", "carbs": "carbs_g", "fat": "fat_g"}
    return {k: sum(float(r.get(v) or 0) for r in logs) for k, v in fields.items()}


def fetch_logs(client, user_id, start_iso=None, end_iso=None):
    query = client.table("food_logs").select("*").eq("user_id", user_id)
    if start_iso:
        query = query.gte("consumed_at", start_iso)
    if end_iso:
        query = query.lt("consumed_at", end_iso)
    response = query.order("consumed_at", desc=True).execute()
    return response.data or []


def fetch_day(client, user_id, day):
    start, end = local_day_bounds(day)
    return fetch_logs(client, user_id, start, end)


def estimate_adult_daily_target(profile):
    """Preserve the original adult-only Mifflin-St Jeor estimate; never calculate it for under-18 profiles."""
    required = ("age", "height_cm", "weight_kg", "gender", "activity_level", "fitness_goal")
    if not profile or any(profile.get(k) in (None, "") for k in required):
        return None
    try:
        age, height, weight = int(profile["age"]), float(profile["height_cm"]), float(profile["weight_kg"])
    except (TypeError, ValueError):
        return None
    if age < 18 or age > 120 or profile.get("gender") not in ("Female", "Male"):
        return None
    if not (50 <= height <= 250 and 20 <= weight <= 400):
        return None
    factor = ACTIVITY_FACTORS.get(profile.get("activity_level"))
    if not factor:
        return None
    constant = 5 if profile["gender"] == "Male" else -161
    base = (10 * weight + 6.25 * height - 5 * age + constant) * factor
    adjustment = {"Lose weight": -300, "Gain weight": 250, "Build muscle": 250}.get(profile.get("fitness_goal"), 0)
    value = base + adjustment
    return int(round(value / 10) * 10) if value > 0 else None


def safe_error(action, exc):
    # Avoid dumping request payloads or credentials into the UI.
    st.error(f"{action} failed ({type(exc).__name__}). Check your Supabase table columns and row-level security policies.")


def nutrition_from_log(row, new_grams, table):
    """Recalculate from the nutrition table when possible; otherwise scale the saved estimate."""
    old_grams = float(row.get("portion_grams") or 0)
    food_name = str(row.get("food_name") or "")
    key = None
    for k, r in table.iterrows():
        if str(r["display_name"]).strip().casefold() == food_name.strip().casefold():
            key = k
            break
    if key:
        est = estimate_nutrition(table, key, new_grams)
        return {"calories_kcal": est["calories"], "protein_g": est["protein"], "carbs_g": est["carbs"], "fat_g": est["fat"]}
    if old_grams <= 0:
        raise ValueError("The original portion size is unavailable, so this entry cannot be recalculated safely.")
    ratio = float(new_grams) / old_grams
    return {c: float(row.get(c) or 0) * ratio for c in ("calories_kcal", "protein_g", "carbs_g", "fat_g")}


def insert_food_log(client, user_id, food_name, grams, estimate, consumed_at, category):
    values = {"user_id": user_id, "food_name": food_name, "portion_grams": float(grams),
              "calories_kcal": float(estimate["calories"]), "protein_g": float(estimate["protein"]),
              "carbs_g": float(estimate["carbs"]), "fat_g": float(estimate["fat"]),
              "consumed_at": consumed_at, "meal_category": category}
    try:
        client.table("food_logs").insert(values).execute()
        return True, None
    except Exception as exc:
        # Compatibility with an older schema: log without category, but make migration requirement visible.
        msg = str(exc).lower()
        if "meal_category" in msg or "column" in msg:
            values.pop("meal_category", None)
            try:
                client.table("food_logs").insert(values).execute()
                return True, "Meal category was not saved because the Supabase migration has not been applied."
            except Exception as retry_exc:
                return False, retry_exc
        return False, exc


def render_auth_and_profile():
    st.markdown('<div class="fb-eyebrow">Your account</div>', unsafe_allow_html=True)
    st.header("Account & profile")
    client, err = get_supabase_client()
    if not client:
        st.info(err or "Connect Supabase to use personal accounts and cloud diary storage.")
        return
    user_id = current_user_id(client)
    if not user_id:
        login, signup = st.tabs(["Log in", "Create account"])
        with login:
            with st.form("login_form"):
                email = st.text_input("Email address")
                password = st.text_input("Password", type="password")
                go = st.form_submit_button("Log in", type="primary", use_container_width=True)
            if go:
                try:
                    result = client.auth.sign_in_with_password({"email": email.strip(), "password": password})
                    st.session_state["supabase_user_id"] = str(result.user.id)
                    st.session_state["login_email"] = email.strip()
                    st.session_state["profile_name_fallback"] = (result.user.user_metadata or {}).get("display_name", "")
                    st.rerun()
                except Exception as exc:
                    safe_error("Login", exc)
        with signup:
            with st.form("signup_form"):
                display_name = st.text_input("Display name", max_chars=100)
                email2 = st.text_input("Email address", key="signup_email")
                password2 = st.text_input("Password", type="password", key="signup_password")
                go2 = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if go2:
                if not display_name.strip() or not email2.strip() or not password2:
                    st.warning("Enter a display name, email address and password.")
                else:
                    try:
                        result = client.auth.sign_up({"email": email2.strip(), "password": password2,
                            "options": {"data": {"display_name": display_name.strip()}}})
                        if result.session and result.user:
                            st.session_state["supabase_user_id"] = str(result.user.id)
                            st.session_state["login_email"] = email2.strip()
                            st.session_state["profile_name_fallback"] = display_name.strip()
                            st.rerun()
                        st.success("Account created. Confirm your email, then log in.")
                    except Exception as exc:
                        safe_error("Account creation", exc)
        return

    try:
        profile_resp = client.table("profiles").select("*").eq("user_id", user_id).limit(1).execute()
        profile = profile_resp.data[0] if profile_resp.data else {}
    except Exception as exc:
        safe_error("Loading profile", exc)
        return
    col_a, col_b = st.columns([3, 1])
    col_a.caption(f"Signed in · {st.session_state.get('login_email', '') or 'Your private FitBite account'}")
    if col_b.button("Log out", use_container_width=True):
        try:
            client.auth.sign_out()
        finally:
            for k in ("supabase_user_id", "supabase_client", "profile_name_fallback", "login_email", "image_bytes", "predictions"):
                st.session_state.pop(k, None)
            st.rerun()

    with st.form("profile_form"):
        st.subheader("Personal details")
        name = st.text_input("Name / display name", value=profile.get("display_name") or st.session_state.get("profile_name_fallback", ""))
        c1, c2 = st.columns(2)
        age_value = int(profile["age"]) if profile.get("age") is not None else None
        age = c1.number_input("Age", min_value=13, max_value=120, value=age_value, step=1, placeholder="Enter age")
        gender_default = profile.get("gender") if profile.get("gender") in GENDERS else GENDERS[0]
        gender = c2.selectbox("Gender", GENDERS, index=GENDERS.index(gender_default))
        height_value = float(profile["height_cm"]) if profile.get("height_cm") is not None else None
        height = c1.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=height_value, step=0.5, placeholder="Enter height")
        weight_value = float(profile["weight_kg"]) if profile.get("weight_kg") is not None else None
        weight = c2.number_input("Weight (kg)", min_value=20.0, max_value=400.0, value=weight_value, step=0.5, placeholder="Enter weight")
        goal_default = profile.get("fitness_goal") if profile.get("fitness_goal") in FITNESS_GOALS else FITNESS_GOALS[0]
        goal = st.selectbox("Wellness goal", FITNESS_GOALS, index=FITNESS_GOALS.index(goal_default))
        activity_default = profile.get("activity_level") if profile.get("activity_level") in ACTIVITY_LEVELS else ACTIVITY_LEVELS[0]
        activity = st.selectbox("Activity level", ACTIVITY_LEVELS, index=ACTIVITY_LEVELS.index(activity_default))
        save = st.form_submit_button("Save profile", type="primary")
    if save:
        missing = []
        if not name.strip(): missing.append("name")
        if age is None: missing.append("age")
        if gender == GENDERS[0]: missing.append("gender")
        if height is None: missing.append("height")
        if weight is None: missing.append("weight")
        if goal == FITNESS_GOALS[0]: missing.append("fitness goal")
        if missing:
            st.warning("Complete these profile fields before saving: " + ", ".join(missing) + ".")
            return
        payload = {"user_id": user_id, "display_name": name.strip(), "age": int(age), "gender": gender,
                   "height_cm": float(height), "weight_kg": float(weight), "fitness_goal": goal,
                   "activity_level": activity}
        try:
            client.table("profiles").upsert(payload, on_conflict="user_id").execute()
            st.success("Profile saved.")
        except Exception as exc:
            if "activity_level" in str(exc).lower() or "column" in str(exc).lower():
                payload.pop("activity_level", None)
                try:
                    client.table("profiles").upsert(payload, on_conflict="user_id").execute()
                    st.warning("Profile saved without activity level because that optional column is not present in your current profiles table.")
                except Exception as retry_exc:
                    safe_error("Saving profile", retry_exc)
            else:
                safe_error("Saving profile", exc)


def render_add_food(table, class_names, client, user_id, predicted_class=None):
    st.markdown('<div class="fb-eyebrow">Add a meal</div>', unsafe_allow_html=True)
    st.subheader("Manual food entry")
    if not class_names:
        st.warning("The trained model class list is missing, so manual food suggestions are unavailable.")
        return
    labels = {c: food_display(c, table) for c in class_names}
    default = predicted_class if predicted_class in class_names else class_names[0]
    chosen = st.selectbox("Search the 15 recognised food classes", class_names,
                          index=class_names.index(default), format_func=lambda x: labels.get(x, x), key="manual_food_class")
    table_key = food_table_key(chosen, table)
    if not table_key:
        st.warning(f"A verified nutrition-table row for **{labels.get(chosen, chosen)}** is not present in the supplied nutrition database. I have not substituted a different dish or invented values, so this food cannot yet be logged with calculated nutrients.")
        return
    row = table.loc[table_key]
    c1, c2 = st.columns(2)
    grams = c1.number_input("Portion size (grams)", min_value=10, max_value=1500,
                            value=int(row["serving_grams"]), step=10, key="manual_grams")
    category = c2.selectbox("Meal category", MEAL_CATEGORIES, index=MEAL_CATEGORIES.index(default_meal_category()), key="manual_category")
    consumed_date = st.date_input("Date", value=datetime.now().astimezone().date(), key="manual_date")
    consumed_time = st.time_input("Time", value=datetime.now().astimezone().time().replace(second=0, microsecond=0), key="manual_time")
    est = estimate_nutrition(table, table_key, grams)
    st.caption(f"Estimated for {grams} g · {est['calories']:.0f} kcal · P {est['protein']:.1f} g · C {est['carbs']:.1f} g · F {est['fat']:.1f} g")
    if st.button("Add to food diary", type="primary", use_container_width=True, key="manual_add"):
        if not client or not user_id:
            st.info("Log in from Account & profile to save food to your personal diary.")
            return
        at = datetime.combine(consumed_date, consumed_time).replace(tzinfo=datetime.now().astimezone().tzinfo).isoformat()
        ok, warning = insert_food_log(client, user_id, labels.get(chosen, str(row["display_name"])), grams, est, at, category)
        if ok:
            st.success("Food added to your diary.")
            if warning:
                st.warning(warning)
            st.rerun()
        else:
            safe_error("Adding food", warning)


def render_food_editor(row, client, user_id, table):
    rid = str(row.get("id"))
    with st.expander("Edit entry", expanded=False):
        old_cat = row.get("meal_category") if row.get("meal_category") in MEAL_CATEGORIES else "Snacks"
        with st.form(f"edit_food_{rid}"):
            grams = st.number_input("Portion (g)", min_value=1, max_value=3000,
                                    value=max(1, int(float(row.get("portion_grams") or 100))), step=5)
            category = st.selectbox("Meal category", MEAL_CATEGORIES, index=MEAL_CATEGORIES.index(old_cat))
            save = st.form_submit_button("Save changes", type="primary")
        if save:
            try:
                nutrients = nutrition_from_log(row, grams, table)
                payload = {"portion_grams": float(grams), **nutrients, "meal_category": category}
                try:
                    client.table("food_logs").update(payload).eq("id", rid).eq("user_id", user_id).execute()
                except Exception as exc:
                    if "meal_category" in str(exc).lower() or "column" in str(exc).lower():
                        payload.pop("meal_category", None)
                        client.table("food_logs").update(payload).eq("id", rid).eq("user_id", user_id).execute()
                        st.warning("Portion updated; meal category needs the supplied Supabase migration.")
                    else:
                        raise
                st.success("Diary entry updated and nutrients recalculated.")
                st.rerun()
            except Exception as exc:
                safe_error("Updating diary entry", exc)
        if st.button("Delete entry", key=f"delete_{rid}"):
            try:
                client.table("food_logs").delete().eq("id", rid).eq("user_id", user_id).execute()
                st.success("Diary entry deleted.")
                st.rerun()
            except Exception as exc:
                safe_error("Deleting diary entry", exc)


def render_diary(table, client, user_id, selected_day=None):
    st.markdown('<div class="fb-eyebrow">Your food diary</div>', unsafe_allow_html=True)
    st.header("Food diary")
    chosen_day = st.date_input("Select a day", value=selected_day or datetime.now().astimezone().date(), key="diary_day")
    if not client or not user_id:
        st.info("Log in from Account & profile to view and manage your private food diary.")
        return
    try:
        logs = fetch_day(client, user_id, chosen_day)
    except Exception as exc:
        safe_error("Loading food diary", exc)
        return
    if not logs:
        st.markdown('<div class="fb-hero"><b>No meals logged for this date yet.</b><br><span class="fb-muted">Use Analyze Food or Manual food entry to add a meal. Nothing is pre-filled with sample data.</span></div>', unsafe_allow_html=True)
        return
    totals = summarize(logs)
    try:
        profile_response = client.table("profiles").select("*").eq("user_id", user_id).limit(1).execute()
        profile = profile_response.data[0] if profile_response.data else {}
        target = estimate_adult_daily_target(profile)
        if target is not None:
            st.caption(f"Adult-only daily energy estimate from your saved profile: {target:,} kcal. This is an approximate reference, not a limit.")
        elif profile.get("age") is not None and int(profile.get("age") or 0) < 18:
            st.caption("Daily calorie targets are not displayed for profiles under 18. Your diary and nutrition estimates remain available.")
    except Exception:
        pass
    cols = st.columns(4)
    for col, (label, value, unit) in zip(cols, [("Energy", totals["calories"], "kcal"), ("Protein", totals["protein"], "g"), ("Carbs", totals["carbs"], "g"), ("Fat", totals["fat"], "g")]):
        col.metric(label, f"{value:.0f} {unit}" if unit == "kcal" else f"{value:.1f} {unit}")
    grouped = defaultdict(list)
    for r in logs:
        cat = r.get("meal_category") if r.get("meal_category") in MEAL_CATEGORIES else "Snacks"
        grouped[cat].append(r)
    for cat in MEAL_CATEGORIES:
        st.subheader(cat)
        entries = grouped.get(cat, [])
        if not entries:
            st.caption("No entries in this category.")
            continue
        for row in entries:
            c1, c2, c3 = st.columns([3, 2, 1])
            c1.markdown(f"**{row.get('food_name', 'Food')}**")
            c1.caption(f"{float(row.get('portion_grams') or 0):.0f} g")
            c2.markdown(f"**{float(row.get('calories_kcal') or 0):.0f} kcal**")
            c2.caption(f"P {float(row.get('protein_g') or 0):.1f} g · C {float(row.get('carbs_g') or 0):.1f} g · F {float(row.get('fat_g') or 0):.1f} g")
            try:
                local_time = datetime.fromisoformat(str(row.get("consumed_at", "")).replace("Z", "+00:00")).astimezone().strftime("%I:%M %p")
            except Exception:
                local_time = ""
            c3.caption(local_time)
            render_food_editor(row, client, user_id, table)


def render_dashboard(table, client, user_id):
    st.markdown('<div class="fb-eyebrow">FitBite · Your daily companion</div>', unsafe_allow_html=True)
    profile = {}
    logs = []
    if client and user_id:
        try:
            p = client.table("profiles").select("*").eq("user_id", user_id).limit(1).execute()
            profile = p.data[0] if p.data else {}
            logs = fetch_day(client, user_id, datetime.now().astimezone().date())
        except Exception as exc:
            safe_error("Loading dashboard", exc)
    name = profile.get("display_name") or "there"
    hour = datetime.now().astimezone().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    st.title(f"{greeting}, {name} 👋")
    st.caption(datetime.now().astimezone().strftime("%A, %B %d") + " · A snapshot of the foods you have logged today")
    if not client or not user_id:
        st.markdown('<div class="fb-hero"><h3>Your food story, in one place</h3><p>Sign in to sync your meals, see your real daily totals and review weekly patterns. You can explore food recognition before signing in.</p></div>', unsafe_allow_html=True)
        st.info("Your dashboard will populate from your own Supabase food logs after you sign in. No sample numbers are shown.")
        return
    totals = summarize(logs)
    calories = totals["calories"]
    st.markdown('<div class="fb-hero"><span class="fb-pill">TODAY · LIVE FROM YOUR DIARY</span><h2 style="margin:.7rem 0 .2rem">Your day at a glance</h2><p class="fb-muted">A neutral summary of logged meals and estimated nutrients. These values are informational, not targets or limits.</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Logged energy", f"{calories:.0f} kcal", help="Estimated total from foods you logged today.")
    c2.metric("Meals logged", str(len(logs)))
    c3.metric("Protein (est.)", f"{totals['protein']:.1f} g")
    c4.metric("Carbs (est.)", f"{totals['carbs']:.1f} g")
    adult_target = estimate_adult_daily_target(profile)
    if adult_target is not None:
        st.caption(f"Adult-only daily energy estimate from your saved profile: {adult_target:,} kcal. This is an approximate reference, not a limit.")
    elif profile.get("age") is not None and int(profile.get("age") or 0) < 18:
        st.caption("Daily calorie targets are not displayed for profiles under 18. Your food diary and nutrition estimates remain available.")
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Today's meals")
        if logs:
            for r in logs[:5]:
                st.markdown(f"**{r.get('food_name','Food')}** · {float(r.get('portion_grams') or 0):.0f} g · {float(r.get('calories_kcal') or 0):.0f} kcal")
                st.caption(r.get("meal_category") or "Meal category not set")
        else:
            st.info("No meals logged today yet. Add a meal to see your dashboard update.")
    with right:
        st.subheader("Macro overview")
        macro = pd.DataFrame({"Nutrient": ["Protein", "Carbohydrates", "Fat"], "Grams": [totals["protein"], totals["carbs"], totals["fat"]]})
        if sum(macro["Grams"]) > 0:
            fig = px.pie(macro, names="Nutrient", values="Grams", hole=.62,
                         color="Nutrient", color_discrete_map={"Protein":"#087443", "Carbohydrates":"#e9a36d", "Fat":"#9b91c8"})
            fig.update_layout(showlegend=True, margin=dict(l=8,r=8,t=20,b=8), paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Macro chart appears after you log food with nutrition estimates.")
    st.divider()
    a, b = st.columns(2)
    if a.button("＋  Add food manually", type="primary", use_container_width=True):
        st.session_state["page"] = "Analyze Food"
        st.session_state["show_manual"] = True
        st.rerun()
    if b.button("Open food diary", use_container_width=True):
        st.session_state["page"] = "Food Diary"
        st.rerun()


@st.cache_data(show_spinner="Recognising food…")
def predict_image(image_bytes):
    model, names_or_error = get_model()
    if model is None:
        raise RuntimeError(names_or_error or "Trained model is unavailable.")
    names = names_or_error
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    array = np.asarray(image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR), dtype=np.float32)[None]
    probabilities = model.predict(array, verbose=0)[0]
    order = np.argsort(probabilities)[::-1][:3]
    return [(str(names[int(i)]), float(probabilities[int(i)])) for i in order]


def render_analyze(table, class_names, client, user_id):
    st.markdown('<div class="fb-eyebrow">Analyze food</div>', unsafe_allow_html=True)
    st.header("What's on your plate?")
    st.caption("Upload a food photo for a model prediction, or enter one of the supported foods manually.")
    tab_photo, tab_manual = st.tabs(["Scan a food photo", "Manual entry"])
    with tab_photo:
        uploaded = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png", "webp"], key="food_upload")
        if uploaded:
            data = uploaded.getvalue()
            st.session_state["image_bytes"] = data
            st.session_state["image_name"] = uploaded.name
            col1, col2 = st.columns([1, 1])
            col1.image(Image.open(io.BytesIO(data)), caption=uploaded.name, use_container_width=True)
            try:
                predictions = predict_image(data)
                st.session_state["predictions"] = predictions
                st.session_state["selected_prediction"] = predictions[0][0]
                best, confidence = predictions[0]
                col2.markdown(f"<div class='fb-pill'>MODEL PREDICTION</div><h2>{food_display(best, table)}</h2>", unsafe_allow_html=True)
                col2.metric("Confidence score", f"{confidence:.1%}")
                col2.caption("This is the model score for this image, not the model's overall test accuracy.")
                if confidence < LOW_CONFIDENCE:
                    col2.warning("The model is uncertain. Review the result before logging it.")
                if st.button("Review recognition result", type="primary", use_container_width=True):
                    st.session_state["page"] = "Recognition Results"
                    st.rerun()
            except Exception as exc:
                st.error(f"Recognition is unavailable: {type(exc).__name__}. Check that the trained model file and TensorFlow installation are present.")
    with tab_manual:
        if st.session_state.get("show_manual"):
            st.info("Manual entry does not run the image model.")
        render_add_food(table, class_names, client, user_id, st.session_state.get("selected_prediction"))


def render_results(table, client, user_id):
    st.markdown('<div class="fb-eyebrow">Recognition results</div>', unsafe_allow_html=True)
    st.header("Review your prediction")
    predictions = st.session_state.get("predictions")
    if not predictions:
        st.info("No recent recognition result. Upload a food image in Analyze Food first.")
        return
    if st.session_state.get("image_bytes"):
        st.image(Image.open(io.BytesIO(st.session_state["image_bytes"])), caption="Your uploaded image", width=360)
    best, conf = predictions[0]
    names = [x[0] for x in predictions]
    chosen = st.selectbox("Confirm or correct the food", names + ["Choose another supported class"],
                          format_func=lambda x: food_display(x, table) if x in names else x)
    if chosen == "Choose another supported class":
        all_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8")) if CLASS_NAMES_PATH.exists() else names
        chosen = st.selectbox("Select food class", all_names, format_func=lambda x: food_display(x, table), key="manual_corrected_class")
    else:
        st.session_state["selected_prediction"] = chosen
    selected_score = dict(predictions).get(chosen)
    if selected_score is not None:
        st.metric("Selected prediction confidence", f"{selected_score:.1%}")
    else:
        st.caption("The selected correction was outside the model's top three, so no confidence score is shown for it.")
    st.subheader("Alternative predictions")
    frame = pd.DataFrame({"Food": [food_display(n, table) for n, _ in predictions], "Confidence": [p for _, p in predictions]})
    fig = px.bar(frame.sort_values("Confidence"), x="Confidence", y="Food", orientation="h", range_x=[0,1], text_auto=".1%",
                 color_discrete_sequence=["#087443"])
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=5,r=5,t=20,b=5))
    st.plotly_chart(fig, use_container_width=True)
    key = food_table_key(chosen, table)
    if not key:
        st.warning(f"No verified nutrition row exists for **{food_display(chosen, table)}** in the supplied CSV. The image model can still classify it, but nutrition logging is unavailable until a reviewed row is added.")
        return
    row = table.loc[key]
    c1, c2 = st.columns(2)
    grams = c1.number_input("Portion size (grams)", 10, 1500, int(row["serving_grams"]), 10, key="result_grams")
    category = c2.selectbox("Meal category", MEAL_CATEGORIES, index=MEAL_CATEGORIES.index(default_meal_category()), key="result_category")
    day = st.date_input("Consumed on", value=datetime.now().astimezone().date(), key="result_day")
    at_time = st.time_input("Time", value=datetime.now().astimezone().time().replace(second=0,microsecond=0), key="result_time")
    est = estimate_nutrition(table, key, grams)
    st.markdown(f"**Estimated nutrition for {grams} g** · {est['calories']:.0f} kcal · Protein {est['protein']:.1f} g · Carbs {est['carbs']:.1f} g · Fat {est['fat']:.1f} g")
    if st.button("Log this food", type="primary", use_container_width=True):
        if not client or not user_id:
            st.info("Log in from Account & profile to save this result to your diary.")
        else:
            consumed_at = datetime.combine(day, at_time).replace(tzinfo=datetime.now().astimezone().tzinfo).isoformat()
            ok, err = insert_food_log(client, user_id, food_display(chosen, table), grams, est, consumed_at, category)
            if ok:
                st.success("Food saved to your diary.")
                if err: st.warning(err)
                st.rerun()
            else:
                safe_error("Saving food log", err)


def render_nutrition_estimator(table, client, user_id, class_names):
    st.markdown('<div class="fb-eyebrow">Nutrition estimator</div>', unsafe_allow_html=True)
    st.header("Estimate a portion")
    options = list(table.index)
    display = get_display_names(table)
    chosen = st.selectbox("Food", options, format_func=lambda k: display[k], key="estimator_food")
    row = table.loc[chosen]
    st.caption(f"Reference serving: {row['serving_description']} · approximately {row['serving_grams']:.0f} g")
    a,b = st.columns([1,2])
    grams = a.number_input("Portion (g)", 10, 1500, int(row["serving_grams"]), 10, key="estimator_grams")
    desc = b.text_input("Dish description (optional)", placeholder="e.g. extra oil, butter, cream", key="estimator_desc")
    labels = {m["id"]:m["label"] for m in MODIFIERS}
    mods = st.multiselect("Ingredient adjustments (review and edit)", list(labels), default=detect_modifiers(desc), format_func=labels.get)
    strength = st.slider("Adjustment strength", .5, 2.0, 1.0, .25)
    est = estimate_nutrition(table, chosen, grams, mods, strength)
    cols = st.columns(3)
    for col, (label,key,unit) in zip(cols, [("Energy","calories","kcal"),("Protein","protein","g"),("Carbohydrates","carbs","g"),("Fat","fat","g"),("Fibre","fiber","g"),("Sugar","sugar","g")]):
        col.metric(label, f"{est[key]:.0f} {unit}" if unit=="kcal" else f"{est[key]:.1f} {unit}")
    if est["applied"]:
        st.caption("Applied assumptions: " + ", ".join(x["label"] for x in est["applied"]) + ". These are approximate adjustments, not measurements.")
    st.warning("The model and nutrition table cannot measure the actual portion, ingredients or cooking method. All nutrition values are estimates.")
    category = st.selectbox("Meal category for this entry", MEAL_CATEGORIES, index=MEAL_CATEGORIES.index(default_meal_category()), key="estimator_category")
    if st.button("Add this estimate to diary", type="primary"):
        if not client or not user_id:
            st.info("Log in from Account & profile to save this estimate.")
        else:
            at = datetime.now().astimezone().isoformat()
            ok, err = insert_food_log(client, user_id, str(row["display_name"]), grams, est, at, category)
            if ok:
                st.success("Estimate added to diary.")
                if err: st.warning(err)
                st.rerun()
            else: safe_error("Saving estimate", err)


def render_insights(client, user_id):
    st.markdown('<div class="fb-eyebrow">Your patterns</div>', unsafe_allow_html=True)
    st.header("Nutrition insights")
    st.caption("Seven-day summaries are calculated from your own saved food diary entries. No sample or generated data is used.")
    if not client or not user_id:
        st.info("Log in to see weekly insights from your personal food logs.")
        return
    today = datetime.now().astimezone().date()
    first = today - timedelta(days=6)
    start_iso, end_iso = local_day_bounds(first)[0], local_day_bounds(today + timedelta(days=1))[0]
    try:
        logs = fetch_logs(client, user_id, start_iso, end_iso)
    except Exception as exc:
        safe_error("Loading weekly insights", exc)
        return
    if not logs:
        st.markdown('<div class="fb-hero"><b>Your weekly insights will appear here.</b><br><span class="fb-muted">Log meals on different days to build a real trend. No example data is substituted.</span></div>', unsafe_allow_html=True)
        return
    daily = {first + timedelta(days=i): {"calories":0.0,"protein":0.0,"carbs":0.0,"fat":0.0,"entries":0} for i in range(7)}
    for row in logs:
        try:
            d = datetime.fromisoformat(str(row.get("consumed_at", "")).replace("Z", "+00:00")).astimezone().date()
        except Exception:
            continue
        if d in daily:
            daily[d]["calories"] += float(row.get("calories_kcal") or 0)
            daily[d]["protein"] += float(row.get("protein_g") or 0)
            daily[d]["carbs"] += float(row.get("carbs_g") or 0)
            daily[d]["fat"] += float(row.get("fat_g") or 0)
            daily[d]["entries"] += 1
    frame = pd.DataFrame([{"Date":d, **v} for d,v in daily.items()])
    active = frame[frame["entries"] > 0]
    totals = summarize(logs)
    a,b,c,d = st.columns(4)
    a.metric("Logged days", f"{len(active)} / 7")
    b.metric("Average energy on logged days", f"{active['calories'].mean():.0f} kcal" if len(active) else "—")
    c.metric("Weekly protein total", f"{totals['protein']:.1f} g")
    d.metric("Meals recorded", str(len(logs)))
    fig = px.bar(frame, x="Date", y="calories", title="Daily logged energy · last 7 days", labels={"calories":"Estimated kcal"},
                 color_discrete_sequence=["#087443"])
    fig.update_layout(paper_bgcolor="white",plot_bgcolor="white",margin=dict(l=8,r=8,t=48,b=8))
    st.plotly_chart(fig, use_container_width=True)
    macro_frame = frame.melt(id_vars="Date", value_vars=["protein","carbs","fat"], var_name="Macro", value_name="Grams")
    macro_fig = px.line(macro_frame, x="Date", y="Grams", color="Macro", markers=True, title="Daily macro trends",
                        color_discrete_map={"protein":"#087443","carbs":"#e9a36d","fat":"#9b91c8"})
    macro_fig.update_layout(paper_bgcolor="white",plot_bgcolor="white",margin=dict(l=8,r=8,t=48,b=8))
    st.plotly_chart(macro_fig, use_container_width=True)
    st.caption("Days without saved entries appear as zero in the daily chart; the average above uses days with at least one logged meal.")


def render_model_performance(metadata):
    st.markdown('<div class="fb-eyebrow">Project evaluation</div>', unsafe_allow_html=True)
    st.header("Model performance")
    models = metadata.get("models", {})
    if not models:
        st.warning("Evaluation metadata is missing. Run the training and evaluation notebook to regenerate it.")
        return
    rows=[]
    for key,v in models.items():
        rows.append({"Model":"Custom CNN" if key=="custom_cnn" else "MobileNetV2", "Test accuracy":v.get("test_accuracy"), "Validation accuracy":v.get("val_accuracy"), "Macro F1":v.get("macro_f1"), "Test loss":v.get("test_loss"), "Top-3 accuracy":v.get("top3_accuracy")})
    df=pd.DataFrame(rows)
    st.dataframe(df.style.format({c:"{:.1%}" for c in ["Test accuracy","Validation accuracy","Macro F1","Top-3 accuracy"]}.copy() | {"Test loss":"{:.3f}"}), hide_index=True, use_container_width=True)
    chart=df.melt(id_vars="Model",value_vars=["Test accuracy","Validation accuracy","Macro F1"],var_name="Metric",value_name="Score")
    fig=px.bar(chart,x="Model",y="Score",color="Metric",barmode="group",range_y=[0,1],text_auto=".1%")
    st.plotly_chart(fig,use_container_width=True)
    st.success(f"Best model recorded in metadata: {metadata.get('best_model','not specified')}")
    reports=HERE/"reports"
    for fn,title in [("model_comparison.png","Model comparison"),("validation_comparison.png","Validation curves"),("confusion_mobilenetv2_counts.png","MobileNetV2 confusion matrix"),("confusion_custom_cnn_counts.png","Custom CNN confusion matrix")]:
        if (reports/fn).exists():
            st.subheader(title)
            st.image(str(reports/fn),use_container_width=True)


def render_about(class_names, table):
    st.markdown('<div class="fb-eyebrow">About FitBite</div>', unsafe_allow_html=True)
    st.header("How FitBite works")
    st.markdown("""
FitBite combines a trained Indian-food image classifier with a local nutrition reference table and a personal food diary.

1. **Recognise:** MobileNetV2 predicts a food category from an uploaded image.
2. **Review:** You confirm or correct the prediction and choose a portion.
3. **Estimate:** The app scales reference nutrition values and clearly labels them as estimates.
4. **Track:** Confirmed entries are stored in your own Supabase account and used for diary and weekly summaries.

**Limitations:** the classifier does not measure serving size, ingredients or calories. The supplied nutrition CSV does not currently cover every model class; unavailable entries are shown rather than guessed. The app is an educational tracker, not medical or dietetic advice.
""")
    st.write(f"Model classes: {len(class_names or [])} · Nutrition table entries: {len(table)}")


def main():
    try:
        table = get_table()
    except Exception as exc:
        st.error(f"Could not load nutrition_table.csv: {type(exc).__name__}")
        st.stop()
    try:
        class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8")) if CLASS_NAMES_PATH.exists() else []
    except Exception:
        class_names=[]
    metadata=get_metadata()
    client, _ = get_supabase_client()
    user_id=current_user_id(client) if client else None

    with st.sidebar:
        st.markdown("# 🥗 FitBite")
        st.caption("Food recognition · Nutrition diary")
        pages=["Dashboard","Analyze Food","Recognition Results","Food Diary","Nutrition Insights","Nutrition Estimator","Model Performance","About Project","Account & Profile"]
        current=st.session_state.get("page","Dashboard")
        if current not in pages: current="Dashboard"
        page=st.radio("Navigate",pages,index=pages.index(current),label_visibility="collapsed")
        st.session_state["page"]=page
        st.divider()
        st.caption("Nutrition values are estimates. No food is labelled as good or bad.")
        if user_id:
            st.success("Signed in")
        else:
            st.caption("Not signed in · diary sync unavailable")

    if page=="Dashboard": render_dashboard(table,client,user_id)
    elif page=="Analyze Food": render_analyze(table,class_names,client,user_id)
    elif page=="Recognition Results": render_results(table,client,user_id)
    elif page=="Food Diary": render_diary(table,client,user_id)
    elif page=="Nutrition Insights": render_insights(client,user_id)
    elif page=="Nutrition Estimator": render_nutrition_estimator(table,client,user_id,class_names)
    elif page=="Model Performance": render_model_performance(metadata)
    elif page=="About Project": render_about(class_names,table)
    elif page=="Account & Profile": render_auth_and_profile()


if __name__ == "__main__":
    main()
