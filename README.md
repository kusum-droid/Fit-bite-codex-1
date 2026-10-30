# FitBite — updated Streamlit application

This project uses the separately supplied **latest `app_updated.py`** as its application foundation. The older trained-artifact ZIP was used only to recover supporting files: the trained MobileNetV2 model, class labels, model metadata, nutrition calculation module, nutrition CSV and evaluation reports. The older ZIP's `streamlit_app/app.py` was deliberately not used.

## What is included

- `app.py` — updated multi-section Streamlit app.
- `nutrition_calc.py`, `nutrition_table.csv` — existing nutrition calculation logic and supplied food estimates.
- `best_model.keras`, `class_names.json`, `model_metadata.json` — trained model and metadata expected by the latest app.
- `reports/` — supplied evaluation charts and reports.
- `supabase_meal_category_migration.sql` — one additive migration for meal categories.
- `SUPABASE_SETUP.md` — Supabase setup notes.
- `backups/app_updated_original.py` — unchanged backup of the separately uploaded latest app.

## Implemented changes

1. Recreated the Stitch visual direction in Streamlit with a soft neutral background, green accents, rounded panels, metrics and responsive columns.
2. Split the experience into Dashboard, Analyze Food, Recognition Results, Food Diary and Nutrition Insights, while retaining Nutrition Estimator, Model Performance, About Project and Account/Profile.
3. Moved top-three model alternatives off the dashboard/analyze view and into Recognition Results.
4. Added searchable manual entry over the 15 model classes, with portion and meal-category controls where a nutrition-table row exists.
5. Added Breakfast, Lunch, Snacks and Dinner categories, diary editing, portion recalculation and deletion.
6. Added seven-day charts and summaries calculated only from the signed-in user's actual Supabase food logs.
7. Kept existing Supabase email/password authentication, profile fields, model evaluation and ingredient-adjustment estimator.

## Important nutrition-data limitation

The supplied model has 15 classes, but the supplied nutrition CSV does not contain a verified record for every class. This app maps only clear food-name equivalents and does not fill missing rows with guessed values. At present the supplied CSV supports direct nutrition estimates for these model classes:

- Chole Bhature
- Dhokla
- Dosa (mapped to Masala Dosa)
- Jalebi
- Kathi Roll
- Naan (mapped to Butter Naan)
- Pakora (mapped to Pakode)
- Pani Puri
- Pav Bhaji

The following model classes currently lack a directly matching row in the supplied CSV: **Biryani, Dabeli, Dal, Kofta, Paneer and Vada Pav**. They can still be predicted by the image model, but nutrition logging for them remains unavailable until verified nutrition entries are supplied and reviewed. See `nutrition_table.csv` before adding any records.

## Run in VS Code

1. Extract this project folder to a normal location on your computer.
2. Open the `FitBite_Updated` folder in VS Code.
3. Install Python 3.10–3.12 (recommended for broad TensorFlow wheel compatibility) and select that interpreter in VS Code.
4. Open the VS Code terminal in this folder and create a virtual environment:

   **Windows PowerShell**
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your own Supabase project URL and publishable/anon key. Never share or upload the real secrets file. Do not use a service-role key.
6. Review and run `supabase_meal_category_migration.sql` in Supabase SQL Editor once. See `SUPABASE_SETUP.md`.
7. Start the app:
   ```bash
   streamlit run app.py
   ```
8. Open the local URL shown in the terminal (normally `http://localhost:8501`).

## Testing notes

The project was checked for Python syntax, nutrition-table loading and class-to-nutrition coverage. A full live Streamlit + TensorFlow + Supabase end-to-end run requires installing the listed packages and using your own Supabase credentials; those credentials and remote database are intentionally not included in this package. The model binary was copied unchanged from the supplied artifact ZIP. No Supabase data was modified during packaging.
