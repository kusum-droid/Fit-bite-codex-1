# Build and validation status

## Passed
- Python bytecode compilation for `app.py` and `nutrition_calc.py`.
- Dashboard startup smoke test with a lightweight Streamlit stub (checks page bootstrap and no-login empty state; not a substitute for a browser run).
- Nutrition calculation unit tests (3/3 passed).
- Food-class mapping audit (15 model labels checked; 9 have a direct/explicitly mapped CSV row, 6 remain unsupported).
- Under-18 calorie-target guard check (17-year-old profile returns no target; adult profile can return the existing estimate).
- Portion recalculation helper test.
- Trained model archive integrity check; `best_model.keras` SHA-256 matches the source artifact's `streamlit_app/models/best_model.keras` exactly.
- Secret-file exclusion check; only `.streamlit/secrets.toml.example` is included.

## Not run in this build environment
- A real Streamlit browser session, because Streamlit is not installed in the execution environment and package installation could not reach the package index.
- TensorFlow model inference, because TensorFlow is not installed in the execution environment.
- Live Supabase authentication, RLS, inserts, updates, deletes or the SQL migration, because no Supabase credentials/database connection were supplied. The migration is provided for you to review and run.

The app is therefore packaged with syntax and helper-level checks passed, but the live model/database integration still needs a local run with the dependencies and your Supabase project.
