# Supabase setup for FitBite

FitBite continues to use the existing Supabase Auth session and the existing `profiles` and `food_logs` tables. It does not create a second database or replace your existing tables.

## 1. Keep your existing tables and policies

The updated app expects:

- `profiles`: `user_id`, `display_name`, `age`, `gender`, `height_cm`, `weight_kg`, `fitness_goal`, `activity_level` (the activity field is read compatibly when absent, but saving the profile requires the column).
- `food_logs`: `id`, `user_id`, `food_name`, `portion_grams`, `calories_kcal`, `protein_g`, `carbs_g`, `fat_g`, `consumed_at`.
- Existing Supabase Auth email/password authentication.

Keep your current Row Level Security (RLS) policies. Every diary read, update and delete in the app is additionally filtered by the authenticated `user_id`. Confirm your RLS policies also restrict each user's rows to `auth.uid() = user_id`.

## 2. Add meal categories

The updated profile form uses `profiles.activity_level`, and the diary enhancement needs one new `food_logs.meal_category` column. In Supabase Dashboard → SQL Editor, review and run `supabase_meal_category_migration.sql` once. Existing rows are retained and assigned `Snacks` as a neutral initial category; you can edit their category in the app.

The app can still read and save logs on the old schema, but categories will not persist until this migration is applied.

## 3. Configure credentials locally

Create `.streamlit/secrets.toml` on your own machine (never commit or share this file):

```toml
[supabase]
url = "https://YOUR_PROJECT.supabase.co"
anon_key = "YOUR_SUPABASE_PUBLISHABLE_OR_ANON_KEY"
```

Use the publishable/anon key only. Do not put a service-role key in the app. The repository includes `.streamlit/secrets.toml.example` with placeholders only.
