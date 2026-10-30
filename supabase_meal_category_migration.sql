-- FitBite schema additions for the updated profile and food diary.
-- Run this once in Supabase SQL Editor. This does not change or replace RLS policies.

ALTER TABLE public.profiles
ADD COLUMN IF NOT EXISTS activity_level text;


ALTER TABLE public.food_logs
ADD COLUMN IF NOT EXISTS meal_category text NOT NULL DEFAULT 'Snacks';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'food_logs_meal_category_check'
          AND conrelid = 'public.food_logs'::regclass
    ) THEN
        ALTER TABLE public.food_logs
        ADD CONSTRAINT food_logs_meal_category_check
        CHECK (meal_category IN ('Breakfast', 'Lunch', 'Snacks', 'Dinner'));
    END IF;
END $$;

-- Existing rows are retained and receive the neutral default category 'Snacks'.
-- Users can reclassify older entries from the Food Diary page.
