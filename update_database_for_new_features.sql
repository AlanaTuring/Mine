-- Update Database for New Features
-- Run this in your Supabase SQL Editor

-- Add new columns to posts table
ALTER TABLE posts ADD COLUMN IF NOT EXISTS video_url text;
ALTER TABLE posts ADD COLUMN IF NOT EXISTS is_job_post boolean DEFAULT false;

-- Add new columns to profiles table for dynamic data
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS job_title text;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS current_location text;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS account_type text;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS organization_name text;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS accessibility_features jsonb;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS is_verified boolean DEFAULT false;

-- Update existing profiles to use new fields
UPDATE profiles 
SET 
  job_title = title,
  current_location = location
WHERE job_title IS NULL OR current_location IS NULL;

-- Create index for job posts
CREATE INDEX IF NOT EXISTS posts_is_job_post_idx ON posts(is_job_post);

-- Create index for video posts
CREATE INDEX IF NOT EXISTS posts_video_url_idx ON posts(video_url); 