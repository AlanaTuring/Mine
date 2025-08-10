-- Fix RLS policies for profile creation
-- Run this in your Supabase SQL Editor

-- Drop existing policies that might be conflicting
DROP POLICY IF EXISTS "Users can insert their own profile" ON profiles;
DROP POLICY IF EXISTS "Users can view their own profile" ON profiles;
DROP POLICY IF EXISTS "Users can update their own profile" ON profiles;
DROP POLICY IF EXISTS "Public profiles are viewable by everyone" ON profiles;

-- Create new, more permissive policies
CREATE POLICY "Users can insert their own profile" ON profiles
  FOR INSERT WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can view their own profile" ON profiles
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile" ON profiles
  FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Public profiles are viewable by everyone" ON profiles
  FOR SELECT USING (profile_visibility = 'public' OR profile_visibility IS NULL);

-- Also allow users to view profiles during signup (when profile might not exist yet)
CREATE POLICY "Allow profile creation during signup" ON profiles
  FOR INSERT WITH CHECK (true);

-- Allow authenticated users to view all profiles (for now, can be restricted later)
CREATE POLICY "Authenticated users can view profiles" ON profiles
  FOR SELECT USING (auth.role() = 'authenticated'); 