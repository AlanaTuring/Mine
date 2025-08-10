-- Disable RLS for Storage (Temporary Fix)
-- Run this in your Supabase SQL Editor

-- Disable RLS on storage.objects to allow all operations
ALTER TABLE storage.objects DISABLE ROW LEVEL SECURITY;

-- Note: This is a temporary fix for development
-- In production, you should use proper RLS policies
-- But for now, this will allow image uploads to work 