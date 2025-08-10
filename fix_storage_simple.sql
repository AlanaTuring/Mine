-- Simple Storage Fix for Supabase
-- Run this in your Supabase SQL Editor

-- Drop existing policies (only the ones we can control)
DROP POLICY IF EXISTS "Avatar images are publicly accessible" ON storage.objects;
DROP POLICY IF EXISTS "Users can upload avatars" ON storage.objects;
DROP POLICY IF EXISTS "Users can update avatars" ON storage.objects;
DROP POLICY IF EXISTS "Users can delete avatars" ON storage.objects;
DROP POLICY IF EXISTS "Post media is publicly accessible" ON storage.objects;
DROP POLICY IF EXISTS "Users can upload post media" ON storage.objects;
DROP POLICY IF EXISTS "Users can update post media" ON storage.objects;
DROP POLICY IF EXISTS "Users can delete post media" ON storage.objects;
DROP POLICY IF EXISTS "Authenticated users can access storage" ON storage.objects;

-- Create simple, permissive policies for authenticated users
CREATE POLICY "Allow authenticated users to access avatars" ON storage.objects
  FOR ALL USING (
    bucket_id = 'avatars' 
    AND auth.role() = 'authenticated'
  );

CREATE POLICY "Allow authenticated users to access post media" ON storage.objects
  FOR ALL USING (
    bucket_id = 'post-media' 
    AND auth.role() = 'authenticated'
  );

-- Create public read access for both buckets
CREATE POLICY "Public read access for avatars" ON storage.objects
  FOR SELECT USING (bucket_id = 'avatars');

CREATE POLICY "Public read access for post media" ON storage.objects
  FOR SELECT USING (bucket_id = 'post-media'); 