-- Complete Storage RLS Fix
-- Run this in your Supabase SQL Editor

-- First, enable RLS on storage.objects if not already enabled
ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;

-- Drop ALL existing storage policies to start fresh
DROP POLICY IF EXISTS "Avatar images are publicly accessible" ON storage.objects;
DROP POLICY IF EXISTS "Users can upload their own avatar" ON storage.objects;
DROP POLICY IF EXISTS "Users can update their own avatar" ON storage.objects;
DROP POLICY IF EXISTS "Users can delete their own avatar" ON storage.objects;
DROP POLICY IF EXISTS "Post media is publicly accessible" ON storage.objects;
DROP POLICY IF EXISTS "Users can upload post media" ON storage.objects;
DROP POLICY IF EXISTS "Users can update their own post media" ON storage.objects;
DROP POLICY IF EXISTS "Users can delete their own post media" ON storage.objects;
DROP POLICY IF EXISTS "Public Access" ON storage.objects;
DROP POLICY IF EXISTS "Authenticated users can upload files" ON storage.objects;
DROP POLICY IF EXISTS "Users can update own files" ON storage.objects;
DROP POLICY IF EXISTS "Users can delete own files" ON storage.objects;

-- Create comprehensive policies for avatars bucket
CREATE POLICY "Avatar images are publicly accessible" ON storage.objects
  FOR SELECT USING (bucket_id = 'avatars');

CREATE POLICY "Users can upload avatars" ON storage.objects
  FOR INSERT WITH CHECK (
    bucket_id = 'avatars' 
    AND auth.role() = 'authenticated'
  );

CREATE POLICY "Users can update avatars" ON storage.objects
  FOR UPDATE USING (
    bucket_id = 'avatars' 
    AND auth.role() = 'authenticated'
  );

CREATE POLICY "Users can delete avatars" ON storage.objects
  FOR DELETE USING (
    bucket_id = 'avatars' 
    AND auth.role() = 'authenticated'
  );

-- Create comprehensive policies for post-media bucket
CREATE POLICY "Post media is publicly accessible" ON storage.objects
  FOR SELECT USING (bucket_id = 'post-media');

CREATE POLICY "Users can upload post media" ON storage.objects
  FOR INSERT WITH CHECK (
    bucket_id = 'post-media' 
    AND auth.role() = 'authenticated'
  );

CREATE POLICY "Users can update post media" ON storage.objects
  FOR UPDATE USING (
    bucket_id = 'post-media' 
    AND auth.role() = 'authenticated'
  );

CREATE POLICY "Users can delete post media" ON storage.objects
  FOR DELETE USING (
    bucket_id = 'post-media' 
    AND auth.role() = 'authenticated'
  );

-- Create a general policy for authenticated users to access all storage
CREATE POLICY "Authenticated users can access storage" ON storage.objects
  FOR ALL USING (auth.role() = 'authenticated');

-- Grant necessary permissions
GRANT ALL ON storage.objects TO authenticated;
GRANT ALL ON storage.buckets TO authenticated; 