-- Test Storage Setup
-- Run this to verify your storage is configured correctly

-- Check if buckets exist
SELECT name, public FROM storage.buckets WHERE name IN ('avatars', 'post-media');

-- Check storage policies
SELECT 
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd,
    qual,
    with_check
FROM pg_policies 
WHERE tablename = 'objects' 
AND schemaname = 'storage';

-- Check if RLS is enabled on storage.objects
SELECT schemaname, tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'storage' 
AND tablename = 'objects';

-- Check user permissions
SELECT 
    grantee,
    privilege_type,
    is_grantable
FROM information_schema.role_table_grants 
WHERE table_schema = 'storage' 
AND table_name = 'objects'; 