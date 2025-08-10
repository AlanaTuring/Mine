-- Check current database status
-- Run this in your Supabase SQL Editor to see what's wrong

-- 1. Check if tables exist
SELECT 'Tables:' as info;
SELECT 
    table_name,
    table_type
FROM information_schema.tables 
WHERE table_schema = 'public' 
    AND table_name IN ('profiles', 'posts', 'comments', 'likes', 'follows', 'accessibility_options')
ORDER BY table_name;

-- 2. Check RLS status
SELECT 'RLS Status:' as info;
SELECT 
    schemaname,
    tablename,
    rowsecurity
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename IN ('profiles', 'posts', 'comments', 'likes', 'follows', 'accessibility_options')
ORDER BY tablename;

-- 3. Check existing policies
SELECT 'Current Policies:' as info;
SELECT 
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd,
    qual
FROM pg_policies 
WHERE schemaname = 'public' 
    AND tablename IN ('profiles', 'posts', 'comments', 'likes', 'follows', 'accessibility_options')
ORDER BY tablename, policyname;

-- 4. Check if there's any data
SELECT 'Data Count:' as info;
SELECT 'profiles' as table_name, COUNT(*) as count FROM public.profiles
UNION ALL
SELECT 'posts' as table_name, COUNT(*) as count FROM public.posts
UNION ALL
SELECT 'comments' as table_name, COUNT(*) as count FROM public.comments
UNION ALL
SELECT 'likes' as table_name, COUNT(*) as count FROM public.likes
UNION ALL
SELECT 'follows' as table_name, COUNT(*) as count FROM public.follows
UNION ALL
SELECT 'accessibility_options' as table_name, COUNT(*) as count FROM public.accessibility_options;

-- 5. Check sample data from posts
SELECT 'Sample Posts:' as info;
SELECT id, user_id, title, content, created_at FROM public.posts LIMIT 3;

-- 6. Check sample data from profiles
SELECT 'Sample Profiles:' as info;
SELECT id, first_name, last_name, email FROM public.profiles LIMIT 3; 