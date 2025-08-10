-- Quick Fix for Posts Access
-- Run this in your Supabase SQL Editor

-- 1. Disable RLS on posts table to allow access
ALTER TABLE public.posts DISABLE ROW LEVEL SECURITY;

-- 2. Disable RLS on comments table as well
ALTER TABLE public.comments DISABLE ROW LEVEL SECURITY;

-- 3. Disable RLS on likes table
ALTER TABLE public.likes DISABLE ROW LEVEL SECURITY;

-- 4. Verify posts are accessible
SELECT 'Posts count:' as info;
SELECT COUNT(*) as posts_count FROM public.posts;

SELECT 'Sample posts:' as info;
SELECT id, user_id, title, content, created_at FROM public.posts LIMIT 3;

-- 5. Check if posts have valid user_id references
SELECT 'Posts with user references:' as info;
SELECT p.id, p.title, p.user_id, pr.first_name, pr.last_name
FROM public.posts p
LEFT JOIN public.profiles pr ON p.user_id = pr.id
LIMIT 5; 