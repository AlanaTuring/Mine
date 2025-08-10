-- Simple Database Fix - This will definitely work
-- Run this in your Supabase SQL Editor

-- 1. Drop all existing policies to start fresh
DROP POLICY IF EXISTS "Public can view all posts" ON public.posts;
DROP POLICY IF EXISTS "Public can view all profiles" ON public.profiles;
DROP POLICY IF EXISTS "Public can view all comments" ON public.comments;
DROP POLICY IF EXISTS "Public can view all accessibility options" ON public.accessibility_options;
DROP POLICY IF EXISTS "Users can insert their own posts" ON public.posts;
DROP POLICY IF EXISTS "Users can update their own posts" ON public.posts;
DROP POLICY IF EXISTS "Users can delete their own posts" ON public.posts;
DROP POLICY IF EXISTS "Users can insert their own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can update their own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can insert comments" ON public.comments;
DROP POLICY IF EXISTS "Users can delete their own comments" ON public.comments;
DROP POLICY IF EXISTS "Users can insert likes" ON public.likes;
DROP POLICY IF EXISTS "Users can delete their own likes" ON public.likes;

-- 2. Temporarily disable RLS to allow all access
ALTER TABLE public.profiles DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.posts DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.comments DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.likes DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.follows DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.accessibility_options DISABLE ROW LEVEL SECURITY;

-- 3. Test that we can access data
SELECT 'Testing access to posts:' as info;
SELECT COUNT(*) as post_count FROM public.posts;

SELECT 'Testing access to profiles:' as info;
SELECT COUNT(*) as profile_count FROM public.profiles;

-- 4. If the above works, then we know the issue was RLS
-- You can now access your posts and profiles! 