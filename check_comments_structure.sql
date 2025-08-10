-- Check Comments Structure - Run this in your Supabase SQL Editor

-- 1. Check if posts table has a comments column
SELECT 'Posts table structure:' as info;
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_schema = 'public' AND table_name = 'posts'
ORDER BY ordinal_position;

-- 2. Check if comments table exists and its structure
SELECT 'Comments table structure:' as info;
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_schema = 'public' AND table_name = 'comments'
ORDER BY ordinal_position;

-- 3. Check if there are any comments in the database
SELECT 'Total comments count:' as info, COUNT(*) as total_comments FROM public.comments;

-- 4. Check comments for a specific post (if any posts exist)
SELECT 'Comments per post:' as info;
SELECT 
    p.id as post_id,
    p.title as post_title,
    COUNT(c.id) as comment_count
FROM public.posts p
LEFT JOIN public.comments c ON p.id = c.post_id
GROUP BY p.id, p.title
ORDER BY p.created_at DESC
LIMIT 10;

-- 5. Check if posts table has likes column
SELECT 'Posts table likes column:' as info;
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_schema = 'public' AND table_name = 'posts' AND column_name = 'likes';

-- 6. Check if posts table has shares column
SELECT 'Posts table shares column:' as info;
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_schema = 'public' AND table_name = 'posts' AND column_name = 'shares';

-- 7. Show sample post data
SELECT 'Sample post data:' as info;
SELECT id, title, likes, comments, shares, created_at
FROM public.posts
ORDER BY created_at DESC
LIMIT 5; 