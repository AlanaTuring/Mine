-- Update Existing Comment Counts - Run this in your Supabase SQL Editor

-- 1. Show current state
SELECT 'Current posts with comment counts:' as info;
SELECT 
    id, 
    title, 
    comments as current_comments,
    likes as current_likes
FROM public.posts
ORDER BY created_at DESC
LIMIT 10;

-- 2. Count actual comments per post
SELECT 'Actual comment counts from comments table:' as info;
SELECT 
    post_id,
    COUNT(*) as actual_comment_count
FROM public.comments
GROUP BY post_id
ORDER BY actual_comment_count DESC;

-- 3. Update comment counts for all posts
UPDATE public.posts 
SET comments = (
    SELECT COALESCE(COUNT(*), 0)
    FROM public.comments 
    WHERE comments.post_id = posts.id
);

-- 4. Update like counts for all posts
UPDATE public.posts 
SET likes = (
    SELECT COALESCE(COUNT(*), 0)
    FROM public.likes 
    WHERE likes.post_id = posts.id
);

-- 5. Show updated state
SELECT 'Updated posts with comment counts:' as info;
SELECT 
    id, 
    title, 
    comments as updated_comments,
    likes as updated_likes
FROM public.posts
ORDER BY created_at DESC
LIMIT 10;

-- 6. Verify the update worked
SELECT 'Verification - posts with comments:' as info;
SELECT 
    p.id,
    p.title,
    p.comments as post_comments,
    COUNT(c.id) as actual_comments
FROM public.posts p
LEFT JOIN public.comments c ON p.id = c.post_id
GROUP BY p.id, p.title, p.comments
HAVING p.comments != COUNT(c.id)
ORDER BY p.created_at DESC;

-- 7. Final status
SELECT '✅ Comment count update complete!' as status; 