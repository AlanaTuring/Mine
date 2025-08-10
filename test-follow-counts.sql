-- Test Follow Counts
-- Run this in your Supabase SQL Editor to verify follow functionality

-- 1. Check if user_follows table exists and has data
SELECT 'User_follows table check:' as info;
SELECT COUNT(*) as total_follows FROM user_follows;

-- 2. Show sample follow relationships
SELECT 'Sample follow relationships:' as info;
SELECT 
    uf.follower_id,
    uf.following_id,
    uf.created_at,
    p1.first_name as follower_name,
    p2.first_name as following_name
FROM user_follows uf
LEFT JOIN profiles p1 ON uf.follower_id = p1.id
LEFT JOIN profiles p2 ON uf.following_id = p2.id
LIMIT 5;

-- 3. Test follower count for a specific user (replace with actual user ID)
SELECT 'Follower count test:' as info;
SELECT 
    p.id,
    p.first_name,
    p.last_name,
    COUNT(uf.follower_id) as followers_count
FROM profiles p
LEFT JOIN user_follows uf ON p.id = uf.following_id
GROUP BY p.id, p.first_name, p.last_name
LIMIT 5;

-- 4. Test following count for a specific user (replace with actual user ID)
SELECT 'Following count test:' as info;
SELECT 
    p.id,
    p.first_name,
    p.last_name,
    COUNT(uf.following_id) as following_count
FROM profiles p
LEFT JOIN user_follows uf ON p.id = uf.follower_id
GROUP BY p.id, p.first_name, p.last_name
LIMIT 5;

-- 5. Create a test follow relationship (uncomment and modify with actual user IDs)
/*
-- Test follow relationship (replace user1_id and user2_id with actual UUIDs)
INSERT INTO user_follows (follower_id, following_id) 
VALUES ('user1-uuid', 'user2-uuid')
ON CONFLICT (follower_id, following_id) DO NOTHING;

-- Verify the relationship was created
SELECT 'Test follow relationship:' as info;
SELECT * FROM user_follows WHERE follower_id = 'user1-uuid' AND following_id = 'user2-uuid';
*/ 