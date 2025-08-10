-- Test Database Setup for Follow Functionality
-- Run this in your Supabase SQL Editor to verify everything is working

-- 1. Check if user_follows table exists
SELECT 'Checking user_follows table:' as info;
SELECT EXISTS (
    SELECT FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_name = 'user_follows'
) as table_exists;

-- 2. Check table structure
SELECT 'User_follows table structure:' as info;
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_schema = 'public' AND table_name = 'user_follows'
ORDER BY ordinal_position;

-- 3. Check foreign key constraints
SELECT 'Foreign key constraints:' as info;
SELECT 
    tc.constraint_name,
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY' 
    AND tc.table_name = 'user_follows';

-- 4. Check RLS policies
SELECT 'RLS policies:' as info;
SELECT policyname, permissive, roles, cmd, qual 
FROM pg_policies 
WHERE tablename = 'user_follows';

-- 5. Test inserting a follow relationship (replace with actual user IDs)
-- First, let's see what users exist
SELECT 'Available users:' as info;
SELECT id, first_name, last_name FROM profiles LIMIT 5;

-- 6. Test the follow functionality (uncomment and modify with actual user IDs)
/*
-- Test follow relationship (replace user1_id and user2_id with actual UUIDs)
INSERT INTO user_follows (follower_id, following_id) 
VALUES ('user1-uuid', 'user2-uuid')
ON CONFLICT (follower_id, following_id) DO NOTHING;

-- Verify the relationship was created
SELECT 'Test follow relationship:' as info;
SELECT * FROM user_follows WHERE follower_id = 'user1-uuid' AND following_id = 'user2-uuid';

-- Test unfollow
DELETE FROM user_follows WHERE follower_id = 'user1-uuid' AND following_id = 'user2-uuid';
*/

-- 7. Check if there are any existing follows
SELECT 'Existing follow relationships:' as info;
SELECT COUNT(*) as total_follows FROM user_follows;

-- 8. Test a simple query to make sure the table is accessible
SELECT 'Testing simple query:' as info;
SELECT follower_id, following_id, created_at 
FROM user_follows 
LIMIT 5; 