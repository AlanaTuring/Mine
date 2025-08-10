-- Fix Posts Table Foreign Key Relationship
-- Run this in your Supabase SQL Editor

-- 1. Check current posts table structure
SELECT 'Current posts table structure:' as info;
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_schema = 'public' AND table_name = 'posts'
ORDER BY ordinal_position;

-- 2. Check current foreign key constraints
SELECT 'Current foreign key constraints:' as info;
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
    AND tc.table_name = 'posts';

-- 3. Drop existing foreign key constraint if it exists
DO $$
BEGIN
    -- Drop the foreign key constraint if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'posts_user_id_fkey' 
        AND table_name = 'posts'
    ) THEN
        ALTER TABLE posts DROP CONSTRAINT posts_user_id_fkey;
    END IF;
END $$;

-- 4. Add the correct foreign key constraint
ALTER TABLE posts 
ADD CONSTRAINT posts_user_id_fkey 
FOREIGN KEY (user_id) REFERENCES profiles(id) ON DELETE CASCADE;

-- 5. Verify the foreign key constraint was created
SELECT 'Updated foreign key constraints:' as info;
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
    AND tc.table_name = 'posts';

-- 6. Test the relationship with a simple query
SELECT 'Testing posts-profiles relationship:' as info;
SELECT 
    p.id as post_id,
    p.title as post_title,
    p.user_id,
    pr.first_name,
    pr.last_name
FROM posts p
LEFT JOIN profiles pr ON p.user_id = pr.id
LIMIT 5; 