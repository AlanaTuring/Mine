-- Fix user_follows table relationships
-- Run this in your Supabase SQL Editor

-- 1. Drop the existing user_follows table if it exists
DROP TABLE IF EXISTS public.user_follows CASCADE;

-- 2. Create user_follows table with correct foreign key references to profiles
CREATE TABLE public.user_follows (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    follower_id uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    following_id uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    UNIQUE(follower_id, following_id)
);

-- 3. Enable RLS on user_follows table
ALTER TABLE public.user_follows ENABLE ROW LEVEL SECURITY;

-- 4. Create RLS policies for user_follows table
-- Policy for inserting follows (users can only follow for themselves)
DROP POLICY IF EXISTS "Users can create their own follows" ON public.user_follows;
CREATE POLICY "Users can create their own follows" ON public.user_follows
    FOR INSERT WITH CHECK (auth.uid() = follower_id);

-- Policy for deleting follows (users can only unfollow their own follows)
DROP POLICY IF EXISTS "Users can delete their own follows" ON public.user_follows;
CREATE POLICY "Users can delete their own follows" ON public.user_follows
    FOR DELETE USING (auth.uid() = follower_id);

-- Policy for selecting follows (users can see all follows)
DROP POLICY IF EXISTS "Users can view all follows" ON public.user_follows;
CREATE POLICY "Users can view all follows" ON public.user_follows
    FOR SELECT USING (true);

-- 5. Create indexes for better performance
CREATE INDEX IF NOT EXISTS user_follows_follower_id_idx ON public.user_follows(follower_id);
CREATE INDEX IF NOT EXISTS user_follows_following_id_idx ON public.user_follows(following_id);
CREATE INDEX IF NOT EXISTS user_follows_follower_following_idx ON public.user_follows(follower_id, following_id);

-- 6. Verify the table structure
SELECT 'User_follows table structure:' as info;
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_schema = 'public' AND table_name = 'user_follows'
ORDER BY ordinal_position;

-- 7. Verify foreign key constraints
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

-- 8. Verify RLS policies
SELECT 'User_follows RLS policies:' as info;
SELECT policyname, permissive, roles, cmd, qual 
FROM pg_policies 
WHERE tablename = 'user_follows'; 