-- Setup user_follows table for follow/unfollow functionality
-- Run this in your Supabase SQL Editor

-- 1. Create user_follows table if it doesn't exist
CREATE TABLE IF NOT EXISTS public.user_follows (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    follower_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    following_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    UNIQUE(follower_id, following_id)
);

-- 2. Enable RLS on user_follows table
ALTER TABLE public.user_follows ENABLE ROW LEVEL SECURITY;

-- 3. Create RLS policies for user_follows table
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

-- 4. Create indexes for better performance
CREATE INDEX IF NOT EXISTS user_follows_follower_id_idx ON public.user_follows(follower_id);
CREATE INDEX IF NOT EXISTS user_follows_following_id_idx ON public.user_follows(following_id);
CREATE INDEX IF NOT EXISTS user_follows_follower_following_idx ON public.user_follows(follower_id, following_id);

-- 5. Create function to check if user is following another user
CREATE OR REPLACE FUNCTION is_following(follower_uuid uuid, following_uuid uuid)
RETURNS boolean AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM user_follows 
    WHERE follower_id = follower_uuid AND following_id = following_uuid
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 6. Create function to get follower count
CREATE OR REPLACE FUNCTION get_follower_count(user_uuid uuid)
RETURNS integer AS $$
BEGIN
  RETURN (
    SELECT COUNT(*) FROM user_follows 
    WHERE following_id = user_uuid
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 7. Create function to get following count
CREATE OR REPLACE FUNCTION get_following_count(user_uuid uuid)
RETURNS integer AS $$
BEGIN
  RETURN (
    SELECT COUNT(*) FROM user_follows 
    WHERE follower_id = user_uuid
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 8. Verify the table structure
SELECT 'User_follows table structure:' as info;
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_schema = 'public' AND table_name = 'user_follows'
ORDER BY ordinal_position;

-- 9. Verify RLS policies
SELECT 'User_follows RLS policies:' as info;
SELECT policyname, permissive, roles, cmd, qual 
FROM pg_policies 
WHERE tablename = 'user_follows';

-- 10. Verify indexes
SELECT 'User_follows indexes:' as info;
SELECT indexname, indexdef
FROM pg_indexes 
WHERE tablename = 'user_follows'; 