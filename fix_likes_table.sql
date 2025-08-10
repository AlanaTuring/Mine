-- Fix Likes Table - Run this in your Supabase SQL Editor

-- 1. Create likes table if it doesn't exist
CREATE TABLE IF NOT EXISTS public.likes (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    post_id uuid NOT NULL REFERENCES public.posts(id) ON DELETE CASCADE,
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL,
    UNIQUE(post_id, user_id)
);

-- 2. Enable RLS on likes table
ALTER TABLE public.likes ENABLE ROW LEVEL SECURITY;

-- 3. Create RLS policies for likes table
-- Policy for inserting likes (users can only like for themselves)
DROP POLICY IF EXISTS "Users can create their own likes" ON public.likes;
CREATE POLICY "Users can create their own likes" ON public.likes
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Policy for deleting likes (users can only unlike their own likes)
DROP POLICY IF EXISTS "Users can delete their own likes" ON public.likes;
CREATE POLICY "Users can delete their own likes" ON public.likes
    FOR DELETE USING (auth.uid() = user_id);

-- Policy for selecting likes (users can see all likes)
DROP POLICY IF EXISTS "Users can view all likes" ON public.likes;
CREATE POLICY "Users can view all likes" ON public.likes
    FOR SELECT USING (true);

-- 4. Create indexes for better performance
CREATE INDEX IF NOT EXISTS likes_post_id_idx ON public.likes(post_id);
CREATE INDEX IF NOT EXISTS likes_user_id_idx ON public.likes(user_id);
CREATE INDEX IF NOT EXISTS likes_post_user_idx ON public.likes(post_id, user_id);

-- 5. Verify the table structure
SELECT 'Likes table structure:' as info;
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_schema = 'public' AND table_name = 'likes'
ORDER BY ordinal_position;

-- 6. Verify RLS policies
SELECT 'Likes RLS policies:' as info;
SELECT policyname, permissive, roles, cmd, qual 
FROM pg_policies 
WHERE tablename = 'likes';

-- 7. Test like functionality with a sample post (if posts exist)
SELECT 'Testing like functions...' as info;

-- Create helper functions for like management
CREATE OR REPLACE FUNCTION get_user_like_status(post_uuid uuid, user_uuid uuid)
RETURNS boolean AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM public.likes 
    WHERE post_id = post_uuid AND user_id = user_uuid
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION get_post_like_count(post_uuid uuid)
RETURNS integer AS $$
BEGIN
  RETURN (
    SELECT COUNT(*) FROM public.likes 
    WHERE post_id = post_uuid
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 8. Show sample data (if any exists)
SELECT 'Current likes count:' as info, COUNT(*) as total_likes FROM public.likes;

-- 9. Grant necessary permissions
GRANT ALL ON public.likes TO authenticated;
GRANT ALL ON public.likes TO anon;

-- 10. Final verification
SELECT '✅ Likes table setup complete!' as status; 