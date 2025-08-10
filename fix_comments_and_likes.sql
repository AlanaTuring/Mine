-- Fix Comments and Likes - Add missing RLS policies
-- Run this in your Supabase SQL Editor

-- 1. Add policy for post owners to delete any comment on their posts
DROP POLICY IF EXISTS "Post owners can delete any comment on their posts" ON comments;
CREATE POLICY "Post owners can delete any comment on their posts" ON comments
  FOR DELETE USING (
    EXISTS (
      SELECT 1 FROM posts 
      WHERE posts.id = comments.post_id 
      AND posts.user_id = auth.uid()
    )
  );

-- 2. Ensure the existing policies are correct
-- Users can delete their own comments (already exists)
-- Post owners can delete any comment on their posts (added above)

-- 3. Verify the likes table has proper constraints
-- The UNIQUE(post_id, user_id) constraint should already exist

-- 4. Add a function to get user's like status for a post
CREATE OR REPLACE FUNCTION get_user_like_status(post_uuid uuid, user_uuid uuid)
RETURNS boolean AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM likes 
    WHERE post_id = post_uuid AND user_id = user_uuid
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 5. Add a function to get like count for a post
CREATE OR REPLACE FUNCTION get_post_like_count(post_uuid uuid)
RETURNS integer AS $$
BEGIN
  RETURN (
    SELECT COUNT(*) FROM likes 
    WHERE post_id = post_uuid
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 6. Verify all policies are in place
SELECT 'Comments policies:' as info;
SELECT policyname, permissive, roles, cmd, qual 
FROM pg_policies 
WHERE tablename = 'comments';

SELECT 'Likes policies:' as info;
SELECT policyname, permissive, roles, cmd, qual 
FROM pg_policies 
WHERE tablename = 'likes'; 