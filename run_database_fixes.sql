-- Run this in your Supabase SQL Editor to fix all database issues

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

-- 2. Ensure the likes table has proper constraints
-- Check if the unique constraint exists, if not add it
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint 
    WHERE conname = 'likes_post_id_user_id_key'
  ) THEN
    ALTER TABLE likes ADD CONSTRAINT likes_post_id_user_id_key UNIQUE (post_id, user_id);
  END IF;
END $$;

-- 3. Add functions for like management
CREATE OR REPLACE FUNCTION get_user_like_status(post_uuid uuid, user_uuid uuid)
RETURNS boolean AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM likes 
    WHERE post_id = post_uuid AND user_id = user_uuid
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION get_post_like_count(post_uuid uuid)
RETURNS integer AS $$
BEGIN
  RETURN (
    SELECT COUNT(*) FROM likes 
    WHERE post_id = post_uuid
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 4. Ensure all RLS policies are in place for posts
DROP POLICY IF EXISTS "Users can update their own posts" ON posts;
CREATE POLICY "Users can update their own posts" ON posts
  FOR UPDATE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own posts" ON posts;
CREATE POLICY "Users can delete their own posts" ON posts
  FOR DELETE USING (auth.uid() = user_id);

-- 5. Ensure all RLS policies are in place for comments
DROP POLICY IF EXISTS "Users can update their own comments" ON comments;
CREATE POLICY "Users can update their own comments" ON comments
  FOR UPDATE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own comments" ON comments;
CREATE POLICY "Users can delete their own comments" ON comments
  FOR DELETE USING (auth.uid() = user_id);

-- 6. Ensure all RLS policies are in place for likes
DROP POLICY IF EXISTS "Users can create their own likes" ON likes;
CREATE POLICY "Users can create their own likes" ON likes
  FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own likes" ON likes;
CREATE POLICY "Users can delete their own likes" ON likes
  FOR DELETE USING (auth.uid() = user_id);

-- 7. Verify all policies are in place
SELECT 'Comments policies:' as info;
SELECT policyname, permissive, roles, cmd, qual 
FROM pg_policies 
WHERE tablename = 'comments';

SELECT 'Likes policies:' as info;
SELECT policyname, permissive, roles, cmd, qual 
FROM pg_policies 
WHERE tablename = 'likes';

SELECT 'Posts policies:' as info;
SELECT policyname, permissive, roles, cmd, qual 
FROM pg_policies 
WHERE tablename = 'posts';

-- 8. Check table constraints
SELECT 'Likes table constraints:' as info;
SELECT conname, contype, pg_get_constraintdef(oid) 
FROM pg_constraint 
WHERE conrelid = 'likes'::regclass; 