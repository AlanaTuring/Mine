-- Fix Posts Table - Run this in your Supabase SQL Editor

-- 1. Add comments column to posts table if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'posts' 
        AND column_name = 'comments'
    ) THEN
        ALTER TABLE public.posts ADD COLUMN comments integer DEFAULT 0;
        RAISE NOTICE 'Added comments column to posts table';
    ELSE
        RAISE NOTICE 'Comments column already exists in posts table';
    END IF;
END $$;

-- 2. Add likes column to posts table if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'posts' 
        AND column_name = 'likes'
    ) THEN
        ALTER TABLE public.posts ADD COLUMN likes integer DEFAULT 0;
        RAISE NOTICE 'Added likes column to posts table';
    ELSE
        RAISE NOTICE 'Likes column already exists in posts table';
    END IF;
END $$;

-- 3. Add shares column to posts table if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'posts' 
        AND column_name = 'shares'
    ) THEN
        ALTER TABLE public.posts ADD COLUMN shares integer DEFAULT 0;
        RAISE NOTICE 'Added shares column to posts table';
    ELSE
        RAISE NOTICE 'Shares column already exists in posts table';
    END IF;
END $$;

-- 4. Update existing posts with correct comment counts
UPDATE public.posts 
SET comments = (
    SELECT COUNT(*) 
    FROM public.comments 
    WHERE comments.post_id = posts.id
);

-- 5. Update existing posts with correct like counts
UPDATE public.posts 
SET likes = (
    SELECT COUNT(*) 
    FROM public.likes 
    WHERE likes.post_id = posts.id
);

-- 6. Create triggers to automatically update comment counts
CREATE OR REPLACE FUNCTION update_post_comment_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE public.posts 
        SET comments = comments + 1 
        WHERE id = NEW.post_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE public.posts 
        SET comments = comments - 1 
        WHERE id = OLD.post_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Drop existing trigger if it exists
DROP TRIGGER IF EXISTS trigger_update_post_comment_count ON public.comments;

-- Create trigger for comment count updates
CREATE TRIGGER trigger_update_post_comment_count
    AFTER INSERT OR DELETE ON public.comments
    FOR EACH ROW
    EXECUTE FUNCTION update_post_comment_count();

-- 7. Create triggers to automatically update like counts
CREATE OR REPLACE FUNCTION update_post_like_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE public.posts 
        SET likes = likes + 1 
        WHERE id = NEW.post_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE public.posts 
        SET likes = likes - 1 
        WHERE id = OLD.post_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Drop existing trigger if it exists
DROP TRIGGER IF EXISTS trigger_update_post_like_count ON public.likes;

-- Create trigger for like count updates
CREATE TRIGGER trigger_update_post_like_count
    AFTER INSERT OR DELETE ON public.likes
    FOR EACH ROW
    EXECUTE FUNCTION update_post_like_count();

-- 8. Verify the changes
SELECT 'Posts table structure after fix:' as info;
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_schema = 'public' AND table_name = 'posts'
ORDER BY ordinal_position;

-- 9. Show sample data with counts
SELECT 'Sample posts with counts:' as info;
SELECT 
    id, 
    title, 
    likes, 
    comments, 
    shares,
    created_at
FROM public.posts
ORDER BY created_at DESC
LIMIT 5;

-- 10. Final verification
SELECT '✅ Posts table fix complete!' as status; 