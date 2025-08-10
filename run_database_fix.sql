-- Comprehensive Database Fix for AccessAble
-- Run this in your Supabase SQL Editor

-- 1. Enable RLS on all tables
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.posts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.likes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.follows ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.accessibility_options ENABLE ROW LEVEL SECURITY;

-- 2. Create public read policies (allow anyone to view posts, profiles, comments)
CREATE POLICY "Public can view all posts" ON public.posts
    FOR SELECT USING (true);

CREATE POLICY "Public can view all profiles" ON public.profiles
    FOR SELECT USING (true);

CREATE POLICY "Public can view all comments" ON public.comments
    FOR SELECT USING (true);

CREATE POLICY "Public can view all accessibility options" ON public.accessibility_options
    FOR SELECT USING (true);

-- 3. Create authenticated user policies
CREATE POLICY "Users can insert their own posts" ON public.posts
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own posts" ON public.posts
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own posts" ON public.posts
    FOR DELETE USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own profile" ON public.profiles
    FOR INSERT WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can update their own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert comments" ON public.comments
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own comments" ON public.comments
    FOR DELETE USING (auth.uid() = user_id);

CREATE POLICY "Users can insert likes" ON public.likes
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own likes" ON public.likes
    FOR DELETE USING (auth.uid() = user_id);

-- 4. Fix function search paths
CREATE OR REPLACE FUNCTION public.update_post_likes_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE public.posts SET likes = likes + 1 WHERE id = NEW.post_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE public.posts SET likes = likes - 1 WHERE id = OLD.post_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = public;

CREATE OR REPLACE FUNCTION public.update_post_comments_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE public.posts SET comments = comments + 1 WHERE id = NEW.post_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE public.posts SET comments = comments - 1 WHERE id = OLD.post_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = public;

-- 5. Verify everything is working
SELECT 'RLS Status:' as info;
SELECT 
    schemaname,
    tablename,
    rowsecurity
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename IN ('profiles', 'posts', 'comments', 'likes', 'follows', 'accessibility_options')
ORDER BY tablename;

SELECT 'Policies:' as info;
SELECT 
    schemaname,
    tablename,
    policyname,
    cmd
FROM pg_policies 
WHERE schemaname = 'public' 
    AND tablename IN ('profiles', 'posts', 'comments', 'likes', 'follows', 'accessibility_options')
ORDER BY tablename, policyname; 