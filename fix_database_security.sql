-- Fix Database Security Issues for AccessAble Platform
-- Run this in your Supabase SQL Editor

-- 1. Enable RLS on accessibility_options table (this is the only table missing RLS)
ALTER TABLE public.accessibility_options ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for accessibility_options table
CREATE POLICY "Users can view all accessibility options" ON public.accessibility_options
    FOR SELECT USING (true);

-- 2. Fix function search paths by adding SET search_path
-- Update update_post_likes_count function
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

-- Update update_post_comments_count function
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

-- Update handle_new_user function (if it exists) - Fixed syntax
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.routines WHERE routine_schema = 'public' AND routine_name = 'handle_new_user') THEN
        EXECUTE 'CREATE OR REPLACE FUNCTION public.handle_new_user()
        RETURNS TRIGGER AS $func$
        BEGIN
            INSERT INTO public.profiles (id, first_name, last_name, email, accessibility_needs)
            VALUES (
                NEW.id,
                NEW.raw_user_meta_data->>''first_name'',
                NEW.raw_user_meta_data->>''last_name'',
                NEW.email,
                CASE 
                    WHEN NEW.raw_user_meta_data->>''accessibility_needs'' IS NOT NULL 
                    THEN string_to_array(NEW.raw_user_meta_data->>''accessibility_needs'', '','')
                    ELSE NULL
                END
            );
            RETURN NEW;
        END;
        $func$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = public;';
    END IF;
END $$;

-- 3. Verify RLS is enabled on all tables
SELECT 
    schemaname,
    tablename,
    rowsecurity
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename IN ('profiles', 'posts', 'comments', 'likes', 'follows', 'accessibility_options')
ORDER BY tablename;

-- 4. Verify function search paths are set
SELECT 
    p.proname as function_name,
    p.prosecdef as security_definer,
    p.proconfig as search_path_config
FROM pg_proc p
JOIN pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname = 'public' 
    AND p.proname IN ('update_post_likes_count', 'update_post_comments_count', 'handle_new_user')
ORDER BY p.proname; 