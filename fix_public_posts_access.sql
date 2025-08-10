-- Fix public access to posts for unauthenticated users
-- Run this in your Supabase SQL Editor

-- 1. Create a policy to allow public read access to posts
CREATE POLICY "Public can view all posts" ON public.posts
    FOR SELECT USING (true);

-- 2. Create a policy to allow public read access to profiles (for post authors)
CREATE POLICY "Public can view profiles" ON public.profiles
    FOR SELECT USING (true);

-- 3. Create a policy to allow public read access to comments
CREATE POLICY "Public can view comments" ON public.comments
    FOR SELECT USING (true);

-- 4. Verify the policies are created
SELECT 
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd,
    qual
FROM pg_policies 
WHERE schemaname = 'public' 
    AND tablename IN ('posts', 'profiles', 'comments')
ORDER BY tablename, policyname; 