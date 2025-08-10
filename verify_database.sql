-- Database Verification Script
-- Run this in your Supabase SQL Editor to check if everything is set up correctly

-- 1. Check if required tables exist
SELECT 'Checking tables...' as info;

SELECT 
  table_name,
  CASE 
    WHEN table_name IN ('posts', 'comments', 'likes', 'profiles') THEN '✅ EXISTS'
    ELSE '❌ MISSING'
  END as status
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('posts', 'comments', 'likes', 'profiles');

-- 2. Check table structures
SELECT 'Checking likes table structure...' as info;
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'likes' 
ORDER BY ordinal_position;

SELECT 'Checking comments table structure...' as info;
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'comments' 
ORDER BY ordinal_position;

-- 3. Check RLS policies
SELECT 'Checking RLS policies...' as info;
SELECT 
  tablename,
  policyname,
  permissive,
  cmd,
  CASE 
    WHEN cmd IN ('INSERT', 'UPDATE', 'DELETE', 'SELECT') THEN '✅ EXISTS'
    ELSE '⚠️  CHECK'
  END as status
FROM pg_policies 
WHERE tablename IN ('posts', 'comments', 'likes')
ORDER BY tablename, cmd;

-- 4. Check constraints
SELECT 'Checking likes table constraints...' as info;
SELECT conname, contype, pg_get_constraintdef(oid) 
FROM pg_constraint 
WHERE conrelid = 'likes'::regclass;

-- 5. Check if RLS is enabled
SELECT 'Checking RLS status...' as info;
SELECT 
  schemaname,
  tablename,
  rowsecurity,
  CASE 
    WHEN rowsecurity THEN '✅ ENABLED'
    ELSE '❌ DISABLED'
  END as status
FROM pg_tables 
WHERE tablename IN ('posts', 'comments', 'likes')
  AND schemaname = 'public';

-- 6. Sample data check
SELECT 'Checking sample data...' as info;
SELECT 'Posts count:' as info, COUNT(*) as count FROM posts;
SELECT 'Comments count:' as info, COUNT(*) as count FROM comments;
SELECT 'Likes count:' as info, COUNT(*) as count FROM likes;
SELECT 'Profiles count:' as info, COUNT(*) as count FROM profiles;

-- 7. Test like functionality
SELECT 'Testing like functions...' as info;
SELECT 
  CASE 
    WHEN EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'get_user_like_status') THEN '✅ EXISTS'
    ELSE '❌ MISSING'
  END as get_user_like_status_function;

SELECT 
  CASE 
    WHEN EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'get_post_like_count') THEN '✅ EXISTS'
    ELSE '❌ MISSING'
  END as get_post_like_count_function; 