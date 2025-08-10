-- Complete Profile Creation Fix
-- Run this in your Supabase SQL Editor

-- 1. First, let's check if the trigger exists
SELECT 'Checking existing triggers:' as info;
SELECT 
    trigger_name,
    event_manipulation,
    event_object_table,
    action_statement
FROM information_schema.triggers 
WHERE trigger_schema = 'public' 
    AND event_object_table = 'users';

-- 2. Create the profile creation function
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger AS $$
BEGIN
  INSERT INTO public.profiles (id, email, first_name, last_name, accessibility_needs)
  VALUES (
    new.id,
    new.email,
    COALESCE(new.raw_user_meta_data->>'first_name', ''),
    COALESCE(new.raw_user_meta_data->>'last_name', ''),
    CASE 
      WHEN new.raw_user_meta_data->>'accessibility_needs' IS NOT NULL 
      THEN string_to_array(new.raw_user_meta_data->>'accessibility_needs', ',')
      ELSE NULL
    END
  );
  RETURN new;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = public;

-- 3. Create the trigger
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- 4. Disable RLS temporarily to allow profile creation
ALTER TABLE public.profiles DISABLE ROW LEVEL SECURITY;

-- 5. Create a manual profile for existing users (if any)
-- This will create profiles for users who already exist but don't have profiles
INSERT INTO public.profiles (id, email, first_name, last_name)
SELECT 
    au.id,
    au.email,
    COALESCE(au.raw_user_meta_data->>'first_name', ''),
    COALESCE(au.raw_user_meta_data->>'last_name', '')
FROM auth.users au
LEFT JOIN public.profiles p ON au.id = p.id
WHERE p.id IS NULL;

-- 6. Re-enable RLS with proper policies
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- 7. Create proper policies
DROP POLICY IF EXISTS "Users can insert their own profile" ON profiles;
DROP POLICY IF EXISTS "Users can view their own profile" ON profiles;
DROP POLICY IF EXISTS "Users can update their own profile" ON profiles;
DROP POLICY IF EXISTS "Public profiles are viewable by everyone" ON profiles;
DROP POLICY IF EXISTS "Allow trigger to create profiles" ON profiles;
DROP POLICY IF EXISTS "Authenticated users can view profiles" ON profiles;

-- Allow public read access to profiles
CREATE POLICY "Public can view profiles" ON profiles
  FOR SELECT USING (true);

-- Allow users to manage their own profiles
CREATE POLICY "Users can insert their own profile" ON profiles
  FOR INSERT WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can update their own profile" ON profiles
  FOR UPDATE USING (auth.uid() = id);

-- Allow the trigger to create profiles
CREATE POLICY "Allow trigger to create profiles" ON profiles
  FOR INSERT WITH CHECK (true);

-- 8. Verify the fix worked
SELECT 'Profile count after fix:' as info;
SELECT COUNT(*) as profile_count FROM public.profiles;

SELECT 'Sample profiles:' as info;
SELECT id, email, first_name, last_name FROM public.profiles LIMIT 3; 