-- Safe Profile Creation Fix (No Conflicts)
-- Run this in your Supabase SQL Editor

-- 1. Create the profile creation function (safe to run multiple times)
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

-- 2. Create the trigger (safe to run multiple times)
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- 3. Temporarily disable RLS to allow profile creation
ALTER TABLE public.profiles DISABLE ROW LEVEL SECURITY;

-- 4. Create profiles for existing users who don't have them
INSERT INTO public.profiles (id, email, first_name, last_name)
SELECT 
    au.id,
    au.email,
    COALESCE(au.raw_user_meta_data->>'first_name', ''),
    COALESCE(au.raw_user_meta_data->>'last_name', '')
FROM auth.users au
LEFT JOIN public.profiles p ON au.id = p.id
WHERE p.id IS NULL;

-- 5. Re-enable RLS
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- 6. Create policies only if they don't exist (using IF NOT EXISTS pattern)
DO $$
BEGIN
    -- Only create policies if they don't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE schemaname = 'public' 
        AND tablename = 'profiles' 
        AND policyname = 'Public can view profiles'
    ) THEN
        CREATE POLICY "Public can view profiles" ON profiles
        FOR SELECT USING (true);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE schemaname = 'public' 
        AND tablename = 'profiles' 
        AND policyname = 'Users can insert their own profile'
    ) THEN
        CREATE POLICY "Users can insert their own profile" ON profiles
        FOR INSERT WITH CHECK (auth.uid() = id);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE schemaname = 'public' 
        AND tablename = 'profiles' 
        AND policyname = 'Users can update their own profile'
    ) THEN
        CREATE POLICY "Users can update their own profile" ON profiles
        FOR UPDATE USING (auth.uid() = id);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE schemaname = 'public' 
        AND tablename = 'profiles' 
        AND policyname = 'Allow trigger to create profiles'
    ) THEN
        CREATE POLICY "Allow trigger to create profiles" ON profiles
        FOR INSERT WITH CHECK (true);
    END IF;
END $$;

-- 7. Also fix posts and comments access
ALTER TABLE public.posts DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.comments DISABLE ROW LEVEL SECURITY;

-- 8. Verify the fix worked
SELECT 'Profile count after fix:' as info;
SELECT COUNT(*) as profile_count FROM public.profiles;

SELECT 'Sample profiles:' as info;
SELECT id, email, first_name, last_name FROM public.profiles LIMIT 3;

SELECT 'Posts count:' as info;
SELECT COUNT(*) as posts_count FROM public.posts; 