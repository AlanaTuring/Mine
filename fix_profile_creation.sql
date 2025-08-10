-- Fix Profile Creation Issue
-- This creates a trigger to automatically create a profile when a user signs up

-- First, let's create a function that will handle profile creation
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
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create the trigger
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Also, let's make sure the RLS policies are correct
DROP POLICY IF EXISTS "Users can insert their own profile" ON profiles;
DROP POLICY IF EXISTS "Users can view their own profile" ON profiles;
DROP POLICY IF EXISTS "Users can update their own profile" ON profiles;
DROP POLICY IF EXISTS "Public profiles are viewable by everyone" ON profiles;
DROP POLICY IF EXISTS "Allow profile creation during signup" ON profiles;
DROP POLICY IF EXISTS "Authenticated users can view profiles" ON profiles;

-- Create new policies
CREATE POLICY "Users can insert their own profile" ON profiles
  FOR INSERT WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can view their own profile" ON profiles
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile" ON profiles
  FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Public profiles are viewable by everyone" ON profiles
  FOR SELECT USING (profile_visibility = 'public' OR profile_visibility IS NULL);

-- Allow the trigger function to create profiles
CREATE POLICY "Allow trigger to create profiles" ON profiles
  FOR INSERT WITH CHECK (true);

-- Allow authenticated users to view profiles
CREATE POLICY "Authenticated users can view profiles" ON profiles
  FOR SELECT USING (auth.role() = 'authenticated'); 