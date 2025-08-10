-- Drop the existing profiles table if it exists
DROP TABLE IF EXISTS profiles CASCADE;

-- Create the profiles table with the correct schema
CREATE TABLE profiles (
  id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  first_name text,
  last_name text,
  title text,
  company text,
  location text,
  bio text,
  email text,
  phone text,
  website text,
  linkedin text,
  twitter text,
  github text,
  availability text,
  work_type text,
  skills text[],
  experience jsonb,
  education jsonb,
  accessibility_needs text[],
  accommodations text,
  show_email boolean DEFAULT false,
  show_phone boolean DEFAULT false,
  profile_visibility text DEFAULT 'public',
  avatar_url text,
  created_at timestamp with time zone DEFAULT timezone('utc'::text, now())
);

-- Enable Row Level Security
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Users can view their own profile" ON profiles
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile" ON profiles
  FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert their own profile" ON profiles
  FOR INSERT WITH CHECK (auth.uid() = id);

-- Create indexes for better performance
CREATE INDEX profiles_email_idx ON profiles(email);
CREATE INDEX profiles_location_idx ON profiles(location);
CREATE INDEX profiles_availability_idx ON profiles(availability); 