-- Complete Database Setup for AccessAble Platform
-- Run this in your Supabase SQL Editor

-- 1. Drop existing tables if they exist
DROP TABLE IF EXISTS likes CASCADE;
DROP TABLE IF EXISTS comments CASCADE;
DROP TABLE IF EXISTS follows CASCADE;
DROP TABLE IF EXISTS posts CASCADE;
DROP TABLE IF EXISTS profiles CASCADE;
DROP TABLE IF EXISTS accessibility_options CASCADE;

-- 2. Create profiles table
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

-- 3. Create posts table
CREATE TABLE posts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES profiles(id) ON DELETE CASCADE,
  title text,
  content text,
  image_url text,
  tags text[],
  likes integer DEFAULT 0,
  comments integer DEFAULT 0,
  shares integer DEFAULT 0,
  created_at timestamp with time zone DEFAULT timezone('utc'::text, now()),
  updated_at timestamp with time zone DEFAULT timezone('utc'::text, now())
);

-- 4. Create comments table
CREATE TABLE comments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  post_id uuid REFERENCES posts(id) ON DELETE CASCADE,
  user_id uuid REFERENCES profiles(id) ON DELETE CASCADE,
  content text NOT NULL,
  author_name text,
  created_at timestamp with time zone DEFAULT timezone('utc'::text, now())
);

-- 5. Create likes table
CREATE TABLE likes (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  post_id uuid REFERENCES posts(id) ON DELETE CASCADE,
  user_id uuid REFERENCES profiles(id) ON DELETE CASCADE,
  created_at timestamp with time zone DEFAULT timezone('utc'::text, now()),
  UNIQUE(post_id, user_id)
);

-- 6. Create follows table
CREATE TABLE follows (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  follower_id uuid REFERENCES profiles(id) ON DELETE CASCADE,
  following_id uuid REFERENCES profiles(id) ON DELETE CASCADE,
  created_at timestamp with time zone DEFAULT timezone('utc'::text, now()),
  UNIQUE(follower_id, following_id)
);

-- 7. Create accessibility_options table
CREATE TABLE accessibility_options (
  id serial PRIMARY KEY,
  label text UNIQUE NOT NULL,
  description text
);

-- 8. Insert accessibility options
INSERT INTO accessibility_options (label, description) VALUES 
  ('Visual Impairment', 'Includes blindness and low vision'),
  ('Hearing Impairment', 'Includes deafness and hard of hearing'),
  ('Mobility Impairment', 'Includes limited use of hands or arms'),
  ('Cognitive Disability', 'Includes learning disabilities, memory issues, etc.'),
  ('Other', 'Other accessibility needs');

-- 9. Enable Row Level Security on all tables
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE posts ENABLE ROW LEVEL SECURITY;
ALTER TABLE comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE likes ENABLE ROW LEVEL SECURITY;
ALTER TABLE follows ENABLE ROW LEVEL SECURITY;

-- 10. Create RLS policies for profiles
CREATE POLICY "Users can view their own profile" ON profiles
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile" ON profiles
  FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert their own profile" ON profiles
  FOR INSERT WITH CHECK (auth.uid() = id);

CREATE POLICY "Public profiles are viewable by everyone" ON profiles
  FOR SELECT USING (profile_visibility = 'public');

-- 11. Create RLS policies for posts
CREATE POLICY "Users can view all posts" ON posts
  FOR SELECT USING (true);

CREATE POLICY "Users can create their own posts" ON posts
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own posts" ON posts
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own posts" ON posts
  FOR DELETE USING (auth.uid() = user_id);

-- 12. Create RLS policies for comments
CREATE POLICY "Users can view all comments" ON comments
  FOR SELECT USING (true);

CREATE POLICY "Users can create their own comments" ON comments
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own comments" ON comments
  FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own comments" ON comments
  FOR DELETE USING (auth.uid() = user_id);

-- 13. Create RLS policies for likes
CREATE POLICY "Users can view all likes" ON likes
  FOR SELECT USING (true);

CREATE POLICY "Users can create their own likes" ON likes
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own likes" ON likes
  FOR DELETE USING (auth.uid() = user_id);

-- 14. Create RLS policies for follows
CREATE POLICY "Users can view all follows" ON follows
  FOR SELECT USING (true);

CREATE POLICY "Users can create their own follows" ON follows
  FOR INSERT WITH CHECK (auth.uid() = follower_id);

CREATE POLICY "Users can delete their own follows" ON follows
  FOR DELETE USING (auth.uid() = follower_id);

-- 15. Create indexes for better performance
CREATE INDEX profiles_email_idx ON profiles(email);
CREATE INDEX profiles_location_idx ON profiles(location);
CREATE INDEX profiles_availability_idx ON profiles(availability);
CREATE INDEX posts_user_id_idx ON posts(user_id);
CREATE INDEX posts_created_at_idx ON posts(created_at DESC);
CREATE INDEX comments_post_id_idx ON comments(post_id);
CREATE INDEX comments_user_id_idx ON comments(user_id);
CREATE INDEX likes_post_id_idx ON likes(post_id);
CREATE INDEX likes_user_id_idx ON likes(user_id);
CREATE INDEX follows_follower_id_idx ON follows(follower_id);
CREATE INDEX follows_following_id_idx ON follows(following_id);

-- 16. Create functions to update post counts
CREATE OR REPLACE FUNCTION update_post_likes_count()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    UPDATE posts SET likes = likes + 1 WHERE id = NEW.post_id;
    RETURN NEW;
  ELSIF TG_OP = 'DELETE' THEN
    UPDATE posts SET likes = likes - 1 WHERE id = OLD.post_id;
    RETURN OLD;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_post_comments_count()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    UPDATE posts SET comments = comments + 1 WHERE id = NEW.post_id;
    RETURN NEW;
  ELSIF TG_OP = 'DELETE' THEN
    UPDATE posts SET comments = comments - 1 WHERE id = OLD.post_id;
    RETURN OLD;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- 17. Create triggers to automatically update counts
CREATE TRIGGER update_likes_count
  AFTER INSERT OR DELETE ON likes
  FOR EACH ROW EXECUTE FUNCTION update_post_likes_count();

CREATE TRIGGER update_comments_count
  AFTER INSERT OR DELETE ON comments
  FOR EACH ROW EXECUTE FUNCTION update_post_comments_count(); 