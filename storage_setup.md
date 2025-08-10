# Supabase Storage Setup

## Required Storage Buckets

You need to create the following storage buckets in your Supabase dashboard:

### 1. Create 'avatars' bucket
1. Go to **Storage** in your Supabase dashboard
2. Click **Create a new bucket**
3. Name: `avatars`
4. Set to **Public**
5. Click **Create bucket**

### 2. Create 'post-media' bucket
1. Go to **Storage** in your Supabase dashboard
2. Click **Create a new bucket**
3. Name: `post-media`
4. Set to **Public**
5. Click **Create bucket**

### 3. Set up RLS policies for avatars bucket
Run this SQL in the SQL Editor:

```sql
-- Allow authenticated users to upload their own avatars
CREATE POLICY "Users can upload their own avatars" ON storage.objects
  FOR INSERT WITH CHECK (
    bucket_id = 'avatars' AND 
    auth.uid()::text = (storage.foldername(name))[1]
  );

-- Allow public access to avatars
CREATE POLICY "Public access to avatars" ON storage.objects
  FOR SELECT USING (bucket_id = 'avatars');

-- Allow users to update their own avatars
CREATE POLICY "Users can update their own avatars" ON storage.objects
  FOR UPDATE USING (
    bucket_id = 'avatars' AND 
    auth.uid()::text = (storage.foldername(name))[1]
  );

-- Allow users to delete their own avatars
CREATE POLICY "Users can delete their own avatars" ON storage.objects
  FOR DELETE USING (
    bucket_id = 'avatars' AND 
    auth.uid()::text = (storage.foldername(name))[1]
  );
```

### 4. Set up RLS policies for post-media bucket
Run this SQL in the SQL Editor:

```sql
-- Allow authenticated users to upload post media
CREATE POLICY "Users can upload post media" ON storage.objects
  FOR INSERT WITH CHECK (
    bucket_id = 'post-media' AND 
    auth.uid()::text = (storage.foldername(name))[1]
  );

-- Allow public access to post media
CREATE POLICY "Public access to post media" ON storage.objects
  FOR SELECT USING (bucket_id = 'post-media');

-- Allow users to update their own post media
CREATE POLICY "Users can update their own post media" ON storage.objects
  FOR UPDATE USING (
    bucket_id = 'post-media' AND 
    auth.uid()::text = (storage.foldername(name))[1]
  );

-- Allow users to delete their own post media
CREATE POLICY "Users can delete their own post media" ON storage.objects
  FOR DELETE USING (
    bucket_id = 'post-media' AND 
    auth.uid()::text = (storage.foldername(name))[1]
  );
```

## Complete Setup Steps

1. **Run the complete database setup**: Copy and paste the content from `complete_database_setup.sql` into your Supabase SQL Editor and run it.

2. **Create storage buckets**: Follow the steps above to create the `avatars` and `post-media` buckets.

3. **Set up storage policies**: Run the RLS policies SQL for both buckets.

4. **Test the application**: Try creating an account, uploading a profile picture, and creating posts with images.

## Features Now Available

After completing this setup, you'll have:

✅ **User Authentication** - Sign up/sign in with email
✅ **Profile Management** - Complete profile editing with avatar upload
✅ **Post Creation** - Create posts with text and images
✅ **Real-time Likes** - Like/unlike posts with database persistence
✅ **Comments System** - Add comments to posts
✅ **Follow System** - Follow/unfollow other users
✅ **Profile Pictures** - Upload and display user avatars
✅ **Post Media** - Upload images to posts
✅ **Real-time Updates** - All interactions update immediately
✅ **Accessibility Features** - High contrast, large text, audio assistance
✅ **Responsive Design** - Works on all devices
✅ **Security** - Row Level Security on all data
✅ **Performance** - Optimized database indexes

Your AccessAble platform will be fully functional! 