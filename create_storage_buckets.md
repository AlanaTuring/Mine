# 🗂️ Create Storage Buckets

## **Step 1: Go to Supabase Storage**

1. **Open Supabase Dashboard**
2. **Click on "Storage"** in the left sidebar
3. **Click "Create a new bucket"**

## **Step 2: Create Avatars Bucket**

1. **Bucket name:** `avatars`
2. **Public bucket:** ✅ **Check this box**
3. **Click "Create bucket"**

## **Step 3: Create Post Media Bucket**

1. **Bucket name:** `post-media`
2. **Public bucket:** ✅ **Check this box**
3. **Click "Create bucket"**

## **Step 4: Verify Buckets**

You should now see two buckets:
- ✅ `avatars` (Public)
- ✅ `post-media` (Public)

## **Step 5: Test Avatar Upload**

1. **Go to your profile page**
2. **Try uploading an avatar**
3. **Should work without errors now!**

## 🚨 **If Still Getting Errors:**

1. **Make sure buckets are PUBLIC**
2. **Run the storage RLS fix again:**
   ```sql
   -- Copy and paste fix_storage_rls.sql
   ```
3. **Clear browser cache and try again**

## ✅ **What This Fixes:**

- **Avatar uploads** in profile
- **Image uploads** in posts
- **No more storage errors**
- **Public access** to uploaded files 