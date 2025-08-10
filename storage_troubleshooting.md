# 🔧 Storage Troubleshooting Guide

## 🚨 **Error: "must be owner of table objects"**

This error occurs because you can't modify the storage schema directly in Supabase. Here's the correct approach:

## **Step 1: Try the Simple Fix First**

1. **Go to Supabase Dashboard** → **SQL Editor**
2. **Copy and paste `fix_storage_simple.sql`**
3. **Click Run**

This creates simple, permissive policies that should work.

## **Step 2: If That Doesn't Work - Disable RLS**

1. **Go to SQL Editor**
2. **Copy and paste `fix_storage_disable_rls.sql`**
3. **Click Run**

This disables RLS temporarily, allowing all storage operations.

## **Step 3: Verify Storage Buckets**

1. **Go to Storage** in Supabase Dashboard
2. **Make sure you have:**
   - ✅ `avatars` bucket (Public)
   - ✅ `post-media` bucket (Public)
3. **If missing, create them:**
   - Click "Create a new bucket"
   - Name: `avatars` → Check "Public bucket"
   - Name: `post-media` → Check "Public bucket"

## **Step 4: Test Image Upload**

1. **Go to your profile page**
2. **Try uploading an avatar**
3. **Go to dashboard**
4. **Try creating a post with image**

## 🚨 **Why the Error Happens:**

- **Supabase limitation**: You can't modify storage schema directly
- **Permission issue**: Only Supabase can modify storage.objects table
- **RLS policies**: Need to be created through the dashboard or simple SQL

## ✅ **Solutions in Order of Preference:**

### **Option 1: Simple Policies** (Recommended)
```sql
-- Use fix_storage_simple.sql
-- Creates permissive policies for authenticated users
```

### **Option 2: Disable RLS** (Quick Fix)
```sql
-- Use fix_storage_disable_rls.sql
-- Disables RLS temporarily for development
```

### **Option 3: Manual Dashboard Setup**
1. **Go to Storage** → **Policies**
2. **Create policies manually** for each bucket
3. **Allow authenticated users** full access

## 🎯 **Expected Results:**

After running either fix:
- ✅ Avatar uploads work
- ✅ Post image uploads work
- ✅ No more RLS policy errors
- ✅ Images are publicly accessible

## 🔄 **If Still Not Working:**

1. **Clear browser cache**
2. **Sign out and sign back in**
3. **Try in incognito mode**
4. **Check browser console for detailed errors**

**Try the simple fix first, then disable RLS if needed!** 🎉 