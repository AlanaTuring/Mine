# 🧪 Test Setup Guide

## Step 1: Run Database Fixes

1. **Go to Supabase Dashboard** → **SQL Editor**
2. **Run the profile creation fix:**
   ```sql
   -- Copy and paste the content from fix_profile_creation.sql
   ```
3. **Run the complete database setup (if not done):**
   ```sql
   -- Copy and paste the content from complete_database_setup.sql
   ```

## Step 2: Create Storage Buckets

1. **Go to Storage** in Supabase Dashboard
2. **Create bucket:** `avatars` (Public)
3. **Create bucket:** `post-media` (Public)

## Step 3: Test Account Creation

1. **Go to** `localhost:3000/auth`
2. **Fill out the signup form:**
   - First Name: Test
   - Last Name: User
   - Email: test@example.com
   - Password: password123
   - Check "I agree to Terms"
3. **Click "Create Account"**
4. **Expected Result:** ✅ Success message, no errors!

## Step 4: Test Navigation

1. **After signup, you should be redirected to dashboard**
2. **Click "Me" button** → Should show dropdown
3. **Click "View Profile"** → Should go to profile page
4. **Click "Back to Dashboard"** → Should return to dashboard
5. **Click "AccessAble" logo** → Should go to dashboard

## Step 5: Test Profile Creation

1. **Check Supabase Dashboard** → **Table Editor** → **profiles**
2. **You should see a new profile** with your test user data
3. **The profile should have:**
   - id: matches auth.users id
   - first_name: "Test"
   - last_name: "User"
   - email: "test@example.com"

## 🎯 What's Fixed:

✅ **Foreign Key Constraint Error** - Trigger automatically creates profiles
✅ **RLS Policy Issues** - Proper policies for profile access
✅ **Navigation** - Complete dashboard ↔ profile navigation
✅ **User Experience** - Smooth signup flow

## 🚨 If Still Having Issues:

1. **Clear browser cache** and try again
2. **Check Supabase logs** for any errors
3. **Verify all SQL scripts ran successfully**
4. **Make sure storage buckets are created**

The trigger approach is the most reliable way to handle profile creation in Supabase! 🎉 