# Feature Testing Guide

## 🧪 Testing Instructions

### 1. **Persistent Like Functionality Test**

**Steps:**
1. Go to the dashboard (`/dashboard`)
2. Find a post and click the like button (heart icon)
3. Verify the heart turns filled and the count increases
4. Refresh the page
5. Verify the heart remains filled and the count is correct
6. Click the like button again to unlike
7. Verify the heart becomes unfilled and count decreases
8. Refresh the page
9. Verify the heart remains unfilled

**Expected Results:**
- ✅ Like state persists after refresh
- ✅ Like count updates correctly
- ✅ Can only like once per post
- ✅ Can unlike posts

### 2. **Comment Username Test**

**Steps:**
1. Go to the dashboard
2. Find a post and add a comment
3. Verify the comment shows YOUR username, not the post owner's
4. Add another comment from a different account
5. Verify each comment shows the correct commenter's username

**Expected Results:**
- ✅ Comments show the commenter's username
- ✅ Not the post owner's username

### 3. **Comment Edit/Delete Test**

**Steps:**
1. Add a comment to any post
2. Verify you see edit and delete buttons on your comment
3. Click edit and modify the comment
4. Save the edit
5. Verify the comment is updated
6. Delete your comment
7. Verify the comment is removed

**Expected Results:**
- ✅ Can edit own comments
- ✅ Can delete own comments
- ✅ Edit/delete buttons only appear on own comments

### 4. **Post Owner Comment Management Test**

**Steps:**
1. Create a post from your account
2. Have another user comment on your post
3. Verify you can see delete button on their comment (as post owner)
4. Delete their comment
5. Verify the comment is removed

**Expected Results:**
- ✅ Post owner can delete any comment on their post
- ✅ Post owner sees delete button on all comments

### 5. **Job Post Edit Test**

**Steps:**
1. Go to `/jobs`
2. Create a job post with all fields
3. Click edit on your job post
4. Modify all fields (title, content, company, location, etc.)
5. Save changes
6. Verify all changes are saved

**Expected Results:**
- ✅ Can edit all job metadata fields
- ✅ Changes persist after save

## 🔧 Debugging

### Check Browser Console
Open browser developer tools (F12) and check the console for:
- Like status checking logs
- Comment creation logs
- Any error messages

### Database Verification
Run this SQL in Supabase SQL Editor to verify tables exist:
```sql
-- Check if likes table exists
SELECT * FROM information_schema.tables WHERE table_name = 'likes';

-- Check if comments table exists  
SELECT * FROM information_schema.tables WHERE table_name = 'comments';

-- Check RLS policies
SELECT tablename, policyname FROM pg_policies WHERE tablename IN ('likes', 'comments', 'posts');
```

## 🚨 Common Issues

1. **Likes not persisting**: Check if `likes` table exists and has proper RLS policies
2. **Comments showing wrong username**: Check if `profiles` table has correct user data
3. **Edit/delete buttons not showing**: Check if `currentUserId` is being passed correctly
4. **Permission errors**: Run the `run_database_fixes.sql` script in Supabase

## 📝 Test Results

After running all tests, note any failures here:
- [ ] Like functionality works
- [ ] Comment usernames are correct  
- [ ] Comment edit/delete works
- [ ] Post owner can delete comments
- [ ] Job editing works 