# Follow Functionality Test Guide

## Database Setup
1. Run the `setup_user_follows_table.sql` script in your Supabase SQL Editor
2. Verify the table was created with proper constraints and RLS policies

## Frontend Components Created
1. **FollowButton** - Handles follow/unfollow actions
2. **FollowersList** - Shows list of users following a profile
3. **FollowingList** - Shows list of users a profile is following
4. **ProfileClient** - Main profile page component with tabs

## Testing Steps

### 1. Database Verification
```sql
-- Check if table exists
SELECT * FROM information_schema.tables WHERE table_name = 'user_follows';

-- Check table structure
SELECT column_name, data_type FROM information_schema.columns 
WHERE table_name = 'user_follows';

-- Check RLS policies
SELECT policyname, cmd FROM pg_policies WHERE tablename = 'user_follows';
```

### 2. Frontend Testing
1. **Follow Button Test**:
   - Visit any user profile page (not your own)
   - Click the "Follow" button
   - Verify button changes to "Following"
   - Click again to unfollow
   - Verify button changes back to "Follow"

2. **Followers List Test**:
   - Visit a profile page
   - Click the "Followers" tab
   - Verify the list loads correctly
   - Check that clicking on a follower navigates to their profile

3. **Following List Test**:
   - Visit a profile page
   - Click the "Following" tab
   - Verify the list loads correctly
   - Check that clicking on a followed user navigates to their profile

4. **About Tab Test**:
   - Visit a profile page
   - Click the "About" tab
   - Verify profile information displays correctly

### 3. Edge Cases to Test
- Follow button doesn't appear on your own profile
- Follow button works for both individual and NGO profiles
- Lists handle empty states correctly
- Error states are handled gracefully
- Loading states work properly

### 4. Database Relationship Verification
```sql
-- Test follow relationship
INSERT INTO user_follows (follower_id, following_id) 
VALUES ('user1-uuid', 'user2-uuid');

-- Verify relationship
SELECT * FROM user_follows WHERE follower_id = 'user1-uuid' AND following_id = 'user2-uuid';

-- Test unfollow
DELETE FROM user_follows WHERE follower_id = 'user1-uuid' AND following_id = 'user2-uuid';
```

## Expected Behavior
- Follow button shows "Follow" for unfollowed users
- Follow button shows "Following" for followed users
- Followers count updates in real-time
- Lists show correct user information
- Navigation between profiles works correctly
- All accessibility features work (ARIA labels, keyboard navigation)

## Troubleshooting
If issues occur:
1. Check browser console for errors
2. Verify database connections
3. Check RLS policies are working
4. Ensure user authentication is working
5. Verify table relationships are correct 