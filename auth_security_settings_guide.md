# Auth Security Settings Guide

## Remaining Issues to Fix

You have 2 warnings remaining that need to be fixed manually in the Supabase Dashboard:

### 1. Leaked Password Protection Disabled

**Issue:** Supabase Auth prevents the use of compromised passwords by checking against HaveIBeenPwned.org, but this feature is currently disabled.

**How to Fix:**
1. Go to your Supabase Dashboard
2. Navigate to **Authentication** → **Settings**
3. Scroll down to **Password Security**
4. Enable **"Leaked password protection"**
5. Click **Save**

**Benefits:**
- Prevents users from using passwords that have been compromised in data breaches
- Enhances overall security posture
- Uses HaveIBeenPwned.org database for checking

### 2. Insufficient MFA Options

**Issue:** Your project has too few multi-factor authentication (MFA) options enabled, which may weaken account security.

**How to Fix:**
1. Go to your Supabase Dashboard
2. Navigate to **Authentication** → **Settings**
3. Scroll down to **Multi-Factor Authentication**
4. Enable additional MFA methods:
   - **TOTP (Time-based One-Time Password)** - Most common
   - **SMS** - For phone-based verification
   - **Email** - For email-based verification
5. Configure each method as needed
6. Click **Save**

**Recommended MFA Methods:**
- **TOTP** (Google Authenticator, Authy, etc.) - Most secure
- **SMS** - Good for user convenience
- **Email** - Good backup option

## Database Fixes

### Run the Jobs Table Check Script

1. Copy the contents of `check_and_fix_jobs_table.sql`
2. Paste into Supabase SQL Editor
3. Execute the script

This will:
- Check if the `jobs` table exists
- Enable RLS if it exists
- Create appropriate policies
- Show you the current RLS status

## Expected Results

After completing these steps:

✅ **RLS Disabled in Public** - Should be resolved (if jobs table exists)
✅ **Leaked Password Protection** - Should be resolved
✅ **Insufficient MFA Options** - Should be resolved

## Verification Steps

1. **Check Security Advisor** - Go back to Security Advisor and click "Rerun linter"
2. **Test Authentication** - Try signing up/signing in to ensure MFA works
3. **Monitor Logs** - Check Authentication logs for any issues

## Additional Security Recommendations

### 1. Password Policy
Consider implementing a strong password policy:
- Minimum 8 characters
- Require uppercase, lowercase, numbers, and symbols
- Prevent common passwords

### 2. Session Management
- Set appropriate session timeouts
- Enable secure session handling
- Consider implementing session rotation

### 3. Rate Limiting
- Enable rate limiting for authentication attempts
- Set up alerts for suspicious activity
- Monitor failed login attempts

### 4. Audit Logging
- Enable comprehensive audit logging
- Monitor authentication events
- Set up alerts for security events

## Support

If you encounter any issues:
1. Check the Supabase documentation for Auth settings
2. Verify your configuration matches the recommendations
3. Test the authentication flow thoroughly
4. Monitor the Security Advisor for any remaining issues 