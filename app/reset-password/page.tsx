'use client';

import { useEffect, useState, Suspense } from 'react';
import { useRouter } from 'next/navigation';
import { getSupabaseClient } from '@/lib/supabaseClient';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Eye, EyeOff, Accessibility, Shield } from 'lucide-react';
import { useAccessibility } from '@/components/accessibility-provider';

function ResetPasswordForm() {
  const router = useRouter();
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [isValidToken, setIsValidToken] = useState(false);
  const { settings } = useAccessibility();
  const highContrast = settings.highContrast;

  useEffect(() => {
    // Check if we have the necessary parameters from the reset link
    if (typeof window !== 'undefined') {
      const urlParams = new URLSearchParams(window.location.search);
      const accessToken = urlParams.get('access_token');
      const refreshToken = urlParams.get('refresh_token');
      const type = urlParams.get('type');
      
      console.log('URL Parameters:', { accessToken, refreshToken, type });
      
      // Check for hash fragment parameters (Supabase sometimes uses these)
      const hashParams = new URLSearchParams(window.location.hash.substring(1));
      const hashAccessToken = hashParams.get('access_token');
      const hashRefreshToken = hashParams.get('refresh_token');
      
      console.log('Hash Parameters:', { hashAccessToken, hashRefreshToken });
      
      // Use either search params or hash params
      const finalAccessToken = accessToken || hashAccessToken;
      const finalRefreshToken = refreshToken || hashRefreshToken;
      
      if (!finalAccessToken) {
        console.log('No access token found in URL');
        setError('Invalid or expired reset link. Please request a new password reset.');
        setIsValidToken(false);
      } else {
        console.log('Access token found, proceeding with password reset');
        setIsValidToken(true);
        setError('');
      }
    }
  }, []);

  const handlePasswordReset = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    // Validate passwords
    if (newPassword.length < 6) {
      setError('Password must be at least 6 characters long.');
      setLoading(false);
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('Passwords do not match.');
      setLoading(false);
      return;
    }

    try {
      const supabase = getSupabaseClient();
      
      // Update the user's password
      const { error } = await supabase.auth.updateUser({
        password: newPassword
      });

      if (error) {
        console.error('Password update error:', error);
        setError(`Error updating password: ${error.message}`);
      } else {
        setSuccess(true);
        // Redirect to login page after 3 seconds
        setTimeout(() => {
          router.push('/auth');
        }, 3000);
      }
    } catch (err) {
      console.error('Password reset error:', err);
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Card className="w-full max-w-md mx-auto px-4">
          <CardContent className="p-8 text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-forest-green to-mustard rounded-full flex items-center justify-center mx-auto mb-6">
              <Shield className="h-8 w-8 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-foreground mb-4">Password Updated Successfully!</h1>
            <p className="text-muted-foreground mb-6">
              Your password has been reset. You will be redirected to the login page shortly.
            </p>
            <Button
              onClick={() => router.push('/auth')}
              className="w-full bg-gradient-to-r from-mustard to-forest-green hover:from-mustard/90 hover:to-forest-green/90 text-white font-medium py-3 px-6 rounded-lg"
            >
              Go to Login
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <Card className="w-full max-w-md mx-auto px-4">
        <CardContent className="p-8">
          {/* Logo and Title */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center space-x-3 mb-6">
              <div className="w-12 h-12 bg-gradient-to-r from-mustard to-forest-green rounded-full flex items-center justify-center">
                <Accessibility className="h-6 w-6 text-white" />
              </div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-mustard to-forest-green bg-clip-text text-transparent">
                AccessAble
              </h1>
            </div>
            <h2 className="text-2xl font-bold mb-2 text-foreground">Reset Your Password</h2>
            <p className="text-sm text-muted-foreground">
              Enter your new password below
            </p>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-600 text-sm">{error}</p>
              <div className="mt-3">
                <Button
                  onClick={() => router.push('/auth')}
                  variant="outline"
                  size="sm"
                  className="text-red-600 border-red-300 hover:bg-red-50"
                >
                  Back to Login
                </Button>
              </div>
            </div>
          )}

          {!error && isValidToken && (
            <form onSubmit={handlePasswordReset} className="space-y-6">
              <div className="space-y-3">
                <label htmlFor="newPassword" className="text-sm font-medium text-foreground">
                  New Password *
                </label>
                <div className="relative">
                  <Input
                    id="newPassword"
                    type={showPassword ? "text" : "password"}
                    required
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    className="w-full border-2 border-input rounded-lg px-4 py-3 text-base focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors pr-12 bg-background text-foreground"
                    placeholder="Enter your new password"
                    minLength={6}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-3 top-1/2 -translate-y-1/2 h-8 w-8 p-0 hover:bg-accent rounded text-foreground"
                    onClick={() => setShowPassword(!showPassword)}
                    aria-label={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </Button>
                </div>
              </div>

              <div className="space-y-3">
                <label htmlFor="confirmPassword" className="text-sm font-medium text-foreground">
                  Confirm New Password *
                </label>
                <div className="relative">
                  <Input
                    id="confirmPassword"
                    type={showConfirmPassword ? "text" : "password"}
                    required
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full border-2 border-input rounded-lg px-4 py-3 text-base focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors pr-12 bg-background text-foreground"
                    placeholder="Confirm your new password"
                    minLength={6}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-3 top-1/2 -translate-y-1/2 h-8 w-8 p-0 hover:bg-accent rounded text-foreground"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    aria-label={showConfirmPassword ? "Hide password" : "Show password"}
                  >
                    {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </Button>
                </div>
              </div>

              <Button
                type="submit"
                disabled={loading}
                className="w-full bg-gradient-to-r from-mustard to-forest-green hover:from-mustard/90 hover:to-forest-green/90 text-white font-medium py-4 px-6 rounded-lg transition-all duration-200 transform hover:scale-[1.02] focus:ring-2 focus:ring-mustard/20 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Updating Password...' : 'Update Password'}
              </Button>
            </form>
          )}

          <div className="mt-6 text-center">
            <Button
              type="button"
              variant="ghost"
              onClick={() => router.push('/auth')}
              className="text-mustard hover:text-forest-green font-medium hover:bg-mustard/10 px-4 py-2 rounded-lg transition-colors"
            >
              Back to Login
            </Button>
          </div>

          {/* Accessibility Statement */}
          <div className="mt-6 p-6 rounded-lg bg-muted border border-border">
            <div className="flex items-center space-x-3 mb-3">
              <Shield className="h-5 w-5 text-mustard" />
              <span className="text-sm font-medium text-foreground">Accessibility Commitment</span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              AccessAble is committed to providing an accessible experience for all users. 
              If you encounter any accessibility barriers, please contact our support team.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-mustard mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    }>
      <ResetPasswordForm />
    </Suspense>
  );
}
