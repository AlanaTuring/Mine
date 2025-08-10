"use client"

import type React from "react"

import { useEffect, useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Checkbox } from "@/components/ui/checkbox"
import { Eye, EyeOff, Shield, Accessibility, Volume2, VolumeX, Type, Contrast, MousePointer, Moon } from "lucide-react"
import { getSupabaseClient } from '@/lib/supabaseClient';
import { useRouter } from 'next/navigation';
import { Textarea } from "@/components/ui/textarea";
import { useAccessibility } from "@/components/accessibility-provider";

export function AuthPage() {
  const [hydrated, setHydrated] = useState(false);
  const [isSignUp, setIsSignUp] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "",
    agreeToTerms: false,
    accessibilityNeeds: [] as string[],
    accountType: "Individual",
    organizationName: "",
    organizationWebsite: "",
    headquartersLocation: "",
    missionStatement: "",
    accessibilityFeatures: [] as string[],
  })
  const [audioEnabled, setAudioEnabled] = useState(false)
  const [largeText, setLargeText] = useState(false)
  const { settings, updateSettings } = useAccessibility();
  const highContrast = settings.highContrast;
  const [emailError, setEmailError] = useState("");

  const router = useRouter();

  useEffect(() => { setHydrated(true); }, []);
  if (!hydrated) return null;

  const handleToggleSignUp = () => {
    setIsSignUp((prev) => {
      const next = !prev;
      setFormData((f) => ({ ...f, confirmPassword: "" }));
      return next;
    });
  };

  const accessibilityOptions = [
    { id: "visual", label: "Visual Impairment", icon: Eye },
    { id: "hearing", label: "Hearing Impairment", icon: Volume2 },
    { id: "mobility", label: "Mobility Impairment", icon: MousePointer },
    { id: "cognitive", label: "Cognitive Disability", icon: Type },
    { id: "other", label: "Other", icon: Accessibility },
  ]

  const handleAccessibilityToggle = (optionId: string) => {
    setFormData((prev) => ({
      ...prev,
      accessibilityNeeds: prev.accessibilityNeeds.includes(optionId)
        ? prev.accessibilityNeeds.filter((id) => id !== optionId)
        : [...prev.accessibilityNeeds, optionId],
    }))
  }

  // Mock users array for duplicate email check
  const mockUsers = [
    { email: "test@example.com" },
    { email: "jane@company.com" },
    // Add more mock users as needed
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setEmailError(""); // Reset error on submit
    console.log('Form submitted. isSignUp:', isSignUp);

    if (isSignUp) {
      // Password match validation
      if (formData.password !== formData.confirmPassword) {
        alert('Passwords do not match.');
        return;
      }
      // Duplicate email check (mock)
      const emailExists = mockUsers.some(u => u.email.toLowerCase() === formData.email.toLowerCase());
      if (emailExists) {
        setEmailError("This email is already registered. Please use a different one.");
        return;
      }

      const supabase = getSupabaseClient();
      let accountType = formData.accountType;
      if (accountType === "NGO / Organization") accountType = "NGO";

      const { data, error } = await supabase.auth.signUp({
        email: formData.email,
        password: formData.password,
        options: {
          data: {
            first_name: formData.firstName,
            last_name: formData.lastName,
            account_type: accountType,
            organization_name: formData.organizationName,
            organization_website: formData.organizationWebsite,
            headquarters_location: formData.headquartersLocation,
            mission_statement: formData.missionStatement,
            accessibility_needs: formData.accessibilityNeeds,
            accessibility_features: formData.accessibilityFeatures,
          }
        }
      });

      if (error) {
        console.error('Sign up error:', error);
        alert('Error creating account. Please try again.');
        return;
      }

      // Create profile record
      if (data.user) {
        const { error: profileError } = await supabase
          .from('profiles')
          .insert([
            {
              id: data.user.id,
              first_name: formData.firstName,
              last_name: formData.lastName,
              email: formData.email,
              account_type: accountType,
              organization_name: formData.organizationName,
              organization_website: formData.organizationWebsite,
              headquarters_location: formData.headquartersLocation,
              mission_statement: formData.missionStatement,
              accessibility_needs: formData.accessibilityNeeds,
              accessibility_features: formData.accessibilityFeatures,
            }
          ]);

        if (profileError) {
          console.error('Profile creation error:', profileError);
        }
      }

      alert('Account created successfully! Please check your email to verify your account.');
      setIsSignUp(false);
    } else {
      // Sign in
      const supabase = getSupabaseClient();
      const { data, error } = await supabase.auth.signInWithPassword({
        email: formData.email,
        password: formData.password,
      });

      if (error) {
        console.error('Sign in error:', error);
        alert('Invalid email or password. Please try again.');
        return;
      }

      console.log('Redirecting to dashboard...');
      router.push('/dashboard');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background dark:bg-black">
      {/* Accessibility Toolbar */}
      <div className="fixed top-4 right-4 z-50 flex space-x-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => updateSettings({ highContrast: !highContrast })}
          aria-label="Toggle dark mode"
          className="bg-card border border-border text-foreground hover:bg-accent dark:bg-card dark:border-border dark:text-foreground dark:shadow-md"
        >
          <Moon className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setLargeText(!largeText)}
          aria-label="Toggle large text"
          className="bg-card border border-border text-foreground hover:bg-accent dark:bg-card dark:border-border dark:text-foreground dark:shadow-md"
        >
          <Type className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setAudioEnabled(!audioEnabled)}
          aria-label="Toggle audio assistance"
          className="bg-card border border-border text-foreground hover:bg-accent dark:bg-card dark:border-border dark:text-foreground dark:shadow-md"
        >
          {audioEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
        </Button>
      </div>

      <div className="w-full max-w-md mx-auto px-4">
        <Card className="shadow-2xl border border-border bg-card">
          <CardContent className="p-8">
            {/* Logo and Title */}
            <div className="text-center mb-8">
              <div className="flex items-center justify-center space-x-3 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-mustard to-forest-green rounded-full flex items-center justify-center">
                  <Accessibility className="h-6 w-6 text-white" />
                </div>
                <h1 className={`text-3xl font-bold bg-gradient-to-r from-mustard to-forest-green bg-clip-text text-transparent ${largeText ? "text-4xl" : ""}`}>AccessAble</h1>
              </div>
              <h2 className={`text-2xl font-bold mb-2 text-foreground ${largeText ? "text-3xl" : ""}`}>{isSignUp ? "Create Account" : "Welcome Back"}</h2>
              <p className={`text-sm text-muted-foreground ${largeText ? "text-base" : ""}`}>{isSignUp ? "Join our inclusive professional community" : "Sign in to your account"}</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-8">
              {isSignUp ? (
                <>
                  <div className="space-y-3">
                    <label htmlFor="accountType" className="text-sm font-medium text-foreground">Account Type</label>
                    <select
                      id="accountType"
                      value={formData.accountType}
                      onChange={e => setFormData(prev => ({ ...prev, accountType: e.target.value }))}
                      className="w-full border-2 border-input rounded-lg px-4 py-3 text-lg focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors bg-background text-foreground"
                      style={{ fontSize: '1.125rem' }} // 18px
                    >
                      <option value="Individual" className="text-lg">Individual</option>
                      <option value="NGO" className="text-lg">NGO / Organization</option>
                    </select>
                  </div>
                  {formData.accountType === "NGO" ? (
                    <div className="space-y-6 p-6 bg-muted rounded-lg border border-border">
                      <h3 className="text-lg font-medium text-foreground">Organization Details</h3>
                      <div className="space-y-4">
                        <div className="space-y-3">
                          <label htmlFor="organizationName" className="text-base font-medium text-foreground">Organization Name *</label>
                          <Input
                            id="organizationName"
                            value={formData.organizationName}
                            onChange={e => setFormData(prev => ({ ...prev, organizationName: e.target.value }))}
                            className="w-full border-2 border-input rounded-lg px-4 py-3 text-base focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors bg-background text-foreground"
                            placeholder="Enter organization name"
                            required
                          />
                        </div>
                        <div className="space-y-3">
                          <label htmlFor="email" className="text-base font-medium text-foreground">Email Address *</label>
                          <Input
                            id="email"
                            type="email"
                            required
                            value={formData.email}
                            onChange={e => {
                              setFormData((prev) => ({ ...prev, email: e.target.value }));
                              setEmailError(""); // Clear error on change
                            }}
                            className="w-full border-2 border-input rounded-lg px-4 py-3 text-base focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors bg-background text-foreground"
                            placeholder="Enter your email address"
                          />
                        </div>
                        {emailError && (
                          <div className="text-destructive text-sm mt-1" role="alert">{emailError}</div>
                        )}
                        <div className="space-y-3">
                          <label htmlFor="password" className="text-base font-medium text-foreground">Password *</label>
                          <Input
                            id="password"
                            type="password"
                            required
                            value={formData.password}
                            onChange={e => setFormData((prev) => ({ ...prev, password: e.target.value }))}
                            className="w-full border-2 border-input rounded-lg px-4 py-3 text-base focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors bg-background text-foreground"
                            placeholder="Enter your password"
                          />
                        </div>
                        <div className="space-y-3">
                          <label htmlFor="confirmPassword" className="text-base font-medium text-foreground">Confirm Password *</label>
                          <Input
                            id="confirmPassword"
                            type="password"
                            required={isSignUp}
                            value={formData.confirmPassword}
                            onChange={e => setFormData((prev) => ({ ...prev, confirmPassword: e.target.value }))}
                            className="w-full border-2 border-input rounded-lg px-4 py-3 text-base focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors bg-background text-foreground"
                            placeholder="Confirm your password"
                          />
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-3">
                          <label htmlFor="firstName" className="text-base font-medium text-foreground">First Name *</label>
                          <Input
                            id="firstName"
                            type="text"
                            required={isSignUp}
                            value={formData.firstName}
                            onChange={(e) => setFormData((prev) => ({ ...prev, firstName: e.target.value }))}
                            className="w-full border-2 border-input rounded-lg px-4 py-3 text-base focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors bg-background text-foreground"
                            placeholder="Enter your first name"
                            disabled={!isSignUp}
                          />
                        </div>
                        <div className="space-y-3">
                          <label htmlFor="lastName" className="text-base font-medium text-foreground">Last Name *</label>
                          <Input
                            id="lastName"
                            type="text"
                            required={isSignUp}
                            value={formData.lastName}
                            onChange={(e) => setFormData((prev) => ({ ...prev, lastName: e.target.value }))}
                            className="w-full border-2 border-input rounded-lg px-4 py-3 text-base focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors bg-background text-foreground"
                            placeholder="Enter your last name"
                            disabled={!isSignUp}
                          />
                        </div>
                      </div>
                      <div className="space-y-3">
                        <label htmlFor="email" className="text-base font-medium text-foreground">Email Address *</label>
                        <Input
                          id="email"
                          type="email"
                          required
                          value={formData.email}
                          onChange={e => {
                            setFormData((prev) => ({ ...prev, email: e.target.value }));
                            setEmailError(""); // Clear error on change
                          }}
                          className="w-full border-2 border-input rounded-lg px-4 py-3 text-base focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors bg-background text-foreground"
                          placeholder="Enter your email address"
                        />
                      </div>
                      {emailError && (
                        <div className="text-destructive text-sm mt-1" role="alert">{emailError}</div>
                      )}
                      <div className="space-y-3">
                        <label htmlFor="password" className="text-base font-medium text-foreground">Password *</label>
                        <Input
                          id="password"
                          type="password"
                          required
                          value={formData.password}
                          onChange={e => setFormData((prev) => ({ ...prev, password: e.target.value }))}
                          className="w-full border-2 border-input rounded-lg px-4 py-3 text-base focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors bg-background text-foreground"
                          placeholder="Enter your password"
                        />
                      </div>
                      {isSignUp && (
                        <div className="space-y-3">
                          <label htmlFor="confirmPassword" className="text-base font-medium text-foreground">Confirm Password *</label>
                          <Input
                            id="confirmPassword"
                            type="password"
                            required
                            value={formData.confirmPassword}
                            onChange={e => setFormData((prev) => ({ ...prev, confirmPassword: e.target.value }))}
                            className="w-full border-2 border-input rounded-lg px-4 py-3 text-base focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors bg-background text-foreground"
                            placeholder="Confirm your password"
                          />
                        </div>
                      )}
                    </div>
                  )}

                  {/* Accessibility Features for Organizations */}
                  {(formData.accountType === "Employer" || formData.accountType === "NGO") && (
                    <div className="space-y-4 p-6 bg-muted rounded-lg border border-border">
                      <h3 className="text-lg font-medium text-foreground">Accessibility Features</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {["Screen Reader Support", "Sign Language Support", "Assistive Technology Provided", "Remote Work Options", "Flexible Hours", "Accessible Office Space"].map(feature => (
                          <label key={feature} className="flex items-center space-x-3 p-3 bg-background rounded-lg border border-border hover:bg-accent cursor-pointer transition-colors">
                            <input
                              type="checkbox"
                              checked={formData.accessibilityFeatures.includes(feature)}
                              onChange={e => {
                                setFormData(prev => ({
                                  ...prev,
                                  accessibilityFeatures: e.target.checked
                                    ? [...prev.accessibilityFeatures, feature]
                                    : prev.accessibilityFeatures.filter(f => f !== feature)
                                }))
                              }}
                              className="h-4 w-4 text-mustard focus:ring-mustard border-input rounded"
                            />
                            <span className="text-sm font-medium text-foreground">{feature}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Always show sign-in/sign-up fields */}
                  {/* Remove the extra Confirm Password input so only one remains in the form for both account types. */}

                  {/* Accessibility Needs Section */}
                  <div className="space-y-4 p-6 bg-muted rounded-lg border border-border">
                    <h3 className="text-lg font-medium text-foreground">Accessibility Needs (Optional)</h3>
                    <p className="text-sm text-muted-foreground">Help us personalize your experience by sharing your accessibility needs.</p>
                    <div className="grid grid-cols-1 gap-3">
                      {accessibilityOptions.map((option) => (
                        <div
                          key={option.id}
                          className={`flex items-center space-x-3 p-4 rounded-lg border cursor-pointer transition-all ${
                            (formData.accessibilityNeeds || []).includes(option.id)
                              ? "bg-mustard/10 border-mustard"
                              : "bg-background border-border hover:bg-accent"
                          }`}
                        >
                          <Checkbox
                            checked={(formData.accessibilityNeeds || []).includes(option.id)}
                            onCheckedChange={() => handleAccessibilityToggle(option.id)}
                            aria-label={`Select ${option.label}`}
                            className="h-4 w-4 text-mustard focus:ring-mustard border-input rounded"
                          />
                          <option.icon
                            className={`h-5 w-5 ${
                              (formData.accessibilityNeeds || []).includes(option.id)
                                ? "text-mustard"
                                : "text-muted-foreground"
                            }`}
                          />
                          <span className="text-sm font-medium text-foreground">
                            {option.label}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-start space-x-3 p-4 bg-muted rounded-lg border border-border">
                    <Checkbox
                      id="terms"
                      checked={formData.agreeToTerms}
                      onCheckedChange={(checked) =>
                        setFormData((prev) => ({ ...prev, agreeToTerms: checked as boolean }))
                      }
                      className="mt-1 h-4 w-4 text-mustard focus:ring-mustard border-input rounded"
                      required
                    />
                    <label
                      htmlFor="terms"
                      className="text-sm leading-relaxed cursor-pointer text-foreground"
                    >
                      I agree to the{" "}
                      <a href="#" className="text-mustard hover:underline font-medium">
                        Terms of Service
                      </a>{" "}
                      and{" "}
                      <a href="#" className="text-mustard hover:underline font-medium">
                        Privacy Policy
                      </a>
                    </label>
                  </div>

                  <Button
                    type="submit"
                    className="w-full bg-gradient-to-r from-mustard to-forest-green hover:from-mustard/90 hover:to-forest-green/90 text-white font-medium py-4 px-6 rounded-lg transition-all duration-200 transform hover:scale-[1.02] focus:ring-2 focus:ring-mustard/20"
                  >
                    Create Account
                  </Button>
                </>
              ) : (
                <>
                  <div className="space-y-3">
                    <label htmlFor="email" className="text-sm font-medium text-foreground">Email Address *</label>
                    <Input
                      id="email"
                      type="email"
                      required
                      value={formData.email}
                      onChange={(e) => setFormData((prev) => ({ ...prev, email: e.target.value }))}
                      className="w-full border-2 border-input rounded-lg px-4 py-3 text-base focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors bg-background text-foreground"
                      placeholder="Enter your email address"
                    />
                  </div>
                  <div className="space-y-3">
                    <label htmlFor="password" className="text-sm font-medium text-foreground">Password *</label>
                    <div className="relative">
                      <Input
                        id="password"
                        type={showPassword ? "text" : "password"}
                        required
                        value={formData.password}
                        onChange={(e) => setFormData((prev) => ({ ...prev, password: e.target.value }))}
                        className="w-full border-2 border-input rounded-lg px-4 py-3 text-base focus:border-mustard focus:ring-2 focus:ring-mustard/20 transition-colors pr-12 bg-background text-foreground"
                        placeholder="Enter your password"
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
                  <Button
                    type="submit"
                    className="w-full bg-gradient-to-r from-mustard to-forest-green hover:from-mustard/90 hover:to-forest-green/90 text-white font-medium py-4 px-6 rounded-lg transition-all duration-200 transform hover:scale-[1.02] focus:ring-2 focus:ring-mustard/20"
                  >
                    Sign In
                  </Button>
                  <div className="text-center">
                    <a href="#" className="text-sm text-mustard hover:underline font-medium">
                      Forgot your password?
                    </a>
                  </div>
                </>
              )}
            </form>

            <div className="mt-8 text-center">
              <p className="text-sm text-muted-foreground">
                {isSignUp ? "Already have an account?" : "Don't have an account?"}
              </p>
              <Button
                type="button"
                variant="ghost"
                onClick={handleToggleSignUp}
                className="mt-2 text-mustard hover:text-forest-green font-medium hover:bg-mustard/10 px-4 py-2 rounded-lg transition-colors"
              >
                {isSignUp ? "Sign In" : "Create Account"}
              </Button>
            </div>

            {/* Accessibility Statement */}
            <div className="mt-6 p-6 rounded-lg bg-muted border border-border">
              <div className="flex items-center space-x-3 mb-3">
                <Shield className="h-5 w-5 text-mustard" />
                <span className="text-sm font-medium text-foreground">Accessibility Commitment</span>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">AccessAble is committed to providing an accessible experience for all users. If you encounter any accessibility barriers, please contact our support team.</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
