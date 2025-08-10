"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Mail,
  Phone,
  Globe,
  Plus,
  X,
  Camera,
  Accessibility,
  Eye,
  Volume2,
  MousePointer,
  Type,
  Contrast,
  VolumeX,
  Shield,
  Save,
  Edit3,
  Linkedin,
  Twitter,
  Github,
  Loader2,
} from "lucide-react"
import { getSupabaseClient } from '@/lib/supabaseClient';
import { useEffect } from 'react';
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAccessibility } from "@/components/accessibility-provider";
import { UserProfileSidebar } from "@/components/profile/UserProfileSidebar";

interface ProfileData {
  firstName: string;
  lastName: string;
  title: string;
  company: string;
  location: string;
  bio: string;
  email: string;
  phone: string;
  website: string;
  linkedin: string;
  twitter: string;
  github: string;
  availability: string;
  workType: string;
  experience: any[]; // or a more specific type if you have one
  education: any[];  // or a more specific type if you have one
  accessibilityNeeds: string[];
  accommodations: string;
  showEmail: boolean;
  showPhone: boolean;
  profileVisibility: string;
  avatarUrl?: string;
  accountType: string;
  organizationName: string;
  headquartersLocation: string;
  missionStatement: string;
  accessibilityFeatures: string[];
  openToCollaboration: boolean;
}

export function ProfileEditPage() {
  const [isEditing, setIsEditing] = useState(false)
  const [audioEnabled, setAudioEnabled] = useState(false)
  const [loading, setLoading] = useState(false);
  const [selectedAvatar, setSelectedAvatar] = useState<File | null>(null);

  // 1. Initialize profileData with empty/default values
  const [profileData, setProfileData] = useState<ProfileData>({
    firstName: "",
    lastName: "",
    title: "",
    company: "",
    location: "",
    bio: "",
    email: "",
    phone: "",
    website: "",
    linkedin: "",
    twitter: "",
    github: "",
    availability: "",
    workType: "",
    experience: [],
    education: [],
    accessibilityNeeds: [],
    accommodations: "",
    showEmail: false,
    showPhone: false,
    profileVisibility: "public",
    avatarUrl: "",
    accountType: "Individual",
    organizationName: "",
    headquartersLocation: "",
    missionStatement: "",
    accessibilityFeatures: [],
    openToCollaboration: false,
  });

  const [newSkill, setNewSkill] = useState("")
  const [profile, setProfile] = useState({
    name: "Jordan Smith",
    title: "Software Engineer",
    location: "San Francisco, CA",
    about:
      "Passionate software engineer with a focus on building accessible and inclusive web applications. Dedicated to creating technology that empowers everyone.",
    email: "jordan.smith@example.com",
    phone: "+1 (555) 123-4567",
    avatar: "/placeholder.svg?height=120&width=128",
  })

  const router = useRouter();
  const [profileSaved, setProfileSaved] = useState(false);
  const { settings } = useAccessibility();
  const isDarkMode = settings.highContrast; // Using highContrast setting for dark mode

  const accessibilityOptions = [
    { id: "visual", label: "Visual Impairment", icon: Eye },
    { id: "hearing", label: "Hearing Impairment", icon: Volume2 },
    { id: "mobility", label: "Mobility Impairment", icon: MousePointer },
    { id: "cognitive", label: "Cognitive Disability", icon: Type },
    { id: "other", label: "Other", icon: Accessibility },
  ]

  // 3. When setting profileData after fetching from Supabase
  useEffect(() => {
    async function checkAuthAndFetchProfile() {
      const supabase = getSupabaseClient();
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        router.push('/auth');
        return;
      }
      // Fetch profile if authenticated
      const { data: profile, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', user.id)
        .single();
      if (profile) {
        setProfileData({
          firstName: profile.first_name || "",
          lastName: profile.last_name || "",
          title: profile.title || "",
          company: profile.company || "",
          location: profile.location || "",
          bio: profile.bio || "",
          email: profile.email || "",
          phone: profile.phone || "",
          website: profile.website || "",
          linkedin: profile.linkedin || "",
          twitter: profile.twitter || "",
          github: profile.github || "",
          availability: profile.availability || "",
          workType: profile.work_type || "",
          experience: profile.experience || [],
          education: profile.education || [],
          accessibilityNeeds: profile.accessibility_needs || [],
          accommodations: profile.accommodations || "",
          showEmail: profile.show_email ?? false,
          showPhone: profile.show_phone ?? false,
          profileVisibility: profile.profile_visibility || "public",
          avatarUrl: profile.avatar_url || "",
          accountType: profile.account_type || "Individual",
          organizationName: profile.organization_name || "",
          headquartersLocation: profile.headquarters_location || "",
          missionStatement: profile.mission_statement || "",
          accessibilityFeatures: profile.accessibility_features || [],
          openToCollaboration: profile.open_to_collaboration ?? false,
        });
        setProfile({
          name: `${profile.first_name || ""} ${profile.last_name || ""}`.trim(),
          title: profile.title || '',
          location: profile.location || '',
          about: profile.bio || '',
          email: profile.email || '',
          phone: profile.phone || '',
          avatar: profile.avatar_url || '',
        });
      }
    }
    checkAuthAndFetchProfile();
  }, [router]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { id, value } = e.target
    setProfile((prev) => ({ ...prev, [id]: value }))
  }

  const handleAvatarUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const supabase = getSupabaseClient();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;

    try {
      // Upload to Supabase Storage
      const fileExt = file.name.split('.').pop();
      const fileName = `${user.id}-${Date.now()}.${fileExt}`;
      
      const { data, error } = await supabase.storage
        .from('avatars')
        .upload(fileName, file);

      if (error) {
        console.error('Storage upload error:', error);
        throw error;
      }

      // Get public URL
      const { data: { publicUrl } } = supabase.storage
        .from('avatars')
        .getPublicUrl(fileName);

      // Update profile with new avatar URL
      setProfileData(prev => ({ ...prev, avatarUrl: publicUrl }));
      setProfile(prev => ({ ...prev, avatar: publicUrl }));

      // Update in database
      const { error: updateError } = await supabase
        .from('profiles')
        .update({ avatar_url: publicUrl })
        .eq('id', user.id);

      if (updateError) {
        console.error('Database update error:', updateError);
        throw updateError;
      }

      alert('Avatar uploaded successfully!');

    } catch (error) {
      console.error('Error uploading avatar:', error);
      alert('Error uploading avatar: ' + (error as any)?.message || 'Unknown error');
    }
  };

  const handleSave = async () => {
    setLoading(true);
    const supabase = getSupabaseClient();
    const { data: { user } } = await supabase.auth.getUser();
    let avatarUrl = profileData.avatarUrl;
    if (user && selectedAvatar) {
      // Upload avatar to Supabase Storage
      const fileExt = selectedAvatar.name.split('.').pop();
      const fileName = `${user.id}-${Date.now()}.${fileExt}`;
      const { data, error } = await supabase.storage
        .from('avatars')
        .upload(fileName, selectedAvatar);
      if (!error) {
        const { data: { publicUrl } } = supabase.storage
          .from('avatars')
          .getPublicUrl(fileName);
        avatarUrl = publicUrl;
      }
    }
    if (user) {
      const { error } = await supabase
        .from('profiles')
        .update({
          first_name: profileData.firstName,
          last_name: profileData.lastName,
          title: profileData.title,
          company: profileData.company,
          location: profileData.location,
          bio: profileData.bio,
          email: profileData.email,
          phone: profileData.phone,
          website: profileData.website,
          linkedin: profileData.linkedin,
          twitter: profileData.twitter,
          github: profileData.github,
          availability: profileData.availability,
          work_type: profileData.workType,
          experience: profileData.experience,
          education: profileData.education,
          accessibility_needs: profileData.accessibilityNeeds,
          accommodations: profileData.accommodations,
          show_email: profileData.showEmail,
          show_phone: profileData.showPhone,
          profile_visibility: profileData.profileVisibility,
          avatar_url: avatarUrl,
          account_type: profileData.accountType,
          organization_name: profileData.organizationName,
          headquarters_location: profileData.headquartersLocation,
          mission_statement: profileData.missionStatement,
          accessibility_features: profileData.accessibilityFeatures,
          open_to_collaboration: profileData.openToCollaboration,
        })
        .eq('id', user.id);
      setLoading(false);
      if (error) {
        alert('Error updating profile: ' + error.message);
      } else {
        alert('Profile updated!');
        setIsEditing(false);
        setProfileSaved(true);
        setTimeout(() => setProfileSaved(false), 1500);
      }
    }
  }



  // 2. Fix handleAccessibilityToggle
  const handleAccessibilityToggle = (optionId: string) => {
    setProfileData((prev) => ({
      ...prev,
      accessibilityNeeds: (prev.accessibilityNeeds || []).includes(optionId)
        ? (prev.accessibilityNeeds || []).filter((id) => id !== optionId)
        : [...(prev.accessibilityNeeds || []), optionId],
    }));
  };

  const isNGO = profileData.accountType === "NGO" || profileData.accountType === "NGO / Organization";

  return (
    <div className="min-h-screen bg-background relative">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/70 z-50">
          <Loader2 className="animate-spin h-8 w-8 text-mustard" />
        </div>
      )}

      {/* Header */}
      <div className="border-b border-border bg-card sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="text-2xl font-bold bg-gradient-to-r from-mustard to-forest-green bg-clip-text text-transparent">
                Access<span className="text-foreground">Able</span>
              </div>
              <h1 className="text-xl font-semibold text-foreground">Profile Settings</h1>
            </div>
            <div className="flex items-center space-x-3">
              {isEditing ? (
                <>
                  <Button variant="outline" onClick={() => setIsEditing(false)} className="border-border text-foreground hover:bg-muted">Cancel</Button>
                  <Button onClick={handleSave} className="bg-gradient-to-r from-mustard to-forest-green hover:from-mustard/90 hover:to-forest-green/90 text-white"><Save className="h-4 w-4 mr-2" />Save Changes</Button>
                </>
              ) : (
                <>
                  <Link href="/dashboard"><Button variant="outline" className="border-border text-foreground hover:bg-muted">Back to Dashboard</Button></Link>
                  <Button onClick={() => setIsEditing(true)} className="bg-mustard text-black font-bold hover:bg-mustard/90 focus:ring-2 focus:ring-mustard focus:ring-offset-2 transition-colors"><Edit3 className="h-4 w-4 mr-2" />Edit Profile</Button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 items-start">
          {/* Left: Profile Picture Card */}
          <UserProfileSidebar
            profileData={profileData}
            showEditButton={false}
            isEditing={isEditing}
            onAvatarUpload={handleAvatarUpload}
          />
          {/* Right: Form Sections - each section in its own Card */}
          <div className="md:col-span-2 space-y-8">
            {isNGO ? (
              // NGO/Organization Edit Page
              <>
                <Card className="bg-card border border-border shadow-lg rounded-2xl p-8">
                  <CardContent className="space-y-6">
                    <h3 className="text-xl font-bold text-foreground mb-4">Organization Info</h3>
                    <div className="space-y-2">
                      <label htmlFor="organizationName">Organization Name</label>
                      <Input id="organizationName" value={profileData.organizationName} onChange={e => setProfileData(prev => ({ ...prev, organizationName: e.target.value }))} disabled={!isEditing} />
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="website">Website</label>
                      <Input id="website" value={profileData.website} onChange={e => setProfileData(prev => ({ ...prev, website: e.target.value }))} disabled={!isEditing} />
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="headquartersLocation">Headquarters Location</label>
                      <Input id="headquartersLocation" value={profileData.headquartersLocation} onChange={e => setProfileData(prev => ({ ...prev, headquartersLocation: e.target.value }))} disabled={!isEditing} />
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="missionStatement">Mission Statement</label>
                      <Textarea id="missionStatement" value={profileData.missionStatement} onChange={e => setProfileData(prev => ({ ...prev, missionStatement: e.target.value }))} disabled={!isEditing} rows={3} placeholder="e.g., Empowering individuals with disabilities through technology." />
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="contactEmail">Contact Email</label>
                      <Input id="contactEmail" value={profileData.email} onChange={e => setProfileData(prev => ({ ...prev, email: e.target.value }))} disabled={!isEditing} />
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="description">Description / Type of Services</label>
                      <Textarea id="description" value={profileData.bio} onChange={e => setProfileData(prev => ({ ...prev, bio: e.target.value }))} disabled={!isEditing} rows={3} placeholder="Describe your organization or services" />
                    </div>
                    <div className="space-y-2">
                      <label>Logo / Cover Image</label>
                      <Input type="file" accept="image/*" onChange={handleAvatarUpload} disabled={!isEditing} />
                      {profileData.avatarUrl && (
                        <Avatar className="h-24 w-24 mt-2">
                          <AvatarImage src={profileData.avatarUrl} alt="Organization Logo" />
                          <AvatarFallback>Logo</AvatarFallback>
                        </Avatar>
                      )}
                    </div>
                    <div>
                      <label>Accessibility Features</label>
                      <div className="flex flex-wrap gap-2">
                        {[
                          "Screen Reader Support", 
                          "Sign Language Support", 
                          "Assistive Technology Provided", 
                          "Remote Work Options", 
                          "Flexible Hours", 
                          "Accessible Office Space"
                        ].map(feature => {
                          const selected = profileData.accessibilityFeatures.includes(feature);
                          return (
                            <button
                              key={feature}
                              type="button"
                              className={`px-4 py-2 rounded-full border text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-mustard focus:ring-offset-2
                                ${selected ? 'bg-mustard text-white border-mustard' : 'bg-white text-charcoal border-gray-300 hover:bg-mustard/10'}
                                ${!isEditing ? 'opacity-60 cursor-not-allowed' : 'cursor-pointer'}`}
                              onClick={() => {
                                if (!isEditing) return;
                                setProfileData(prev => ({
                                  ...prev,
                                  accessibilityFeatures: selected
                                    ? prev.accessibilityFeatures.filter(f => f !== feature)
                                    : [...prev.accessibilityFeatures, feature]
                                }))
                              }}
                              disabled={!isEditing}
                              aria-pressed={selected}
                            >
                              {feature}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <label>Open to Collaboration</label>
                      <input
                        type="checkbox"
                        checked={profileData.openToCollaboration}
                        onChange={e => setProfileData(prev => ({ ...prev, openToCollaboration: e.target.checked }))}
                        disabled={!isEditing}
                      />
                    </div>
                  </CardContent>
                </Card>
              </>
            ) : (
              // Regular User Edit Page
              <>
                {/* Basic Info */}
                <Card className="bg-card border border-border shadow-lg rounded-2xl p-8">
                  <CardContent className="space-y-6">
                    <h3 className="text-xl font-bold text-foreground mb-4">Basic Info</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <label htmlFor="firstName">First Name</label>
                        <Input id="firstName" value={profileData.firstName} onChange={(e) => setProfileData((prev) => ({ ...prev, firstName: e.target.value }))} disabled={!isEditing} />
                      </div>
                      <div className="space-y-2">
                        <label htmlFor="lastName">Last Name</label>
                        <Input id="lastName" value={profileData.lastName} onChange={(e) => setProfileData((prev) => ({ ...prev, lastName: e.target.value }))} disabled={!isEditing} />
                      </div>
                    </div>
                  </CardContent>
                </Card>
                {/* Professional Details */}
                <Card className="bg-card border border-border shadow-lg rounded-2xl p-8">
                  <CardContent className="space-y-6">
                    <h3 className="text-xl font-bold text-foreground mb-4">Professional Details</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <label htmlFor="title">Title</label>
                        <Input id="title" value={profileData.title} onChange={(e) => setProfileData((prev) => ({ ...prev, title: e.target.value }))} disabled={!isEditing} />
                      </div>
                      <div className="space-y-2">
                        <label htmlFor="company">Company</label>
                        <Input id="company" value={profileData.company} onChange={(e) => setProfileData((prev) => ({ ...prev, company: e.target.value }))} disabled={!isEditing} />
                      </div>
                      <div className="space-y-2">
                        <label htmlFor="location">Location</label>
                        <Input id="location" value={profileData.location} onChange={(e) => setProfileData((prev) => ({ ...prev, location: e.target.value }))} disabled={!isEditing} />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="bio">Bio</label>
                      <Textarea id="bio" value={profileData.bio} onChange={(e) => setProfileData((prev) => ({ ...prev, bio: e.target.value }))} disabled={!isEditing} rows={3} placeholder="Tell us about yourself" />
                    </div>
                  </CardContent>
                </Card>
                {/* Contact & Social */}
                <Card className="bg-card border border-border shadow-lg rounded-2xl p-8">
                  <CardContent className="space-y-6">
                    <h3 className="text-xl font-bold text-foreground mb-4">Contact & Social</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <label htmlFor="email">Email</label>
                        <div className="flex items-center gap-2">
                          <Input id="email" value={profileData.email} onChange={(e) => setProfileData((prev) => ({ ...prev, email: e.target.value }))} disabled={!isEditing} />
                          <Checkbox
                            id="showEmail"
                            checked={profileData.showEmail}
                            onCheckedChange={(checked) => isEditing && setProfileData((prev) => ({ ...prev, showEmail: !!checked }))}
                            disabled={!isEditing}
                            aria-label="Show email on public profile"
                          />
                          <label htmlFor="showEmail" className="text-xs text-gray-500 select-none cursor-pointer">Show on public profile</label>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <label htmlFor="phone">Phone</label>
                        <div className="flex items-center gap-2">
                          <Input id="phone" value={profileData.phone} onChange={(e) => setProfileData((prev) => ({ ...prev, phone: e.target.value }))} disabled={!isEditing} />
                          <Checkbox
                            id="showPhone"
                            checked={profileData.showPhone}
                            onCheckedChange={(checked) => isEditing && setProfileData((prev) => ({ ...prev, showPhone: !!checked }))}
                            disabled={!isEditing}
                            aria-label="Show phone on public profile"
                          />
                          <label htmlFor="showPhone" className="text-xs text-gray-500 select-none cursor-pointer">Show on public profile</label>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <label htmlFor="website">Website</label>
                        <Input id="website" value={profileData.website} onChange={(e) => setProfileData((prev) => ({ ...prev, website: e.target.value }))} disabled={!isEditing} />
                      </div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div className="space-y-2">
                        <label htmlFor="linkedin">LinkedIn</label>
                        <Input id="linkedin" value={profileData.linkedin} onChange={(e) => setProfileData((prev) => ({ ...prev, linkedin: e.target.value }))} disabled={!isEditing} />
                      </div>
                      <div className="space-y-2">
                        <label htmlFor="twitter">Twitter</label>
                        <Input id="twitter" value={profileData.twitter} onChange={(e) => setProfileData((prev) => ({ ...prev, twitter: e.target.value }))} disabled={!isEditing} />
                      </div>
                      <div className="space-y-2">
                        <label htmlFor="github">GitHub</label>
                        <Input id="github" value={profileData.github} onChange={(e) => setProfileData((prev) => ({ ...prev, github: e.target.value }))} disabled={!isEditing} />
                      </div>
                    </div>
                  </CardContent>
                </Card>
                {/* Accessibility & Preferences */}
                <Card className="bg-card border border-border shadow-lg rounded-2xl p-8">
                  <CardContent className="space-y-6">
                    <h3 className="text-xl font-bold text-foreground mb-4">Accessibility & Preferences</h3>
                    <div className="flex flex-wrap gap-2">
                      {accessibilityOptions.map((option) => (
                        <div
                          key={option.id}
                          className={`flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors select-none border
                            ${(profileData.accessibilityNeeds || []).includes(option.id)
                              ? 'bg-mustard text-white border-mustard'
                              : 'bg-background border-border hover:bg-accent'
                            }`}
                          onClick={() => isEditing && handleAccessibilityToggle(option.id)}
                        >
                          <Checkbox
                            checked={(profileData.accessibilityNeeds || []).includes(option.id)}
                            onCheckedChange={() => isEditing && handleAccessibilityToggle(option.id)}
                            disabled={!isEditing}
                            aria-label={`Select ${option.label}`}
                          />
                          <option.icon
                            className={
                              `h-5 w-5 ${
                                (profileData.accessibilityNeeds || []).includes(option.id)
                                  ? "text-white"
                                  : "text-muted-foreground"
                              }`
                            }
                          />
                          <span className="text-foreground">{option.label}</span>
                        </div>
                      ))}
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="accommodations">Required Accommodations</label>
                      <Textarea id="accommodations" value={profileData.accommodations} onChange={(e) => setProfileData((prev) => ({ ...prev, accommodations: e.target.value }))} disabled={!isEditing} rows={3} placeholder="e.g., Flexible work hours, specialized software, ergonomic equipment" />
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
            {profileSaved && (
              <div className="text-center mt-8">
                <Button className="bg-mustard hover:bg-forest-green text-white font-medium" onClick={() => router.push("/dashboard")}>Go to Dashboard</Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
} 
              
 