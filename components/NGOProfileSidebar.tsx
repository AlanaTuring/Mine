"use client"

import { useEffect, useState, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Accessibility, Eye, Volume2, Edit, RefreshCw, Camera } from "lucide-react";
import Link from "next/link";
import { getSupabaseClient } from "@/lib/supabaseClient";
import { useAccessibility } from "@/components/accessibility-provider";

export function NGOProfileSidebar({ showEditButton = true, isEditing = false }: { showEditButton?: boolean, isEditing?: boolean }) {
  const { settings } = useAccessibility();
  const [profile, setProfile] = useState<any>(null);
  const [followers, setFollowers] = useState(0);
  const [following, setFollowing] = useState(0);
  const [loading, setLoading] = useState(true);
  const [avatarPreview, setAvatarPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchProfileAndCounts = async () => {
    setLoading(true);
    try {
      const supabase = getSupabaseClient();
      const { data: { user }, error: authError } = await supabase.auth.getUser();
      if (authError) {
        setLoading(false);
        return;
      }
      if (!user) {
        setLoading(false);
        return;
      }
      const { data: profileData, error: profileError } = await supabase
        .from("profiles")
        .select("*")
        .eq("id", user.id)
        .single();
      if (profileError) {
        setLoading(false);
        return;
      }
      setProfile(profileData);
      setAvatarPreview(null); // Reset preview on profile fetch
      const { count: followersCount } = await supabase
        .from("user_follows")
        .select("*", { count: "exact", head: true })
        .eq("following_id", user.id);
      setFollowers(followersCount || 0);
      const { count: followingCount } = await supabase
        .from("user_follows")
        .select("*", { count: "exact", head: true })
        .eq("follower_id", user.id);
      setFollowing(followingCount || 0);
    } catch (error) {
      setLoading(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchProfileAndCounts(); }, []);
  useEffect(() => { const interval = setInterval(fetchProfileAndCounts, 30000); return () => clearInterval(interval); }, []);

  const handleAvatarClick = () => {
    if (isEditing && fileInputRef.current) fileInputRef.current.click();
  };
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && isEditing) {
      const url = URL.createObjectURL(file);
      setAvatarPreview(url);
    }
  };

  if (loading) return (
    <Card className="bg-card shadow-md rounded-lg overflow-hidden">
      <CardContent className="p-4">
        <div className="flex items-center justify-center py-8">
          <RefreshCw className="h-6 w-6 animate-spin text-mustard" />
          <span className="ml-2 text-muted-foreground">Loading profile...</span>
        </div>
      </CardContent>
    </Card>
  );
  if (!profile) return (
    <Card className="bg-card shadow-md rounded-lg overflow-hidden">
      <CardContent className="p-4">
        <div className="text-center py-8">
          <p className="text-muted-foreground">Profile not found</p>
        </div>
      </CardContent>
    </Card>
  );
  const isNGO = profile?.account_type === "NGO" || profile?.account_type === "NGO / Organization";
  const displayName = isNGO
    ? profile?.organization_name || "Organization"
    : `${profile?.first_name || ''} ${profile?.last_name || ''}`.trim() || "User";
  const displayTitle = isNGO ? undefined : profile?.title;
  return (
    <Card className="shadow-md rounded-xl overflow-hidden bg-card text-foreground">
      <CardHeader className="relative p-0">
        <div className="h-20 bg-gradient-to-r from-mustard to-forest-green" />
        <div className="absolute top-8 left-1/2 -translate-x-1/2 group cursor-pointer" onClick={handleAvatarClick} tabIndex={isEditing ? 0 : -1} aria-label={isEditing ? 'Change profile photo' : undefined} role="button">
          <Avatar className="w-24 h-24 rounded-full object-cover border-4 border-white shadow-md">
            <AvatarImage src={avatarPreview || profile?.avatar_url || "/placeholder.svg?height=96&width=96"} alt={profile?.first_name} />
            <AvatarFallback className="bg-gradient-to-br from-mustard to-forest-green text-white text-3xl">
              {profile?.first_name?.[0] || ''}{profile?.last_name?.[0] || ''}
            </AvatarFallback>
          </Avatar>
          {isEditing && (
            <>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
                aria-label="Upload new profile photo"
              />
              <div className="absolute inset-0 bg-black/40 flex flex-col items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity rounded-full">
                <Camera className="h-8 w-8 text-white mb-1" aria-hidden="true" />
                <span className="text-white text-sm font-medium">Change Photo</span>
              </div>
            </>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-16 text-center">
        <h2 className="text-xl font-bold text-foreground">{displayName}</h2>
        {displayTitle && <p className="text-sm mb-4 text-muted-foreground">{displayTitle}</p>}
        <div className="flex justify-around text-center mb-4">
          <div>
            <p className="text-sm font-semibold text-foreground">{followers}</p>
            <p className="text-xs text-muted-foreground">Followers</p>
          </div>
          <div>
            <p className="text-sm font-semibold text-foreground">{following}</p>
            <p className="text-xs text-muted-foreground">Following</p>
          </div>
        </div>
        {showEditButton && (
          <Link href="/profile">
            <Button className="w-full bg-mustard hover:bg-forest-green text-white font-medium">
              <Edit className="h-4 w-4 mr-2" />
              Edit Profile
            </Button>
          </Link>
        )}
        <Separator className="my-4" />
        <CardTitle className="text-lg font-bold mb-3 flex items-center justify-center text-foreground">
          <Accessibility className="h-5 w-5 mr-2 text-mustard" />
          Accessibility Profile
        </CardTitle>
        <div className="space-y-2">
          {(profile?.accessibility_needs || []).map((need: string) => (
            <div key={need} className="flex items-center justify-center space-x-2 text-foreground">
              {need === "visual" && <Eye className="h-4 w-4 text-indigo" />}
              {need === "hearing" && <Volume2 className="h-4 w-4 text-indigo" />}
              {/* Add more icons as needed */}
              <span className="text-sm">{need.replace(/^\w/, c => c.toUpperCase())} Impairment</span>
            </div>
          ))}
          {(profile?.accessibility_needs || []).length === 0 && (
            <p className="text-sm text-center text-muted-foreground">No accessibility needs specified</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
} 