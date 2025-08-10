import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Camera } from "lucide-react";
import React, { useRef } from "react";

interface ProfileData {
  firstName?: string;
  lastName?: string;
  organizationName?: string;
  avatarUrl?: string;
  accountType?: string;
  title?: string;
}

interface ProfileSidebarProps {
  profileData: ProfileData;
  isEditing: boolean;
  onAvatarUpload?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  showEditButton?: boolean;
}

export function UserProfileSidebar({ profileData, isEditing, onAvatarUpload }: ProfileSidebarProps) {
  const isNGO = profileData.accountType === "NGO" || profileData.accountType === "NGO / Organization";
  const name = isNGO
    ? profileData.organizationName || "Organization"
    : `${profileData.firstName || ""} ${profileData.lastName || ""}`.trim() || "User";
  const subtitle = isNGO ? undefined : profileData.title;
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleAvatarClick = () => {
    if (isEditing && fileInputRef.current) fileInputRef.current.click();
  };

  return (
    <Card className="shadow-md rounded-xl overflow-hidden bg-card text-foreground">
      <CardHeader className="relative p-0">
        <div className="h-20 bg-gradient-to-r from-mustard to-forest-green" />
        <div className="absolute top-8 left-1/2 -translate-x-1/2 group cursor-pointer" onClick={handleAvatarClick} tabIndex={isEditing ? 0 : -1} aria-label={isEditing ? 'Change profile photo' : undefined} role="button">
          <Avatar className="w-24 h-24 rounded-full ring-2 ring-mustard border-4 border-white shadow-md bg-white">
            <AvatarImage src={profileData.avatarUrl || '/placeholder-user.jpg'} alt={name} />
            <AvatarFallback>{name[0]}</AvatarFallback>
          </Avatar>
          {isEditing && (
            <>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={onAvatarUpload}
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
      <CardContent className="pt-16 pb-6 text-center">
        <div className="text-xl font-bold text-center mb-1 text-foreground">{name}</div>
        {subtitle && <div className="text-sm text-muted-foreground text-center mb-2">{subtitle}</div>}
      </CardContent>
    </Card>
  );
} 