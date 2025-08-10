import { Card } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import React from "react";

interface ProfileData {
  firstName?: string;
  lastName?: string;
  organizationName?: string;
  avatarUrl?: string;
  accountType?: string;
}

interface ProfileSidebarProps {
  profileData: ProfileData;
  isEditing: boolean;
  onAvatarUpload?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  showEditButton?: boolean;
}

export function UserProfileSidebar({ profileData, isEditing, onAvatarUpload }: ProfileSidebarProps) {
  const displayName = profileData.accountType === "NGO"
    ? profileData.organizationName || "Organization"
    : `${profileData.firstName || ""} ${profileData.lastName || ""}`.trim() || "User";

  return (
    <Card className="p-5 space-y-2 rounded-xl shadow-md bg-card hover:shadow-lg transition w-full flex flex-col items-center">
      <div className="flex flex-col items-center gap-3">
        <Avatar className="w-24 h-24 rounded-full ring-2 ring-mustard hover:ring-4 transition-all cursor-pointer mb-2">
          <AvatarImage src={profileData.avatarUrl || '/placeholder-user.jpg'} alt={displayName} />
          <AvatarFallback>{displayName[0]}</AvatarFallback>
        </Avatar>
        {isEditing && (
          <input
            type="file"
            accept="image/*"
            onChange={onAvatarUpload}
            className="mb-2"
          />
        )}
        <div className="text-xl font-bold text-center text-foreground">{displayName}</div>
        {/* No subtitle for NGOs to avoid duplicate name display */}
      </div>
    </Card>
  );
} 