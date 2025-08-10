'use client'

import { Card } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import React from "react";

interface ProfileCardProps {
  avatarUrl?: string;
  name: string;
  subtitle?: string;
  isEditing?: boolean;
  onAvatarUpload?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  compact?: boolean; // NEW: for post feed
}

export function ProfileCard({ avatarUrl, name, subtitle, isEditing = false, onAvatarUpload, compact = false }: ProfileCardProps) {
  if (compact) {
    // Compact horizontal layout for post feed
    return (
      <div
        className="flex items-center gap-2 cursor-pointer focus:outline-none rounded-full transition-shadow"
        tabIndex={0}
        role="link"
        aria-label={`View ${name}'s profile`}
      >
        <Avatar className="w-10 h-10 rounded-full ring-2 ring-mustard">
          <AvatarImage src={avatarUrl || '/placeholder-user.jpg'} alt={name} />
          <AvatarFallback>{name[0]}</AvatarFallback>
        </Avatar>
        <div>
          <div className="font-semibold text-sm leading-tight">{name}</div>
          {subtitle && <div className="text-xs text-gray-500 leading-tight">{subtitle}</div>}
        </div>
      </div>
    );
  }
  // Default large card for profile pages
  return (
    <Card className="p-5 space-y-2 rounded-xl shadow-md bg-card transition w-full flex flex-col items-center">
      <div className="flex flex-col items-center gap-3">
        <Avatar className="w-24 h-24 rounded-full ring-2 ring-mustard transition-all cursor-pointer mb-2">
          <AvatarImage src={avatarUrl || '/placeholder-user.jpg'} alt={name} />
          <AvatarFallback>{name[0]}</AvatarFallback>
        </Avatar>
        {isEditing && onAvatarUpload && (
          <input
            type="file"
            accept="image/*"
            onChange={onAvatarUpload}
            className="mb-2"
          />
        )}
        <div className="text-xl font-bold text-center text-foreground">{name}</div>
        {subtitle && <div className="text-sm text-muted-foreground text-center">{subtitle}</div>}
      </div>
    </Card>
  );
} 