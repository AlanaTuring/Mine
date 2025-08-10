"use client"

import { useState, useEffect, useCallback } from "react";
import { FollowButton } from "@/components/FollowButton";
import { FollowersList } from "@/components/FollowersList";
import { FollowingList } from "@/components/FollowingList";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getSupabaseClient } from "@/lib/supabaseClient";
import ProfilePostsClient from "./ProfilePostsClient";
import ErrorBoundary from "@/components/ErrorBoundary";

interface ProfileClientProps {
  profile: any;
  posts: any[];
  ngo: boolean;
}

export default function ProfileClient({ profile, posts, ngo }: ProfileClientProps) {
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("posts");

  const getCurrentUser = useCallback(async () => {
    try {
      const supabase = getSupabaseClient();
      const { data: { user } } = await supabase.auth.getUser();
      setCurrentUserId(user?.id || null);
    } catch (error) {
      console.error('Error getting current user:', error);
    }
  }, []);

  useEffect(() => {
    getCurrentUser();
  }, [getCurrentUser]);

  return (
    <ErrorBoundary>
      <div className="space-y-6">
        {/* Follow Button */}
        <div className="flex justify-center">
          <FollowButton
            targetUserId={profile.id}
            currentUserId={currentUserId}
          />
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="posts">Posts</TabsTrigger>
            <TabsTrigger value="followers">Followers</TabsTrigger>
            <TabsTrigger value="following">Following</TabsTrigger>
            <TabsTrigger value="about">About</TabsTrigger>
          </TabsList>

          <TabsContent value="posts" className="mt-6">
            <ProfilePostsClient posts={posts} profile={profile} ngo={ngo} />
          </TabsContent>

          <TabsContent value="followers" className="mt-6">
            <FollowersList userId={profile.id} currentUserId={currentUserId} />
          </TabsContent>

          <TabsContent value="following" className="mt-6">
            <FollowingList userId={profile.id} currentUserId={currentUserId} />
          </TabsContent>

          <TabsContent value="about" className="mt-6">
            <div className="space-y-4">
              {profile.bio && (
                <div className="bg-card p-4 rounded-lg">
                  <h3 className="font-semibold mb-2">Bio</h3>
                  <p className="text-muted-foreground">{profile.bio}</p>
                </div>
              )}
              
              {profile.location && (
                <div className="bg-card p-4 rounded-lg">
                  <h3 className="font-semibold mb-2">Location</h3>
                  <p className="text-muted-foreground">{profile.location}</p>
                </div>
              )}

              {profile.title && (
                <div className="bg-card p-4 rounded-lg">
                  <h3 className="font-semibold mb-2">Title</h3>
                  <p className="text-muted-foreground">{profile.title}</p>
                </div>
              )}

              {profile.company && (
                <div className="bg-card p-4 rounded-lg">
                  <h3 className="font-semibold mb-2">Company</h3>
                  <p className="text-muted-foreground">{profile.company}</p>
                </div>
              )}

              {profile.accessibility_needs && profile.accessibility_needs.length > 0 && (
                <div className="bg-card p-4 rounded-lg">
                  <h3 className="font-semibold mb-2">Accessibility Needs</h3>
                  <div className="flex flex-wrap gap-2">
                    {profile.accessibility_needs.map((need: string, index: number) => (
                      <span key={index} className="bg-mustard text-white px-2 py-1 rounded text-sm">
                        {need}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {profile.skills && profile.skills.length > 0 && (
                <div className="bg-card p-4 rounded-lg">
                  <h3 className="font-semibold mb-2">Skills</h3>
                  <div className="flex flex-wrap gap-2">
                    {profile.skills.map((skill: string, index: number) => (
                      <span key={index} className="bg-accent text-accent-foreground px-2 py-1 rounded text-sm">
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </ErrorBoundary>
  );
} 