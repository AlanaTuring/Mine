"use client"

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Loader2, Users } from "lucide-react";
import { getSupabaseClient } from "@/lib/supabaseClient";
import Link from "next/link";

interface Following {
  id: string;
  follower_id: string;
  following_id: string;
  created_at: string;
  following_profile: {
    id: string;
    first_name: string;
    last_name: string;
    avatar_url?: string;
    organization_name?: string;
    account_type?: string;
  };
}

interface FollowingListProps {
  userId: string;
  currentUserId?: string;
}

export function FollowingList({ userId, currentUserId }: FollowingListProps) {
  const [following, setFollowing] = useState<Following[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

    const fetchFollowing = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const supabase = getSupabaseClient();
      
      // First get the following relationships
      const { data: followsData, error: followsError } = await supabase
        .from('user_follows')
        .select('following_id, created_at')
        .eq('follower_id', userId)
        .order('created_at', { ascending: false });

      if (followsError) {
        setError('Failed to load following');
        console.error('Error fetching following:', followsError);
        return;
      }

      if (!followsData || followsData.length === 0) {
        setFollowing([]);
        return;
      }

      // Then get the profile data for each followed user
      const followingIds = followsData.map(follow => follow.following_id);
      const { data: profilesData, error: profilesError } = await supabase
        .from('profiles')
        .select('id, first_name, last_name, avatar_url, organization_name, account_type')
        .in('id', followingIds);

      if (profilesError) {
        setError('Failed to load following profiles');
        console.error('Error fetching following profiles:', profilesError);
        return;
      }

      // Combine the data
      const profilesMap = new Map(profilesData?.map(profile => [profile.id, profile]) || []);
      const followingWithProfiles = followsData.map(follow => ({
        id: follow.following_id,
        follower_id: userId,
        following_id: follow.following_id,
        created_at: follow.created_at,
        following_profile: profilesMap.get(follow.following_id) || {
          id: follow.following_id,
          first_name: 'Unknown',
          last_name: 'User',
          avatar_url: null,
          organization_name: null,
          account_type: null
        }
      }));

      setFollowing(followingWithProfiles);
    } catch (error) {
      setError('Failed to load following');
      console.error('Error fetching following:', error);
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchFollowing();
  }, [fetchFollowing]);

  const getDisplayName = (following: Following) => {
    const profile = following.following_profile;
    if (profile.account_type === 'NGO' || profile.account_type === 'NGO / Organization') {
      return profile.organization_name || 'Organization';
    }
    return `${profile.first_name || ''} ${profile.last_name || ''}`.trim() || 'User';
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Following
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Following
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <p className="mb-4">{error}</p>
            <Button onClick={fetchFollowing} variant="outline">
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Users className="h-5 w-5" />
          Following ({following.length})
        </CardTitle>
      </CardHeader>
      <CardContent>
        {following.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            Not following anyone yet
          </div>
        ) : (
          <div className="space-y-3">
            {following.map((follow) => (
                             <div key={follow.id} className="flex items-center gap-3 p-3 rounded-lg transition-colors">
                <Link href={`/profile/${follow.following_id}`} className="flex items-center gap-3 flex-1">
                  <Avatar className="w-10 h-10">
                    <AvatarImage 
                      src={follow.following_profile.avatar_url || '/placeholder-user.jpg'} 
                      alt={getDisplayName(follow)} 
                    />
                    <AvatarFallback>
                      {getDisplayName(follow)[0]}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-foreground truncate">
                      {getDisplayName(follow)}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {follow.following_profile.account_type === 'NGO' ? 'Organization' : 'Individual'}
                    </p>
                  </div>
                </Link>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
} 