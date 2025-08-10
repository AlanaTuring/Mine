"use client"

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Loader2, Users } from "lucide-react";
import { getSupabaseClient } from "@/lib/supabaseClient";
import Link from "next/link";

interface Follower {
  id: string;
  follower_id: string;
  following_id: string;
  created_at: string;
  follower_profile: {
    id: string;
    first_name: string;
    last_name: string;
    avatar_url?: string;
    organization_name?: string;
    account_type?: string;
  };
}

interface FollowersListProps {
  userId: string;
  currentUserId?: string;
}

export function FollowersList({ userId, currentUserId }: FollowersListProps) {
  const [followers, setFollowers] = useState<Follower[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchFollowers = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const supabase = getSupabaseClient();
      
      // First get the followers
      const { data: followsData, error: followsError } = await supabase
        .from('user_follows')
        .select('follower_id, created_at')
        .eq('following_id', userId)
        .order('created_at', { ascending: false });

      if (followsError) {
        setError('Failed to load followers');
        console.error('Error fetching followers:', followsError);
        return;
      }

      if (!followsData || followsData.length === 0) {
        setFollowers([]);
        return;
      }

      // Then get the profile data for each follower
      const followerIds = followsData.map(follow => follow.follower_id);
      const { data: profilesData, error: profilesError } = await supabase
        .from('profiles')
        .select('id, first_name, last_name, avatar_url, organization_name, account_type')
        .in('id', followerIds);

      if (profilesError) {
        setError('Failed to load follower profiles');
        console.error('Error fetching follower profiles:', profilesError);
        return;
      }

      // Combine the data
      const profilesMap = new Map(profilesData?.map(profile => [profile.id, profile]) || []);
      const followersWithProfiles = followsData.map(follow => ({
        id: follow.follower_id,
        follower_id: follow.follower_id,
        following_id: userId,
        created_at: follow.created_at,
        follower_profile: profilesMap.get(follow.follower_id) || {
          id: follow.follower_id,
          first_name: 'Unknown',
          last_name: 'User',
          avatar_url: null,
          organization_name: null,
          account_type: null
        }
      }));

      setFollowers(followersWithProfiles);
    } catch (error) {
      setError('Failed to load followers');
      console.error('Error fetching followers:', error);
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchFollowers();
  }, [fetchFollowers]);

  const getDisplayName = (follower: Follower) => {
    const profile = follower.follower_profile;
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
            Followers
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
            Followers
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <p className="mb-4">{error}</p>
            <Button onClick={fetchFollowers} variant="outline">
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
          Followers ({followers.length})
        </CardTitle>
      </CardHeader>
      <CardContent>
        {followers.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            No followers yet
          </div>
        ) : (
          <div className="space-y-3">
            {followers.map((follower) => (
                             <div key={follower.id} className="flex items-center gap-3 p-3 rounded-lg transition-colors">
                <Link href={`/profile/${follower.follower_id}`} className="flex items-center gap-3 flex-1">
                  <Avatar className="w-10 h-10">
                    <AvatarImage 
                      src={follower.follower_profile.avatar_url || '/placeholder-user.jpg'} 
                      alt={getDisplayName(follower)} 
                    />
                    <AvatarFallback>
                      {getDisplayName(follower)[0]}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-foreground truncate">
                      {getDisplayName(follower)}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {follower.follower_profile.account_type === 'NGO' ? 'Organization' : 'Individual'}
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