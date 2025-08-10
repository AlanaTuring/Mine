"use client"

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { UserPlus, UserCheck, Loader2 } from "lucide-react";
import { getSupabaseClient } from "@/lib/supabaseClient";

interface FollowButtonProps {
  targetUserId: string;
  currentUserId?: string;
  initialIsFollowing?: boolean;
  onFollowChange?: (isFollowing: boolean) => void;
}

export function FollowButton({ 
  targetUserId, 
  currentUserId, 
  initialIsFollowing = false,
  onFollowChange 
}: FollowButtonProps) {
  const [isFollowing, setIsFollowing] = useState(initialIsFollowing);
  const [isLoading, setIsLoading] = useState(false);
  const [followersCount, setFollowersCount] = useState(0);

  const checkFollowStatus = useCallback(async () => {
    if (!currentUserId) return;
    
    try {
      const supabase = getSupabaseClient();
      const { data, error } = await supabase
        .from('user_follows')
        .select('*')
        .eq('follower_id', currentUserId)
        .eq('following_id', targetUserId)
        .single();

      if (!error && data) {
        setIsFollowing(true);
      } else {
        setIsFollowing(false);
      }
    } catch (error) {
      console.error('Error checking follow status:', error);
    }
  }, [currentUserId, targetUserId]);

  const fetchFollowersCount = useCallback(async () => {
    try {
      const supabase = getSupabaseClient();
      const { count, error } = await supabase
        .from('user_follows')
        .select('*', { count: 'exact', head: true })
        .eq('following_id', targetUserId);

      if (!error && count !== null) {
        setFollowersCount(count);
      }
    } catch (error) {
      console.error('Error fetching followers count:', error);
    }
  }, [targetUserId]);

  useEffect(() => {
    if (currentUserId && currentUserId !== targetUserId) {
      checkFollowStatus();
      fetchFollowersCount();
    }
  }, [targetUserId, currentUserId, checkFollowStatus, fetchFollowersCount]);

  const handleFollowToggle = async () => {
    if (!currentUserId || isLoading) return;

    setIsLoading(true);
    try {
      const supabase = getSupabaseClient();

      if (isFollowing) {
        // Unfollow
        const { error } = await supabase
          .from('user_follows')
          .delete()
          .eq('follower_id', currentUserId)
          .eq('following_id', targetUserId);

        if (!error) {
          setIsFollowing(false);
          setFollowersCount(prev => Math.max(0, prev - 1));
          onFollowChange?.(false);
        }
      } else {
        // Follow
        const { error } = await supabase
          .from('user_follows')
          .insert({
            follower_id: currentUserId,
            following_id: targetUserId
          });

        if (!error) {
          setIsFollowing(true);
          setFollowersCount(prev => prev + 1);
          onFollowChange?.(true);
        }
      }
    } catch (error) {
      console.error('Error toggling follow status:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Don't show follow button if user is viewing their own profile
  if (!currentUserId || currentUserId === targetUserId) {
    return null;
  }

  return (
    <div className="flex items-center gap-3">
      <Button
        onClick={handleFollowToggle}
        disabled={isLoading}
        className={`flex items-center gap-2 ${
          isFollowing 
            ? 'bg-muted text-muted-foreground hover:bg-destructive hover:text-destructive-foreground' 
            : 'bg-mustard hover:bg-forest-green text-white'
        }`}
        aria-label={isFollowing ? 'Unfollow user' : 'Follow user'}
        aria-pressed={isFollowing}
      >
        {isLoading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : isFollowing ? (
          <UserCheck className="h-4 w-4" />
        ) : (
          <UserPlus className="h-4 w-4" />
        )}
        {isLoading ? 'Loading...' : isFollowing ? 'Following' : 'Follow'}
      </Button>
      
      <div className="text-sm text-muted-foreground">
        {followersCount} {followersCount === 1 ? 'follower' : 'followers'}
      </div>
    </div>
  );
} 