"use client";
import React from "react";
import { getSupabaseClient } from '@/lib/supabaseClient';
import { FeedPost } from '@/components/feed-post';

export default function ProfilePostsClient({ posts, profile, ngo }: { posts: any[]; profile: any; ngo: boolean }) {
  const [currentUserId, setCurrentUserId] = React.useState<string | null>(null);
  React.useEffect(() => {
    (async () => {
      const supabase = getSupabaseClient();
      const { data: { user } } = await supabase.auth.getUser();
      setCurrentUserId(user?.id || null);
    })();
  }, []);

  if (!posts) return null;
  if (posts.length === 0) return <div className="text-gray-500">{ngo ? 'No campaigns or posts yet.' : 'No posts yet.'}</div>;
  return (
    <div className="space-y-6">
      {posts.map((post) => (
        <FeedPost
          key={post.id}
          postId={post.id}
          author={profile.user_metadata?.displayName || (ngo ? profile.organization_name : `${profile.first_name} ${profile.last_name}`)}
          title={post.title || ''}
          time={new Date(post.created_at).toLocaleString('en-US', {
            year: 'numeric', month: 'numeric', day: 'numeric', hour: 'numeric', minute: 'numeric', hour12: true
          })}
          content={post.content || ''}
          likes={post.likes || 0}
          comments={post.comments || 0}
          shares={post.shares || 0}
          avatar={profile.avatar_url}
          isVerified={false}
          imageUrl={post.image_url}
          videoUrl={post.video_url}
          isJobPost={post.is_job_post}
          postUserId={profile.id}
          currentUserId={currentUserId || undefined}
        />
      ))}
    </div>
  );
} 