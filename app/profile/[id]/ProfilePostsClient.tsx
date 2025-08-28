"use client";
import React from "react";
import { getSupabaseClient } from '@/lib/supabaseClient';
import { FeedPost } from '@/components/feed-post';

export default function ProfilePostsClient({ posts, profile, ngo }: { posts: any[]; profile: any; ngo: boolean }) {
  const [currentUserId, setCurrentUserId] = React.useState<string | null>(null);
  const jobPosts = React.useMemo(() => posts.filter(p => p.is_job_post), [posts]);
  const regularPosts = React.useMemo(() => posts.filter(p => !p.is_job_post), [posts]);
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
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Left sidebar: Jobs slider (moved here) */}
      <div className="hidden lg:block space-y-4">
        {jobPosts.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-lg font-semibold text-foreground">Jobs</h3>
            <div className="overflow-x-auto">
              <div className="flex gap-4 pb-2">
                {jobPosts.map((post) => (
                  <div key={post.id} className="min-w-[320px] max-w-[360px] w-[360px] bg-card border border-border rounded-lg p-4">
                    <div className="text-sm text-muted-foreground mb-1">{new Date(post.created_at).toLocaleDateString()}</div>
                    <div className="font-semibold text-foreground mb-1 truncate">{post.title || 'Job'}</div>
                    <div className="text-sm text-muted-foreground line-clamp-3 mb-3">{post.content || ''}</div>
                    <FeedPost
                      key={post.id}
                      postId={post.id}
                      author={profile.user_metadata?.displayName || (ngo ? profile.organization_name : `${profile.first_name} ${profile.last_name}`)}
                      title={post.title || ''}
                      time={new Date(post.created_at).toLocaleString('en-US', { year: 'numeric', month: 'numeric', day: 'numeric', hour: 'numeric', minute: 'numeric', hour12: true })}
                      content={post.content || ''}
                      likes={post.likes || 0}
                      comments={post.comments || 0}
                      shares={post.shares || 0}
                      avatar={profile.avatar_url}
                      isVerified={false}
                      imageUrl={post.image_url}
                      videoUrl={post.video_url}
                      isJobPost={true}
                      postUserId={profile.id}
                      currentUserId={currentUserId || undefined}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Main column: posts with horizontal jobs slider */}
      <div className="space-y-8 lg:col-span-2">
        {regularPosts.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-foreground">Posts</h3>
            {regularPosts.map((post) => (
              <FeedPost
                key={post.id}
                postId={post.id}
                author={profile.user_metadata?.displayName || (ngo ? profile.organization_name : `${profile.first_name} ${profile.last_name}`)}
                title={post.title || ''}
                time={new Date(post.created_at).toLocaleString('en-US', { year: 'numeric', month: 'numeric', day: 'numeric', hour: 'numeric', minute: 'numeric', hour12: true })}
                content={post.content || ''}
                likes={post.likes || 0}
                comments={post.comments || 0}
                shares={post.shares || 0}
                avatar={profile.avatar_url}
                isVerified={false}
                imageUrl={post.image_url}
                videoUrl={post.video_url}
                isJobPost={false}
                postUserId={profile.id}
                currentUserId={currentUserId || undefined}
              />
            ))}
          </div>
        )}

        {/* Jobs removed from bottom; now left sidebar */}
      </div>

      {/* Right sidebar intentionally left empty for now */}
    </div>
  );
} 