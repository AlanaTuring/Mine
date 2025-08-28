"use client";
import { useEffect, useState } from 'react';
import { getSupabaseClient } from '@/lib/supabaseClient';
import { FeedPost } from "@/components/feed-post";

interface TransformedPost {
  id: string;
  user_id: string;
  title: string;
  content: string;
  tags: string[];
  likes: number;
  comments: number;
  shares: number;
  created_at: string;
  updated_at: string;
  author_name: string;
  avatar_url?: string;
  image_url?: string;
  video_url?: string;
  is_job_post?: boolean;
}

export default function PostsList() {
  const [posts, setPosts] = useState<TransformedPost[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    async function fetchCurrentUser() {
      const supabase = getSupabaseClient();
      const { data: { user } } = await supabase.auth.getUser();
      setCurrentUserId(user?.id || null);
    }
    fetchCurrentUser();
  }, []);

  const fetchPosts = async () => {
    try {
      const supabase = getSupabaseClient();
      // First fetch posts
      const { data: postsData, error: postsError } = await supabase
        .from('posts')
        .select('*')
        .order('created_at', { ascending: false });

      if (postsError) throw postsError;

      // Then fetch profiles for all users
      const userIds = Array.from(new Set(postsData?.map(post => post.user_id) || []));
      const { data: profilesData, error: profilesError } = await supabase
        .from('profiles')
        .select('id, first_name, last_name, avatar_url, organization_name, account_type')
        .in('id', userIds);

      if (profilesError) throw profilesError;

      // Create a map of user profiles
      const profilesMap = new Map();
      profilesData?.forEach(profile => {
        profilesMap.set(profile.id, profile);
      });

      // Fetch comment counts for all posts
      const postIds = postsData?.map(post => post.id) || [];
      const { data: commentCounts, error: commentError } = await supabase
        .from('comments')
        .select('post_id')
        .in('post_id', postIds);

      if (commentError) {
        console.error('Error fetching comment counts:', commentError);
      }

      // Create a map of comment counts
      const commentCountMap = new Map();
      commentCounts?.forEach(comment => {
        commentCountMap.set(comment.post_id, (commentCountMap.get(comment.post_id) || 0) + 1);
      });

      // Fetch like counts for all posts
      const { data: likeCounts, error: likeError } = await supabase
        .from('likes')
        .select('post_id')
        .in('post_id', postIds);

      if (likeError) {
        console.error('Error fetching like counts:', likeError);
      }

      // Create a map of like counts
      const likeCountMap = new Map();
      likeCounts?.forEach(like => {
        likeCountMap.set(like.post_id, (likeCountMap.get(like.post_id) || 0) + 1);
      });

      // Transform the data to include author information and real counts
      const transformedPosts: TransformedPost[] = postsData?.map((post: any) => {
        const profile = profilesMap.get(post.user_id);
        const realCommentCount = commentCountMap.get(post.id) || 0;
        const realLikeCount = likeCountMap.get(post.id) || 0;
        
        console.log('Post data:', post.id, 'real comments:', realCommentCount, 'real likes:', realLikeCount);
        
        let author_name = 'Anonymous';
        if (profile) {
          if (profile.account_type === 'NGO' || profile.account_type === 'NGO / Organization') {
            author_name = profile.organization_name || 'Organization';
          } else {
            author_name = `${profile.first_name} ${profile.last_name}`.trim();
          }
        }
        return {
          id: post.id,
          user_id: post.user_id,
          title: post.title,
          content: post.content,
          tags: post.tags,
          likes: realLikeCount,
          comments: realCommentCount, // Use real-time count instead of database column
          shares: post.shares || 0,
          created_at: post.created_at,
          updated_at: post.updated_at,
          author_name,
          avatar_url: profile?.avatar_url || undefined,
          image_url: post.image_url || undefined,
          video_url: post.video_url || undefined,
          is_job_post: post.is_job_post || false,
        };
      }) || [];

      setPosts(transformedPosts);
    } catch (err) {
      console.error('Error fetching posts:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!mounted) return;
    fetchPosts();
  }, [mounted]);

  const handlePostUpdate = (postId: string, newTitle: string, newContent: string) => {
    setPosts(prevPosts => 
      prevPosts.map(post => 
        post.id === postId 
          ? { ...post, title: newTitle, content: newContent }
          : post
      )
    );
  };

  const handlePostCreated = () => {
    // Refresh the posts list when a new post is created
    fetchPosts();
  };

  const handlePostDelete = (postId: string) => {
    setPosts(prevPosts => prevPosts.filter(post => post.id !== postId));
  };

  // Function to refresh a specific post's counts
  const refreshPostCounts = (postId: string) => {
    setPosts(prevPosts => 
      prevPosts.map(post => {
        if (post.id === postId) {
          // Trigger a re-fetch of this specific post's counts
          return { ...post, _refresh: Date.now() };
        }
        return post;
      })
    );
  };

  // Function to refresh all posts (when comments/likes change)
  const refreshAllPosts = () => {
    console.log('🔄 Refreshing all posts due to comment/like change');
    fetchPosts();
  };

  if (!mounted) {
    return <div className="text-center py-8">Loading...</div>;
  }

  if (loading) return <div className="text-center py-8">Loading posts...</div>;
  if (error) return <div className="text-center py-8 text-red-600">Error: {error}</div>;

  return (
    <div className="space-y-6">
      {posts.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">
          No posts yet. Be the first to share something!
        </div>
      ) : (
        <div className="space-y-6">
          {posts.map(post => (
            <div key={post.id} className="bg-card rounded-lg shadow-sm border border-border p-4 mb-4 hover:shadow-md hover:scale-105 transition duration-200">
              <FeedPost
                postId={post.id}
                author={post.author_name}
                title={post.title || ""}
                time={new Date(post.created_at).toLocaleString('en-US', {
                  year: 'numeric',
                  month: 'numeric',
                  day: 'numeric',
                  hour: 'numeric',
                  minute: 'numeric',
                  hour12: true
                })}
                content={post.content || ""}
                likes={post.likes || 0}
                comments={post.comments || 0}
                shares={post.shares || 0}
                avatar={post.avatar_url}
                isVerified={false}
                imageUrl={post.image_url}
                videoUrl={post.video_url}
                isJobPost={post.is_job_post}
                currentUserId={currentUserId || undefined}
                postUserId={post.user_id}
                onPostUpdate={handlePostUpdate}
                onPostDelete={handlePostDelete}
                onCommentChange={refreshAllPosts}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
