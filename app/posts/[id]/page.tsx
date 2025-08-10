"use client";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { getSupabaseClient } from "@/lib/supabaseClient";
import { Header } from "@/components/header";
import { FeedPost } from "@/components/feed-post";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import Link from "next/link";

interface Post {
  id: string;
  user_id: string;
  title: string;
  content: string;
  created_at: string;
  updated_at: string;
  likes: number;
  comments: number;
  shares: number;
  image_url?: string;
  video_url?: string;
  is_job_post: boolean;
  author_name: string;
  avatar_url?: string;
}

export default function PostDetailPage() {
  const params = useParams();
  const router = useRouter();
  const postId = params.id as string;
  const [post, setPost] = useState<Post | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);

  useEffect(() => {
    async function fetchCurrentUser() {
      const supabase = getSupabaseClient();
      const { data: { user } } = await supabase.auth.getUser();
      setCurrentUserId(user?.id || null);
    }
    fetchCurrentUser();
  }, []);

  useEffect(() => {
    async function fetchPost() {
      if (!postId) return;
      
      try {
        setLoading(true);
        const supabase = getSupabaseClient();
        
        // Fetch the post
        const { data: postData, error: postError } = await supabase
          .from('posts')
          .select('*')
          .eq('id', postId)
          .single();

        if (postError) {
          setError("Post not found");
          setLoading(false);
          return;
        }

        // Fetch the author's profile
        const { data: profileData, error: profileError } = await supabase
          .from('profiles')
          .select('first_name, last_name, avatar_url, organization_name, account_type')
          .eq('id', postData.user_id)
          .single();

        if (profileError) {
          setError("Error loading post author");
          setLoading(false);
          return;
        }

        // Fetch comment count
        const { count: commentCount, error: commentError } = await supabase
          .from('comments')
          .select('*', { count: 'exact', head: true })
          .eq('post_id', postId);

        // Fetch like count
        const { count: likeCount, error: likeError } = await supabase
          .from('likes')
          .select('*', { count: 'exact', head: true })
          .eq('post_id', postId);

        // Determine author name
        let authorName = 'Anonymous';
        if (profileData) {
          if (profileData.account_type === 'NGO' || profileData.account_type === 'NGO / Organization') {
            authorName = profileData.organization_name || 'Organization';
          } else {
            authorName = `${profileData.first_name} ${profileData.last_name}`.trim();
          }
        }

        const transformedPost: Post = {
          id: postData.id,
          user_id: postData.user_id,
          title: postData.title,
          content: postData.content,
          created_at: postData.created_at,
          updated_at: postData.updated_at,
          likes: likeCount || 0,
          comments: commentCount || 0,
          shares: postData.shares || 0,
          image_url: postData.image_url,
          video_url: postData.video_url,
          is_job_post: postData.is_job_post || false,
          author_name: authorName,
          avatar_url: profileData?.avatar_url,
        };

        setPost(transformedPost);
      } catch (err) {
        setError("Error loading post");
      } finally {
        setLoading(false);
      }
    }

    fetchPost();
  }, [postId]);

  if (loading) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="max-w-4xl mx-auto px-4 py-8">
          <div className="text-center text-muted-foreground py-12">Loading post...</div>
        </div>
      </div>
    );
  }

  if (error || !post) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="max-w-4xl mx-auto px-4 py-8">
          <div className="text-center text-destructive py-12">
            {error || "Post not found"}
          </div>
          <div className="text-center">
            <Link href="/dashboard">
              <Button variant="outline">Back to Feed</Button>
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* Back Button */}
        <div className="mb-6">
          <Link href="/dashboard">
            <Button variant="ghost" className="flex items-center space-x-2 text-foreground hover:bg-accent">
              <ArrowLeft className="h-4 w-4" />
              <span>Back to Feed</span>
            </Button>
          </Link>
        </div>

        {/* Post */}
        <div className="bg-card rounded-lg shadow-sm border border-border">
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
            likes={post.likes}
            comments={post.comments}
            shares={post.shares}
            avatar={post.avatar_url}
            isVerified={false}
            imageUrl={post.image_url}
            videoUrl={post.video_url}
            isJobPost={post.is_job_post}
            currentUserId={currentUserId || undefined}
            postUserId={post.user_id}
            onPostUpdate={(postId, newTitle, newContent) => {
              setPost(prev => prev ? {
                ...prev,
                title: newTitle,
                content: newContent
              } : null);
            }}
            onPostDelete={(postId) => {
              router.push('/dashboard');
            }}
          />
        </div>
      </div>
    </div>
  );
} 