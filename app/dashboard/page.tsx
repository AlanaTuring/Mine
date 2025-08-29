'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { getSupabaseClient } from '@/lib/supabaseClient';
import { Header } from "../../components/header"
import { PostCreation } from "../../components/post-creation"
import { NGOProfileSidebar } from "../../components/NGOProfileSidebar"
import { JobSidebar } from "../../components/job-sidebar"
import PostsList from '@/components/PostsList';
import { FollowingPosts } from '@/components/FollowingPosts';
import { Loader2 } from "lucide-react";

export default function DashboardPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [recLoading, setRecLoading] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<Array<{
    id: string;
    title: string;
    description: string;
    job_metadata?: any;
    score?: number;
  }>>([]);

  const handlePostCreated = () => {
    // Trigger a refresh of the posts
    setRefreshKey(prev => prev + 1);
  };

  useEffect(() => {
    setMounted(true);
    async function checkAuth() {
      const supabase = getSupabaseClient();
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        router.push('/auth');
      } else {
        setCurrentUserId(user.id);
        setLoading(false);
      }
    }
    checkAuth();
  }, [router]);

  useEffect(() => {
    if (!mounted || !currentUserId) return;
    const controller = new AbortController();
    const fetchRecommendations = async () => {
      try {
        setRecError(null);
        setRecLoading(true);
        const res = await fetch(`/api/recommendations?n=5`, { signal: controller.signal });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `Failed to fetch recommendations (${res.status})`);
        }
        const json = await res.json();
        setRecommendations(Array.isArray(json?.recommendations) ? json.recommendations : []);
      } catch (err: any) {
        if (err?.name !== 'AbortError') {
          setRecError(err?.message || 'Failed to load recommendations');
        }
      } finally {
        setRecLoading(false);
      }
    };
    fetchRecommendations();
    return () => controller.abort();
  }, [mounted, currentUserId]);

  // Prevent hydration mismatch by not rendering until mounted
  if (!mounted) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="animate-spin h-12 w-12 text-mustard mx-auto mb-4" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="animate-spin h-12 w-12 text-mustard mx-auto mb-4" />
          <p className="text-muted-foreground">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background relative">
      <Header />
      <div className="max-w-7xl mx-auto px-2 sm:px-4 py-4 sm:py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-6">
          {/* Left Sidebar - Hidden on mobile, shown on large screens */}
          <div className="hidden lg:block lg:col-span-3">
            <NGOProfileSidebar />
          </div>

          {/* Main Feed - Full width on mobile, 6 columns on large screens */}
          <div className="lg:col-span-6 space-y-4 sm:space-y-6">
            <PostCreation onPostCreated={handlePostCreated} />
            <FollowingPosts currentUserId={currentUserId} key={refreshKey} />
          </div>

          {/* Right Sidebar - Hidden on mobile, shown on large screens */}
          <div className="hidden lg:block lg:col-span-3">
            <div className="bg-muted rounded-lg shadow-sm border border-border p-4 sm:p-6">
              <h3 className="text-lg font-semibold text-foreground mb-4">Recommended Jobs</h3>
              {recLoading && (
                <div className="flex items-center justify-center py-6 text-muted-foreground">
                  <Loader2 className="animate-spin h-5 w-5 mr-2" />
                  <span className="text-sm">Fetching recommendations…</span>
                </div>
              )}
              {!recLoading && recError && (
                <div className="text-center py-6 text-destructive text-sm">
                  {recError}
                </div>
              )}
              {!recLoading && !recError && (
                <div className="space-y-4">
                  {recommendations.length === 0 ? (
                    <div className="text-center py-6 text-muted-foreground text-sm">
                      No recommendations yet.
                    </div>
                  ) : (
                    recommendations.map((rec) => (
                      <div key={rec.id} className="p-3 rounded-md border border-border bg-background">
                        <div className="text-sm font-medium text-foreground">{rec.title}</div>
                        {rec.description ? (
                          <p className="text-xs text-muted-foreground mt-1 line-clamp-3">{rec.description}</p>
                        ) : null}
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 