"use client";
import { useEffect, useState, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Link from "next/link";
import { getSupabaseClient } from "@/lib/supabaseClient";
import { Header } from "@/components/header";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";

function SearchResults() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const query = searchParams.get("q") || "";
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState({ people: [], jobs: [], companies: [], posts: [] });

  useEffect(() => {
    if (!query.trim()) {
      setResults({ people: [], jobs: [], companies: [], posts: [] });
      setError(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const supabase = getSupabaseClient();
        // People (profiles, not NGOs)
        const { data: people, error: peopleError } = await supabase
          .from('profiles')
          .select('id, first_name, last_name, job_title, avatar_url, account_type')
          .or(`first_name.like.%${query}%,last_name.like.%${query}%,job_title.like.%${query}%`)
          .neq('account_type', 'NGO');
        // Companies (NGOs/Organizations)
        const { data: companies, error: companiesError } = await supabase
          .from('profiles')
          .select('id, organization_name, avatar_url, account_type')
          .or(`organization_name.like.%${query}%`)
          .in('account_type', ['NGO', 'NGO / Organization']);
        // Jobs (posts with is_job_post)
        const { data: jobs, error: jobsError } = await supabase
          .from('posts')
          .select('id, title, job_metadata, created_at')
          .eq('is_job_post', true)
          .or(`title.like.%${query}%,content.like.%${query}%,job_metadata->>company_name.like.%${query}%`)
          .order('created_at', { ascending: false });
        // Posts (not jobs)
        const { data: posts, error: postsError } = await supabase
          .from('posts')
          .select('id, title, content, created_at')
          .eq('is_job_post', false)
          .or(`title.like.%${query}%,content.like.%${query}%`)
          .order('created_at', { ascending: false });
        if (peopleError || companiesError || jobsError || postsError) {
          setError("Error searching. Try again.");
        } else {
          setResults({
            people: people || [],
            jobs: jobs || [],
            companies: companies || [],
            posts: posts || [],
          });
        }
      } catch (err) {
        setError("Error searching. Try again.");
      } finally {
        setLoading(false);
      }
    })();
  }, [query]);

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
        
        <h1 className="text-2xl font-bold mb-6 text-foreground">Search Results for "{query}"</h1>
        {loading ? (
          <div className="text-center text-muted-foreground py-12">Searching...</div>
        ) : error ? (
          <div className="text-center text-destructive py-12">{error}</div>
        ) : (
          <>
            {['people', 'companies', 'jobs', 'posts'].map(type => (
              <div key={type} className="mb-8">
                <h2 className="text-lg font-semibold mb-2 text-mustard">{type.charAt(0).toUpperCase() + type.slice(1)}</h2>
                {results[type].length === 0 ? (
                  <div className="text-muted-foreground text-sm mb-4">No {type} found.</div>
                ) : (
                  <div className="space-y-2">
                    {results[type].map((item: any) => {
                      if (type === 'people') {
                        return (
                          <Link key={item.id} href={`/profile/${item.id}`} className="block px-4 py-2 rounded hover:bg-accent transition-colors">
                            <div className="flex items-center space-x-3">
                              <Avatar className="h-10 w-10 flex-shrink-0">
                                <AvatarImage src={item.avatar_url} alt={`${item.first_name} ${item.last_name}`} />
                                <AvatarFallback className="bg-gradient-to-br from-mustard to-forest-green text-white text-sm">
                                  {item.first_name?.[0] || ''}{item.last_name?.[0] || ''}
                                </AvatarFallback>
                              </Avatar>
                              <div className="flex-1 min-w-0">
                                <span className="font-medium text-foreground block">{item.first_name} {item.last_name}</span>
                                {item.job_title && <span className="text-sm text-muted-foreground block">{item.job_title}</span>}
                              </div>
                            </div>
                          </Link>
                        );
                      }
                      if (type === 'companies') {
                        return (
                          <Link key={item.id} href={`/profile/${item.id}`} className="block px-4 py-2 rounded hover:bg-accent transition-colors">
                            <div className="flex items-center space-x-3">
                              <Avatar className="h-10 w-10 flex-shrink-0">
                                <AvatarImage src={item.avatar_url} alt={item.organization_name} />
                                <AvatarFallback className="bg-gradient-to-br from-mustard to-forest-green text-white text-sm">
                                  {item.organization_name?.[0] || 'O'}
                                </AvatarFallback>
                              </Avatar>
                              <div className="flex-1 min-w-0">
                                <span className="font-medium text-foreground block">{item.organization_name}</span>
                                <span className="text-sm text-muted-foreground block">Company</span>
                              </div>
                            </div>
                          </Link>
                        );
                      }
                      if (type === 'jobs') {
                        return (
                          <Link key={item.id} href={`/jobs`} className="block px-4 py-2 rounded hover:bg-accent transition-colors">
                            <div className="flex items-center space-x-3">
                              <div className="h-10 w-10 flex-shrink-0 bg-mustard rounded-full flex items-center justify-center">
                                <svg className="h-5 w-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2-2v2m8 0V6a2 2 0 012 2v6a2 2 0 01-2 2H8a2 2 0 01-2-2V8a2 2 0 012-2V6" />
                                </svg>
                              </div>
                              <div className="flex-1 min-w-0">
                                <span className="font-medium text-foreground block">{item.title}</span>
                                {item.job_metadata?.company_name && <span className="text-sm text-muted-foreground block">{item.job_metadata.company_name}</span>}
                              </div>
                            </div>
                          </Link>
                        );
                      }
                      if (type === 'posts') {
                        return (
                          <Link key={item.id} href={`/posts/${item.id}`} className="block px-4 py-2 rounded hover:bg-accent transition-colors">
                            <div className="flex items-center space-x-3">
                              <div className="h-10 w-10 flex-shrink-0 bg-forest-green rounded-full flex items-center justify-center">
                                <svg className="h-5 w-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                              </div>
                              <div className="flex-1 min-w-0">
                                <span className="font-medium text-foreground block">{item.title || item.content?.slice(0, 30) + '...'}</span>
                                <span className="text-sm text-muted-foreground block">Post</span>
                              </div>
                            </div>
                          </Link>
                        );
                      }
                      return null;
                    })}
                  </div>
                )}
              </div>
            ))}
            {Object.values(results).every(arr => arr.length === 0) && (
              <div className="text-center text-muted-foreground py-12">No results found.</div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-background flex items-center justify-center"><div className="text-center text-muted-foreground">Loading search...</div></div>}>
      <SearchResults />
    </Suspense>
  );
} 