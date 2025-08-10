import { notFound } from 'next/navigation';
import { getSupabaseClient } from '@/lib/supabaseClient';
import { Card } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import ProfileClient from './ProfileClient';

interface ProfilePageProps {
  params: { id: string };
}

async function getProfileAndPosts(userId: string) {
  const supabase = getSupabaseClient();
  // Fetch user profile
  const { data: profile, error: profileError } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', userId)
    .single();
  if (profileError || !profile) return { profile: null, posts: [] };

  // Fetch user's posts
  const { data: posts, error: postsError } = await supabase
    .from('posts')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false });
  return { profile, posts: posts || [] };
}

export default async function ProfilePage({ params }: ProfilePageProps) {
  const { id } = params;
  const { profile, posts } = await getProfileAndPosts(id);
  if (!profile) return notFound();

  const isNGO = profile.account_type === 'NGO' || profile.account_type === 'NGO / Organization';

  if (isNGO) {
    // NGO Profile Layout
    return (
      <div className="min-h-screen bg-background">
        <div className="max-w-3xl mx-auto py-10 px-4 md:px-0">
          {/* Back Button */}
          <div className="mb-6">
            <Link href="/dashboard">
              <Button variant="ghost" className="flex items-center space-x-2 text-foreground hover:bg-accent">
                <ArrowLeft className="h-4 w-4" />
                <span>Back to Feed</span>
              </Button>
            </Link>
          </div>
          
          <Card className="p-0 mb-8 overflow-hidden shadow-lg rounded-2xl border-0 bg-card">
            {/* Banner or fallback */}
            <div className="h-32 bg-gradient-to-r from-mustard to-forest-green relative">
              {profile.banner_url ? (
                <img src={profile.banner_url} alt="NGO Banner" className="object-cover w-full h-full" />
              ) : null}
              <div className="absolute left-1/2 -bottom-12 -translate-x-1/2">
                <Avatar className="w-24 h-24 ring-4 ring-mustard border-4 border-white bg-white">
                  <AvatarImage src={profile.avatar_url || '/placeholder-logo.png'} alt={profile.organization_name || 'NGO'} />
                  <AvatarFallback>{(profile.user_metadata?.displayName || profile.organization_name || 'N')[0]}</AvatarFallback>
                </Avatar>
              </div>
            </div>
            <div className="pt-16 pb-8 px-6 text-center">
              <div className="text-2xl font-bold text-foreground mb-1">{profile.user_metadata?.displayName || profile.organization_name || 'Organization'}</div>
              <div className="text-mustard font-medium text-lg mb-2">{profile.mission_statement || 'No mission statement provided.'}</div>
              <div className="text-muted-foreground mb-4">{profile.bio || profile.description || 'No description available.'}</div>
              <div className="flex flex-wrap justify-center gap-4 mb-4">
                <div className="bg-muted rounded-lg px-4 py-2 text-sm text-foreground">
                  <span className="font-semibold">Location:</span> {profile.headquarters_location || profile.location || 'No data available'}
                </div>
                {profile.show_email && profile.email && (
                  <div className="bg-muted rounded-lg px-4 py-2 text-sm text-foreground">
                    <span className="font-semibold">Email:</span> <a href={`mailto:${profile.email}`} className="text-mustard hover:underline">{profile.email}</a>
                  </div>
                )}
                {profile.show_phone && profile.phone && (
                  <div className="bg-muted rounded-lg px-4 py-2 text-sm text-foreground">
                    <span className="font-semibold">Phone:</span> {profile.phone}
                  </div>
                )}
              </div>
              <div className="flex flex-wrap justify-center gap-4 mb-4">
                {profile.website && (
                  <a href={profile.website} target="_blank" rel="noopener noreferrer" className="text-mustard hover:underline font-medium">Website</a>
                )}
                {profile.linkedin && (
                  <a href={profile.linkedin} target="_blank" rel="noopener noreferrer" className="text-mustard hover:underline font-medium">LinkedIn</a>
                )}
                {profile.twitter && (
                  <a href={profile.twitter} target="_blank" rel="noopener noreferrer" className="text-mustard hover:underline font-medium">Twitter</a>
                )}
                {profile.github && (
                  <a href={profile.github} target="_blank" rel="noopener noreferrer" className="text-mustard hover:underline font-medium">GitHub</a>
                )}
              </div>
            </div>
          </Card>
          <div className="mt-8">
            <ProfileClient profile={profile} posts={posts} ngo={true} />
          </div>
        </div>
      </div>
    );
  }

  // Default (Individual) Profile Layout
  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-2xl mx-auto py-8 px-4">
        {/* Back Button */}
        <div className="mb-6">
          <Link href="/dashboard">
            <Button variant="ghost" className="flex items-center space-x-2 text-foreground hover:bg-accent">
              <ArrowLeft className="h-4 w-4" />
              <span>Back to Feed</span>
            </Button>
          </Link>
        </div>
        
        {/* Profile Header */}
        <Card className="p-0 mb-8 overflow-hidden shadow-lg rounded-2xl border-0 bg-card">
          {/* Subtle Banner */}
          <div className="h-28 bg-gradient-to-r from-mustard to-forest-green relative">
            <div className="absolute left-1/2 -bottom-12 -translate-x-1/2">
              <Avatar className="w-24 h-24 ring-4 ring-mustard border-4 border-white bg-white">
                <AvatarImage src={profile.avatar_url || '/placeholder-user.jpg'} alt={profile.user_metadata?.displayName || `${profile.first_name} ${profile.last_name}`} />
                <AvatarFallback>{(profile.user_metadata?.displayName || `${profile.first_name} ${profile.last_name}`)[0]}</AvatarFallback>
              </Avatar>
            </div>
          </div>
          <div className="pt-16 pb-8 px-6 text-center">
            <div className="text-2xl font-bold text-foreground mb-1">{profile.user_metadata?.displayName || `${profile.first_name} ${profile.last_name}`}</div>
            {profile.title && <div className="text-mustard font-medium text-lg mb-2">{profile.title}</div>}
            {profile.bio && <div className="text-muted-foreground mb-4">{profile.bio}</div>}
            <div className="flex flex-wrap justify-center gap-4 mb-4">
              {profile.location && (
                <div className="bg-muted rounded-lg px-4 py-2 text-sm text-foreground">
                  <span className="font-semibold">Location:</span> {profile.location}
                </div>
              )}
              {profile.show_email && profile.email && (
                <div className="bg-muted rounded-lg px-4 py-2 text-sm text-foreground">
                  <span className="font-semibold">Email:</span> <a href={`mailto:${profile.email}`} className="text-mustard hover:underline">{profile.email}</a>
                </div>
              )}
              {profile.show_phone && profile.phone && (
                <div className="bg-muted rounded-lg px-4 py-2 text-sm text-foreground">
                  <span className="font-semibold">Phone:</span> {profile.phone}
                </div>
              )}
            </div>
            <div className="flex flex-wrap justify-center gap-4 mb-4">
              {profile.website && (
                <a href={profile.website} target="_blank" rel="noopener noreferrer" className="text-mustard hover:underline font-medium">Website</a>
              )}
              {profile.linkedin && (
                <a href={profile.linkedin} target="_blank" rel="noopener noreferrer" className="text-mustard hover:underline font-medium">LinkedIn</a>
              )}
              {profile.twitter && (
                <a href={profile.twitter} target="_blank" rel="noopener noreferrer" className="text-mustard hover:underline font-medium">Twitter</a>
              )}
              {profile.github && (
                <a href={profile.github} target="_blank" rel="noopener noreferrer" className="text-mustard hover:underline font-medium">GitHub</a>
              )}
            </div>
          </div>
        </Card>
        {/* Posts remain unchanged */}
        <div className="mt-8">
          <ProfileClient profile={profile} posts={posts} ngo={false} />
        </div>
      </div>
    </div>
  );
}
