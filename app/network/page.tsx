"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { getSupabaseClient } from "@/lib/supabaseClient";
import { Header } from "../../components/header"
import { Users, UserPlus, MapPin, Briefcase } from "lucide-react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

interface UserProfile {
  id: string
  first_name: string
  last_name: string
  job_title?: string
  current_location?: string
  avatar_url?: string
  bio?: string
  organization_name?: string
  account_type?: string
}

export default function NetworkPage() {
  const router = useRouter();
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);

  useEffect(() => {
    async function checkAuth() {
      const supabase = getSupabaseClient();
      const { data: { user }, error } = await supabase.auth.getUser();
      
      if (error || !user) {
        router.push('/auth');
        return;
      }
      
      setCurrentUserId(user.id);
      fetchUsers(user.id);
    }
    checkAuth();
  }, [router]);

  const fetchUsers = async (excludeUserId: string) => {
    try {
      const supabase = getSupabaseClient();
      const { data, error } = await supabase
        .from('profiles')
        .select('id, first_name, last_name, job_title, current_location, avatar_url, bio, organization_name, account_type')
        .neq('id', excludeUserId)
        .order('first_name', { ascending: true });

      if (error) throw error;
      setUsers(data || []);
    } catch (error) {
      console.error('Error fetching users:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleViewProfile = (userId: string) => {
    router.push(`/profile/${userId}`);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="max-w-4xl mx-auto px-4 py-6">
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-mustard mx-auto"></div>
            <p className="mt-2 text-muted-foreground">Loading users...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="max-w-6xl mx-auto px-4 py-6">
        <div className="flex items-center space-x-3 mb-6">
          <Users className="h-6 w-6 text-mustard" />
          <h1 className="text-2xl font-bold text-foreground">Network</h1>
        </div>
        
        {users.length === 0 ? (
          <div className="text-center py-12">
            <UserPlus className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-foreground mb-2">Connect with Others</h3>
            <p className="text-muted-foreground mb-4">
              Discover and connect with other professionals on AccessAble
            </p>
            <div className="bg-muted rounded-lg p-4">
              <p className="text-sm text-muted-foreground">
                No other users found yet. Be the first to invite others to join AccessAble!
              </p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {users.map((user) => (
              <Card key={user.id} className="bg-card shadow-md hover:shadow-lg transition-shadow">
                <CardContent className="p-6 flex flex-col h-full">
                  <div className="flex items-center space-x-4 mb-4">
                    <Avatar className="h-16 w-16 ring-2 ring-mustard">
                      <AvatarImage src={user.avatar_url} />
                      <AvatarFallback className="bg-gradient-to-br from-mustard to-forest-green text-white text-lg">
                        {user.account_type === 'NGO' || user.account_type === 'NGO / Organization'
                          ? user.organization_name?.[0] || 'O'
                          : `${user.first_name?.[0] || ''}${user.last_name?.[0] || ''}`}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1">
                      <h3 className="font-semibold text-foreground text-lg">
                        {user.account_type === 'NGO' || user.account_type === 'NGO / Organization'
                          ? user.organization_name
                          : `${user.first_name} ${user.last_name}`}
                      </h3>
                      {user.job_title && (
                        <div className="flex items-center text-sm text-muted-foreground mb-1">
                          <Briefcase className="h-3 w-3 mr-1" />
                          {user.job_title}
                        </div>
                      )}
                      {user.current_location && (
                        <div className="flex items-center text-sm text-muted-foreground">
                          <MapPin className="h-3 w-3 mr-1" />
                          {user.current_location}
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {user.bio && (
                    <p className="text-muted-foreground text-sm mb-4 line-clamp-2">
                      {user.bio}
                    </p>
                  )}
                  
                  <div className="mt-auto">
                    <Button 
                      onClick={() => handleViewProfile(user.id)}
                      className="w-full bg-mustard hover:bg-forest-green text-white"
                    >
                      View Profile
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
} 