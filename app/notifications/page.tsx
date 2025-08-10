"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { getSupabaseClient } from "@/lib/supabaseClient";
import { Header } from "../../components/header"
import { Bell } from "lucide-react"

export default function NotificationsPage() {
  const router = useRouter();

  useEffect(() => {
    async function checkAuth() {
      const supabase = getSupabaseClient();
      const { data: { user }, error } = await supabase.auth.getUser();
      
      if (error || !user) {
        router.push('/auth');
        return;
      }
    }
    checkAuth();
  }, [router]);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="max-w-4xl mx-auto px-4 py-6">
        <div className="bg-card rounded-lg shadow-sm border border-border p-6">
          <div className="flex items-center space-x-3 mb-6">
            <Bell className="h-6 w-6 text-mustard" />
            <h1 className="text-2xl font-bold text-foreground">Notifications</h1>
          </div>
          
          <div className="text-center py-12">
            <Bell className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-foreground mb-2">All Caught Up!</h3>
            <p className="text-muted-foreground">
              You don't have any notifications yet. They'll appear here when you receive them.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
} 