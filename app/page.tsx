'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { getSupabaseClient } from '@/lib/supabaseClient';
import { Loader2, Moon, Type, Volume2, VolumeX } from "lucide-react";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { useAccessibility } from "@/components/accessibility-provider";

export default function HomePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState<any>(null);
  const [mounted, setMounted] = useState(false);
  const [largeText, setLargeText] = useState(false);
  const [audioEnabled, setAudioEnabled] = useState(false);
  const { settings, updateSettings } = useAccessibility();
  const highContrast = settings.highContrast;

  useEffect(() => {
    setMounted(true);
    async function checkAuth() {
      const supabase = getSupabaseClient();
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        // Redirect authenticated users to dashboard
        router.push('/dashboard');
      } else {
        setUser(null);
        setLoading(false);
      }
    }
    checkAuth();
  }, [router]);

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
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  // Landing page for unauthenticated users
  return (
    <div className="min-h-screen bg-background dark:bg-black">
      {/* Accessibility Toolbar */}
      <div className="fixed top-4 right-4 z-50 flex space-x-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => updateSettings({ highContrast: !highContrast })}
          aria-label="Toggle dark mode"
          className="bg-card border border-border text-foreground hover:bg-accent dark:bg-card dark:border-border dark:text-foreground dark:shadow-md"
        >
          <Moon className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setLargeText(!largeText)}
          aria-label="Toggle large text"
          className="bg-card border border-border text-foreground hover:bg-accent dark:bg-card dark:border-border dark:text-foreground dark:shadow-md"
        >
          <Type className="h-4 w-4" />
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setAudioEnabled(!audioEnabled)}
          aria-label="Toggle audio assistance"
          className="bg-card border border-border text-foreground hover:bg-accent dark:bg-card dark:border-border dark:text-foreground dark:shadow-md"
        >
          {audioEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
        </Button>
      </div>

      <div className="max-w-4xl mx-auto px-4 py-16">
        <div className="text-center">
          <h1 className={`font-bold text-foreground mb-4 ${largeText ? "text-5xl" : "text-4xl"}`}>
            Welcome to <span className="bg-gradient-to-r from-mustard to-forest-green bg-clip-text text-transparent">AccessAble</span>
          </h1>
          <p className={`text-muted-foreground mb-8 ${largeText ? "text-2xl" : "text-xl"}`}>
            The professional network designed for people with disabilities
          </p>
          <div className="space-x-4">
            <Link href="/auth">
              <Button className="bg-mustard hover:bg-forest-green text-white px-8 py-3">
                Get Started
              </Button>
            </Link>
            <Link href="/auth">
              <Button variant="outline" className="px-8 py-3 border-border text-foreground hover:bg-accent">
                Sign In
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
