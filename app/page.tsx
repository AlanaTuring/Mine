'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { getSupabaseClient } from '@/lib/supabaseClient';
import { Loader2, Users, DollarSign, Eye, Accessibility, ArrowRight, Briefcase, Heart, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import Link from "next/link";
import { useAccessibility } from "@/components/accessibility-provider";

export default function HomePage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState<any>(null);
  const [mounted, setMounted] = useState(false);
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
    <div className="min-h-screen bg-background">
      {/* Skip to content link for accessibility */}
      <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-mustard text-white px-4 py-2 rounded z-50">
        Skip to main content
      </a>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center bg-gradient-to-br from-mustard/10 via-forest-green/10 to-mustard/5">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <div className="w-16 h-16 bg-gradient-to-r from-mustard to-forest-green rounded-full flex items-center justify-center">
              <Accessibility className="h-8 w-8 text-white" />
            </div>
            <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-mustard to-forest-green bg-clip-text text-transparent">
              AccessAble
            </h1>
          </div>
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">
            Welcome to AccessAble
          </h2>
          <p className="text-xl md:text-2xl mb-8 max-w-2xl mx-auto text-muted-foreground">
            A professional job network built with and for People with Disabilities (PwDs).
          </p>
          <Link href="/auth">
            <Button className="bg-gradient-to-r from-mustard to-forest-green hover:from-mustard/90 hover:to-forest-green/90 text-white px-8 py-4 text-lg font-semibold rounded-lg transition-all duration-200 transform hover:scale-105 focus:ring-4 focus:ring-mustard/30">
              Get Started
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </Link>
        </div>
      </section>

      <main id="main-content" className="max-w-6xl mx-auto px-4 py-16 space-y-16">
        {/* Definition Section */}
        <section>
          <Card className="bg-card border border-border shadow-sm hover:shadow-md transition-shadow rounded-lg">
            <CardContent className="p-8">
              <div className="flex items-start space-x-6">
                <div className="w-16 h-16 bg-gradient-to-r from-mustard to-forest-green rounded-full flex items-center justify-center flex-shrink-0">
                  <Accessibility className="h-8 w-8 text-white" />
                </div>
                <div className="flex-1">
                  <h2 className="text-3xl font-bold text-foreground mb-4">
                    What does PwDs mean?
                  </h2>
                  <p className="text-lg text-muted-foreground leading-relaxed mb-6">
                    PwDs stands for Persons with Disabilities. It includes people with visual, hearing, mobility, cognitive, neurodivergent, and other conditions. AccessAble recognizes their talent and ensures equal opportunity in the workforce.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {['Visual', 'Hearing', 'Mobility', 'Cognitive', 'Neurodivergent', 'Speech', 'Invisible'].map((category) => (
                      <span key={category} className="text-sm px-3 py-1 rounded-full bg-muted text-muted-foreground border border-border">
                        {category}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Misconceptions Section */}
        <section>
          <h2 className="text-3xl font-bold text-foreground mb-12 text-center">
            Common Misconceptions
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            {/* Card 1 */}
            <Card className="bg-card border border-border shadow-sm hover:shadow-md transition-shadow rounded-lg">
              <CardContent className="p-6">
                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-mustard/20 rounded-full flex items-center justify-center flex-shrink-0">
                    <Users className="h-6 w-6 text-mustard" aria-hidden="true" />
                  </div>
                  <div className="flex-1">
                    <div className="inline-block bg-red-100 text-red-700 text-xs px-2 py-1 rounded-full mb-3">
                      Myth
                    </div>
                    <h3 className="text-lg font-semibold text-foreground mb-2">
                      PwDs can't work effectively
                    </h3>
                    <p className="text-muted-foreground text-sm">
                      With inclusive design and accommodations, performance is comparable to anyone's.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Card 2 */}
            <Card className="bg-card border border-border shadow-sm hover:shadow-md transition-shadow rounded-lg">
              <CardContent className="p-6">
                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-forest-green/20 rounded-full flex items-center justify-center flex-shrink-0">
                    <DollarSign className="h-6 w-6 text-forest-green" aria-hidden="true" />
                  </div>
                  <div className="flex-1">
                    <div className="inline-block bg-red-100 text-red-700 text-xs px-2 py-1 rounded-full mb-3">
                      Myth
                    </div>
                    <h3 className="text-lg font-semibold text-foreground mb-2">
                      Hiring PwDs is costly
                    </h3>
                    <p className="text-muted-foreground text-sm">
                      Most accommodations cost little or nothing and improve retention.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Card 3 */}
            <Card className="bg-card border border-border shadow-sm hover:shadow-md transition-shadow rounded-lg">
              <CardContent className="p-6">
                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 bg-muted rounded-full flex items-center justify-center flex-shrink-0">
                    <Eye className="h-6 w-6 text-muted-foreground" aria-hidden="true" />
                  </div>
                  <div className="flex-1">
                    <div className="inline-block bg-red-100 text-red-700 text-xs px-2 py-1 rounded-full mb-3">
                      Myth
                    </div>
                    <h3 className="text-lg font-semibold text-foreground mb-2">
                      Disabilities are always visible
                    </h3>
                    <p className="text-muted-foreground text-sm">
                      Many are invisible—chronic pain, fatigue, learning differences.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* About Section */}
        <section>
          <Card className="bg-card border border-border shadow-sm hover:shadow-md transition-shadow rounded-lg">
            <CardContent className="p-8">
              <div className="flex items-start space-x-6">
                <div className="w-16 h-16 bg-gradient-to-r from-mustard to-forest-green rounded-full flex items-center justify-center flex-shrink-0">
                  <Briefcase className="h-8 w-8 text-white" />
                </div>
                <div className="flex-1">
                  <h2 className="text-3xl font-bold text-foreground mb-4">
                    About AccessAble
                  </h2>
                  <p className="text-lg text-muted-foreground leading-relaxed mb-6">
                    AccessAble is a platform where people with disabilities can create professional profiles, apply for jobs, and connect with inclusive employers. Our mission is to break barriers, fight stigma, and empower talent to shine.
                  </p>
                  <div className="flex flex-wrap gap-4">
                    {[
                      { icon: Heart, label: 'Accessible by default' },
                      { icon: Shield, label: 'Skills-first' },
                      { icon: Briefcase, label: 'Inclusive employers' }
                    ].map((item, index) => (
                      <div key={index} className="flex items-center space-x-2 text-muted-foreground">
                        <item.icon className="h-4 w-4 text-mustard" aria-hidden="true" />
                        <span className="text-sm">{item.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>
      </main>

      {/* Footer */}
      <footer className="py-8 bg-card border-t border-border">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <p className="text-sm text-muted-foreground">
            Contact: <a href="mailto:fhb09@mail.aub.edu" className="text-mustard hover:underline">fhb09@mail.aub.edu</a>
          </p>
        </div>
      </footer>
    </div>
  );
}
