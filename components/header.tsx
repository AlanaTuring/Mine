"use client"

import { Search, Bell, Users, Briefcase, Home, ChevronDown, User, Settings, LogOut, Accessibility } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { useState, useEffect, useRef } from "react"
import { useRouter, usePathname } from "next/navigation"
import { getSupabaseClient } from "@/lib/supabaseClient";
import Link from "next/link"
import { useAccessibility } from "./accessibility-provider"
import { AccessibilitySettingsModal } from "./accessibility-settings-modal"

export function Header() {
  const [userProfile, setUserProfile] = useState<any>(null)
  const [showProfileMenu, setShowProfileMenu] = useState(false)
  const [showAccessibilityModal, setShowAccessibilityModal] = useState(false)
  const router = useRouter()
  const pathname = usePathname()
  const profileMenuRef = useRef<HTMLDivElement>(null)
  const { settings } = useAccessibility()

  // --- SEARCH STATE ---
  const [searchValue, setSearchValue] = useState("");
  const [searchResults, setSearchResults] = useState({ people: [], jobs: [], companies: [], posts: [] });
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [showDropdown, setShowDropdown] = useState(false);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const searchDropdownRef = useRef<HTMLDivElement>(null);
  const searchTimeout = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    async function fetchUserProfile() {
      const supabase = getSupabaseClient();
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        const { data: profile } = await supabase
          .from('profiles')
          .select('*')
          .eq('id', user.id)
          .single();
        setUserProfile(profile);
      }
    }
    fetchUserProfile();
  }, []);

  // --- SEARCH HANDLER ---
  useEffect(() => {
    if (!searchValue.trim()) {
      setSearchResults({ people: [], jobs: [], companies: [], posts: [] });
      setShowDropdown(false);
      setSearchLoading(false);
      setSearchError(null);
      return;
    }
    setSearchLoading(true);
    setSearchError(null);
    setShowDropdown(true);
    if (searchTimeout.current) clearTimeout(searchTimeout.current);
    searchTimeout.current = setTimeout(async () => {
      try {
        const supabase = getSupabaseClient();
        // People (profiles, not NGOs)
        const { data: people, error: peopleError } = await supabase
          .from('profiles')
          .select('id, first_name, last_name, job_title, avatar_url, account_type')
          .or(`first_name.ilike.%${searchValue}%,last_name.ilike.%${searchValue}%,job_title.ilike.%${searchValue}%`)
          .neq('account_type', 'NGO')
          .limit(3);
        // Companies (NGOs/Organizations)
        const { data: companies, error: companiesError } = await supabase
          .from('profiles')
          .select('id, organization_name, avatar_url, account_type')
          .or(`organization_name.ilike.%${searchValue}%`)
          .in('account_type', ['NGO', 'NGO / Organization'])
          .limit(3);
        // Jobs (posts with is_job_post)
        const { data: jobs, error: jobsError } = await supabase
          .from('posts')
          .select('id, title, job_metadata, created_at')
          .eq('is_job_post', true)
          .or(`title.ilike.%${searchValue}%,content.ilike.%${searchValue}%,job_metadata->>company_name.ilike.%${searchValue}%`)
          .order('created_at', { ascending: false })
          .limit(3);
        // Posts (not jobs)
        const { data: posts, error: postsError } = await supabase
          .from('posts')
          .select('id, title, content, created_at')
          .eq('is_job_post', false)
          .or(`title.ilike.%${searchValue}%,content.ilike.%${searchValue}%`)
          .order('created_at', { ascending: false })
          .limit(3);
        if (peopleError || companiesError || jobsError || postsError) {
          setSearchError("Error searching. Try again.");
        } else {
          setSearchResults({
            people: people || [],
            jobs: jobs || [],
            companies: companies || [],
            posts: posts || [],
          });
        }
      } catch (err) {
        setSearchError("Error searching. Try again.");
      } finally {
        setSearchLoading(false);
      }
    }, 350); // debounce
    // eslint-disable-next-line
  }, [searchValue]);

  // --- CLOSE DROPDOWN ON OUTSIDE CLICK ---
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        searchDropdownRef.current &&
        !searchDropdownRef.current.contains(event.target as Node) &&
        searchInputRef.current &&
        !searchInputRef.current.contains(event.target as Node)
      ) {
        setShowDropdown(false);
      }
    }
    if (showDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
    } else {
      document.removeEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showDropdown]);

  // Close profile menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (profileMenuRef.current && !profileMenuRef.current.contains(event.target as Node)) {
        setShowProfileMenu(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleSignOut = async () => {
    const supabase = getSupabaseClient();
    await supabase.auth.signOut();
    router.push('/auth');
  };

  // Determine active tab based on current pathname
  const getActiveTab = () => {
    if (pathname === '/dashboard') return 'home'
    if (pathname === '/network') return 'network'
    if (pathname === '/jobs') return 'jobs'
    if (pathname === '/notifications') return 'notifications'
    return 'home'
  }

  const activeTab = getActiveTab()

  return (
    <>
      <header className="bg-background border-b border-border sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between h-16 min-w-0">
            {/* Logo and Search */}
            <div className="flex items-center space-x-4 lg:space-x-6 flex-1 min-w-0">
              <Link href="/dashboard" className="text-xl lg:text-2xl font-bold bg-gradient-to-r from-mustard to-forest-green bg-clip-text text-transparent hover:opacity-80 transition-opacity flex-shrink-0">
                Access<span className="text-foreground">Able</span>
              </Link>
              <div className="relative group flex-1 max-w-md lg:max-w-lg">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4 group-focus-within:text-mustard transition-colors" />
                <Input
                  ref={searchInputRef}
                  value={searchValue}
                  onChange={e => setSearchValue(e.target.value)}
                  onFocus={() => searchValue && setShowDropdown(true)}
                  onKeyDown={e => {
                    if (e.key === 'Enter') {
                      setShowDropdown(false);
                      router.push(`/search?q=${encodeURIComponent(searchValue)}`);
                    }
                  }}
                  placeholder="Search for jobs, people, companies..."
                  className="pl-10 w-full bg-muted border-border focus:bg-background focus:border-mustard transition-all duration-200 text-foreground placeholder:text-muted-foreground"
                  aria-label="Search for jobs, people, companies"
                  autoComplete="off"
                />
                {/* DROPDOWN */}
                {showDropdown && searchValue && (
                  <div ref={searchDropdownRef} className="absolute left-0 right-0 mt-2 bg-card border border-border rounded-lg shadow-lg z-50 max-h-96 overflow-y-auto">
                    {searchLoading ? (
                      <div className="p-4 text-center text-muted-foreground text-sm">Searching...</div>
                    ) : searchError ? (
                      <div className="p-4 text-center text-destructive text-sm">{searchError}</div>
                    ) : (
                      <>
                        {['people', 'companies', 'jobs', 'posts'].map(type => (
                          searchResults[type].length > 0 && (
                            <div key={type} className="border-b last:border-b-0 border-border">
                              <div className="px-4 pt-3 pb-1 text-xs font-semibold text-muted-foreground uppercase">{type.charAt(0).toUpperCase() + type.slice(1)}</div>
                              {searchResults[type].map((item: any) => {
                                if (type === 'people') {
                                  return (
                                    <Link key={item.id} href={`/profile/${item.id}`} onClick={() => setShowDropdown(false)} className="block px-4 py-2 hover:bg-accent transition-colors">
                                      <div className="flex items-center space-x-3">
                                        <Avatar className="h-8 w-8 flex-shrink-0">
                                          <AvatarImage src={item.avatar_url} alt={`${item.first_name} ${item.last_name}`} />
                                          <AvatarFallback className="bg-gradient-to-br from-mustard to-forest-green text-white text-xs">
                                            {item.first_name?.[0] || ''}{item.last_name?.[0] || ''}
                                          </AvatarFallback>
                                        </Avatar>
                                        <div className="flex-1 min-w-0">
                                          <span className="font-medium text-foreground block truncate">{item.first_name} {item.last_name}</span>
                                          {item.job_title && <span className="text-xs text-muted-foreground block truncate">{item.job_title}</span>}
                                        </div>
                                      </div>
                                    </Link>
                                  );
                                }
                                if (type === 'companies') {
                                  return (
                                    <Link key={item.id} href={`/profile/${item.id}`} onClick={() => setShowDropdown(false)} className="block px-4 py-2 hover:bg-accent transition-colors">
                                      <div className="flex items-center space-x-3">
                                        <Avatar className="h-8 w-8 flex-shrink-0">
                                          <AvatarImage src={item.avatar_url} alt={item.organization_name} />
                                          <AvatarFallback className="bg-gradient-to-br from-mustard to-forest-green text-white text-xs">
                                            {item.organization_name?.[0] || 'O'}
                                          </AvatarFallback>
                                        </Avatar>
                                        <div className="flex-1 min-w-0">
                                          <span className="font-medium text-foreground block truncate">{item.organization_name}</span>
                                          <span className="text-xs text-muted-foreground block">Company</span>
                                        </div>
                                      </div>
                                    </Link>
                                  );
                                }
                                if (type === 'jobs') {
                                  return (
                                    <Link key={item.id} href={`/jobs`} onClick={() => setShowDropdown(false)} className="block px-4 py-2 hover:bg-accent transition-colors">
                                      <div className="flex items-center space-x-3">
                                        <div className="h-8 w-8 flex-shrink-0 bg-mustard rounded-full flex items-center justify-center">
                                          <svg className="h-4 w-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2-2v2m8 0V6a2 2 0 012 2v6a2 2 0 01-2 2H8a2 2 0 01-2-2V8a2 2 0 012-2V6" />
                                          </svg>
                                        </div>
                                        <div className="flex-1 min-w-0">
                                          <span className="font-medium text-foreground block truncate">{item.title}</span>
                                          {item.job_metadata?.company_name && <span className="text-xs text-muted-foreground block truncate">{item.job_metadata.company_name}</span>}
                                        </div>
                                      </div>
                                    </Link>
                                  );
                                }
                                if (type === 'posts') {
                                  return (
                                    <Link key={item.id} href={`/posts/${item.id}`} onClick={() => setShowDropdown(false)} className="block px-4 py-2 hover:bg-accent transition-colors">
                                      <div className="flex items-center space-x-3">
                                        <div className="h-8 w-8 flex-shrink-0 bg-forest-green rounded-full flex items-center justify-center">
                                          <svg className="h-4 w-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                          </svg>
                                        </div>
                                        <div className="flex-1 min-w-0">
                                          <span className="font-medium text-foreground block truncate">{item.title || item.content?.slice(0, 30) + '...'}</span>
                                          <span className="text-xs text-muted-foreground block">Post</span>
                                        </div>
                                      </div>
                                    </Link>
                                  );
                                }
                                return null;
                              })}
                            </div>
                          )
                        ))}
                        {/* If no results at all */}
                        {Object.values(searchResults).every(arr => arr.length === 0) && (
                          <div className="p-4 text-center text-muted-foreground text-sm">No results found.</div>
                        )}
                        {/* See all results */}
                        <div className="border-t border-border">
                          <button
                            className="w-full text-mustard font-semibold py-2 hover:bg-accent transition-colors text-sm"
                            onClick={() => {
                              setShowDropdown(false);
                              router.push(`/search?q=${encodeURIComponent(searchValue)}`);
                            }}
                          >
                            See all results for "{searchValue}"
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex items-center space-x-1 lg:space-x-2 flex-shrink-0">
              {[
                { icon: Home, label: "Home", key: "home", badge: null, href: "/dashboard" },
                { icon: Users, label: "Network", key: "network", badge: null, href: "/network" },
                { icon: Briefcase, label: "Jobs", key: "jobs", badge: null, href: "/jobs" },
                { icon: Bell, label: "Notifications", key: "notifications", badge: null, href: "/notifications" },
              ].map(({ icon: Icon, label, key, badge, href }) => (
                <Link key={key} href={href}>
                  <Button
                    variant="ghost"
                    size="sm"
                    className={`flex flex-col items-center p-2 lg:p-3 h-auto relative transition-all duration-200 hover:bg-accent min-w-0 ${
                      activeTab === key ? "text-mustard bg-accent" : "text-foreground"
                    }`}
                    aria-label={`Navigate to ${label}`}
                  >
                    <div className="relative">
                      <Icon className="h-4 w-4 lg:h-5 lg:w-5" />
                      {badge && (
                        <Badge className="absolute -top-2 -right-2 h-4 w-4 p-0 text-xs bg-mustard hover:bg-mustard text-foreground">
                          {badge}
                        </Badge>
                      )}
                    </div>
                    <span className="text-xs mt-1 font-medium hidden sm:block">{label}</span>
                    {activeTab === key && (
                      <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-mustard rounded-full" />
                    )}
                  </Button>
                </Link>
              ))}

              {/* Accessibility Settings Button - hide on /profile */}
              {pathname !== "/profile" && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="flex flex-col items-center p-2 lg:p-3 h-auto hover:bg-accent transition-colors min-w-0 text-foreground"
                  onClick={() => setShowAccessibilityModal(true)}
                  aria-label="Accessibility settings"
                >
                  <Accessibility className="h-4 w-4 lg:h-5 lg:w-5" />
                  <span className="text-xs mt-1 font-medium hidden sm:block">Access</span>
                </Button>
              )}

              {/* Profile Menu */}
              <div className="ml-2 lg:ml-4 pl-2 lg:pl-4 border-l border-border relative">
                <Button
                  variant="ghost"
                  size="sm"
                  className="flex items-center space-x-2 p-2 h-auto hover:bg-accent transition-colors min-w-0"
                  onClick={() => setShowProfileMenu(!showProfileMenu)}
                  aria-label="Profile menu"
                >
                  <Avatar className="h-6 w-6 lg:h-8 lg:w-8 ring-2 ring-mustard flex-shrink-0">
                    <AvatarImage src={userProfile?.avatar_url || "/placeholder.svg?height=32&width=32"} alt={`${userProfile?.first_name || 'User'} profile picture`} />
                    <AvatarFallback className="bg-gradient-to-br from-mustard to-forest-green text-white text-xs lg:text-sm">
                      {userProfile ? `${userProfile.first_name?.[0] || ''}${userProfile.last_name?.[0] || ''}` : 'ME'}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex flex-col items-start min-w-0">
                    <span className="text-xs font-medium text-foreground truncate">Me</span>
                  </div>
                  <ChevronDown className="h-3 w-3 text-muted-foreground flex-shrink-0" aria-hidden="true" />
                </Button>

                {/* Profile Dropdown Menu */}
                {showProfileMenu && (
                  <div className="absolute right-0 mt-2 w-48 bg-card rounded-md shadow-lg border border-border z-50" ref={profileMenuRef}>
                    <div className="py-1">
                      <Link href="/profile">
                        <Button
                          variant="ghost"
                          className="w-full justify-start px-4 py-2 text-sm text-foreground hover:bg-accent"
                          onClick={() => setShowProfileMenu(false)}
                          aria-label="Edit profile"
                        >
                          <Settings className="h-4 w-4 mr-2" aria-hidden="true" />
                          Edit Profile
                        </Button>
                      </Link>
                      <hr className="my-1 border-border" />
                      <Button
                        variant="ghost"
                        className="w-full justify-start px-4 py-2 text-sm text-destructive hover:bg-destructive/10"
                        onClick={handleSignOut}
                        aria-label="Sign out of account"
                      >
                        <LogOut className="h-4 w-4 mr-2" aria-hidden="true" />
                        Sign Out
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </nav>
          </div>
        </div>
      </header>

      {/* Accessibility Settings Modal - hide on /profile */}
      {pathname !== "/profile" && (
        <AccessibilitySettingsModal 
          isOpen={showAccessibilityModal}
          onClose={() => setShowAccessibilityModal(false)}
        />
      )}
    </>
  )
}
