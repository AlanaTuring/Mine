"use client"

import { useEffect, useMemo, useState, useCallback } from "react"
import { useRouter } from "next/navigation"
import { getSupabaseClient } from "@/lib/supabaseClient";
import { Header } from "../../components/header"
import { Bell, MessageSquare, Heart, UserPlus2, Check } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import Link from "next/link"

type NotificationRow = {
  id: string
  recipient_id: string
  actor_id: string
  type: "follow" | "post_like" | "comment" | "comment_like"
  post_id?: string | null
  comment_id?: string | null
  metadata?: any
  created_at: string
  read_at?: string | null
  actor?: { id: string; first_name: string | null; last_name: string | null; avatar_url: string | null } | null
  post?: { id: string; title: string | null; content: string | null } | null
}

export default function NotificationsPage() {
  const router = useRouter();
  const [notifications, setNotifications] = useState<NotificationRow[]>([])
  const [loading, setLoading] = useState(true)
  const [marking, setMarking] = useState(false)

  const supabase = useMemo(() => getSupabaseClient(), [])

  const fetchNotifications = useCallback(async () => {
    setLoading(true)
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) {
      router.push('/auth')
      return
    }

    // fetch last 100 notifications with actor profile basic info
    const { data, error } = await supabase
      .from('notifications')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(100)

    if (!error && data) {
      // hydrate actor info in parallel
      const actorIds = Array.from(new Set(data.map(n => n.actor_id)))
      const { data: actors } = await supabase
        .from('profiles')
        .select('id, first_name, last_name, avatar_url')
        .in('id', actorIds)

      // hydrate post info for post-related notifications
      const postIds = Array.from(new Set(data.map(n => n.post_id).filter(Boolean))) as string[]
      let posts: { id: string; title: string | null; content: string | null }[] | null = null
      if (postIds.length > 0) {
        const { data: postsData } = await supabase
          .from('posts')
          .select('id, title, content')
          .in('id', postIds)
        posts = postsData || []
      }

      const actorMap = new Map((actors || []).map(a => [a.id, a]))
      const postMap = new Map((posts || []).map(p => [p.id, p]))
      setNotifications(
        data.map(n => ({
          ...n,
          actor: actorMap.get(n.actor_id) || null,
          post: n.post_id ? (postMap.get(n.post_id) || null) : null,
        }))
      )
    }
    setLoading(false)
  }, [router, supabase])

  useEffect(() => {
    fetchNotifications()
  }, [fetchNotifications])

  // realtime subscription to new notifications for the current user
  useEffect(() => {
    let channel: any
    (async () => {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) return
      channel = supabase
        .channel(`notifications:${user.id}`)
        .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'notifications', filter: `recipient_id=eq.${user.id}` }, () => {
          fetchNotifications()
        })
        .subscribe()
    })()
    return () => {
      if (channel) supabase.removeChannel(channel)
    }
  }, [fetchNotifications, supabase])

  const markAllAsRead = async () => {
    if (marking) return
    setMarking(true)
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) return
    await supabase
      .from('notifications')
      .update({ read_at: new Date().toISOString() })
      .is('read_at', null)
    await fetchNotifications()
    setMarking(false)
  }

  const renderIcon = (type: NotificationRow['type']) => {
    switch (type) {
      case 'follow':
        return <UserPlus2 className="h-5 w-5 text-forest-green" />
      case 'post_like':
        return <Heart className="h-5 w-5 text-rose-500" />
      case 'comment_like':
        return <Heart className="h-5 w-5 text-rose-500" />
      case 'comment':
        return <MessageSquare className="h-5 w-5 text-mustard" />
    }
  }

  const renderText = (n: NotificationRow) => {
    const postLabel = () => {
      if (!n.post) return ''
      const t = (n.post.title || '').trim()
      if (t) return ` — "${t}"`
      const c = (n.post.content || '').trim()
      if (c) return ` — "${c.slice(0, 40)}${c.length > 40 ? '…' : ''}"`
      return ''
    }
    if (n.type === 'follow') return `started following you`
    if (n.type === 'post_like') return `liked your post${postLabel()}`
    if (n.type === 'comment_like') return `liked your comment${postLabel()}`
    if (n.type === 'comment') return `commented on your post${n.metadata?.excerpt ? `: "${n.metadata.excerpt}"` : ''}${postLabel()}`
    return 'New notification'
  }

  const targetLink = (n: NotificationRow) => {
    switch (n.type) {
      case 'follow':
        return `/profile/${n.actor_id}`
      case 'post_like':
      case 'comment':
      case 'comment_like':
        return n.post_id ? `/posts/${n.post_id}` : '/dashboard'
      default:
        return '/notifications'
    }
  }

  const actorInitials = (n: NotificationRow) => {
    const f = (n.actor?.first_name || '').trim()
    const l = (n.actor?.last_name || '').trim()
    const initials = `${f[0] || ''}${l[0] || ''}`
    return initials || 'U'
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="max-w-4xl mx-auto px-4 py-6">
        <div className="bg-card rounded-lg shadow-sm border border-border p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <Bell className="h-6 w-6 text-mustard" />
              <h1 className="text-2xl font-bold text-foreground">Notifications</h1>
            </div>
            {notifications.some(n => !n.read_at) && (
              <Button size="sm" onClick={markAllAsRead} disabled={marking} className="bg-mustard hover:bg-forest-green text-white">
                <Check className="h-4 w-4 mr-1" /> Mark all as read
              </Button>
            )}
          </div>

          {loading ? (
            <div className="text-center py-12 text-muted-foreground">Loading...</div>
          ) : notifications.length === 0 ? (
            <div className="text-center py-12">
              <Bell className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-foreground mb-2">All Caught Up!</h3>
              <p className="text-muted-foreground">No notifications yet.</p>
            </div>
          ) : (
            <ul className="divide-y divide-border">
              {notifications.map(n => (
                <li key={n.id} className={`flex items-start gap-3 py-4 ${!n.read_at ? 'bg-accent/30' : ''}`}>
                  <Avatar className="h-10 w-10 ring-1 ring-border">
                    <AvatarImage src={n.actor?.avatar_url || undefined} />
                    <AvatarFallback className="bg-gradient-to-br from-mustard to-forest-green text-white text-xs">
                      {actorInitials(n)}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1">
                    <div className="text-foreground text-sm flex items-center gap-2">
                      <span>
                        <span className="font-medium">{n.actor ? `${n.actor.first_name || ''} ${n.actor.last_name || ''}`.trim() : 'Someone'}</span>
                        <span className="ml-1">{renderText(n)}</span>
                      </span>
                      <span className="inline-flex items-center justify-center">{renderIcon(n.type)}</span>
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {new Date(n.created_at).toLocaleString()}
                    </div>
                    <div className="mt-2">
                      <Link href={targetLink(n)} className="text-mustard text-sm font-medium hover:underline">View</Link>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}