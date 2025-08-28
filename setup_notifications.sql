-- Notifications setup for follows, post likes, comments, and comment likes
-- Run this in your Supabase SQL Editor

-- 1) Create notifications table
CREATE TABLE IF NOT EXISTS public.notifications (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  recipient_id uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  actor_id uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  type text NOT NULL CHECK (type IN ('follow','post_like','comment','comment_like')),
  post_id uuid NULL REFERENCES public.posts(id) ON DELETE CASCADE,
  comment_id uuid NULL REFERENCES public.comments(id) ON DELETE CASCADE,
  metadata jsonb NULL,
  created_at timestamptz NOT NULL DEFAULT timezone('utc'::text, now()),
  read_at timestamptz NULL
);

-- 2) Indexes
CREATE INDEX IF NOT EXISTS notifications_recipient_id_idx ON public.notifications(recipient_id);
CREATE INDEX IF NOT EXISTS notifications_created_at_idx ON public.notifications(created_at DESC);
CREATE INDEX IF NOT EXISTS notifications_unread_idx ON public.notifications(recipient_id, read_at) WHERE read_at IS NULL;

-- 3) RLS
ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;

-- Allow recipients to read their notifications
DROP POLICY IF EXISTS "Users can read their notifications" ON public.notifications;
CREATE POLICY "Users can read their notifications" ON public.notifications
  FOR SELECT USING (auth.uid() = recipient_id);

-- Allow recipients to mark notifications as read
DROP POLICY IF EXISTS "Users can update their notifications" ON public.notifications;
CREATE POLICY "Users can update their notifications" ON public.notifications
  FOR UPDATE USING (auth.uid() = recipient_id) WITH CHECK (auth.uid() = recipient_id);

-- NOTE: We do NOT create a general INSERT policy; inserts are performed via SECURITY DEFINER functions called by triggers.

-- 4) Helper function to safely create a notification (bypasses RLS as table owner)
CREATE OR REPLACE FUNCTION public.create_notification(
  p_recipient_id uuid,
  p_actor_id uuid,
  p_type text,
  p_post_id uuid DEFAULT NULL,
  p_comment_id uuid DEFAULT NULL,
  p_metadata jsonb DEFAULT NULL
) RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
  -- Guard: do not notify self
  IF p_recipient_id IS NULL OR p_actor_id IS NULL OR p_recipient_id = p_actor_id THEN
    RETURN;
  END IF;

  INSERT INTO public.notifications (recipient_id, actor_id, type, post_id, comment_id, metadata)
  VALUES (p_recipient_id, p_actor_id, p_type, p_post_id, p_comment_id, p_metadata);
END;
$$;

-- 5) Triggers for follows, likes, and comments

-- Follow notifications: when someone follows you
CREATE OR REPLACE FUNCTION public.notify_on_follow()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
  PERFORM public.create_notification(
    NEW.following_id, -- recipient
    NEW.follower_id,  -- actor
    'follow',
    NULL,
    NULL,
    NULL
  );
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_notify_on_follow ON public.user_follows;
CREATE TRIGGER trg_notify_on_follow
AFTER INSERT ON public.user_follows
FOR EACH ROW EXECUTE FUNCTION public.notify_on_follow();

-- Post like notifications: when someone likes your post
CREATE OR REPLACE FUNCTION public.notify_on_post_like()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
  v_post_owner uuid;
BEGIN
  SELECT user_id INTO v_post_owner FROM public.posts WHERE id = NEW.post_id;
  IF v_post_owner IS NULL OR v_post_owner = NEW.user_id THEN
    RETURN NEW; -- no notify if no owner or self-like
  END IF;

  PERFORM public.create_notification(
    v_post_owner,      -- recipient (post owner)
    NEW.user_id,       -- actor (liker)
    'post_like',
    NEW.post_id,
    NULL,
    NULL
  );
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_notify_on_post_like ON public.likes;
CREATE TRIGGER trg_notify_on_post_like
AFTER INSERT ON public.likes
FOR EACH ROW EXECUTE FUNCTION public.notify_on_post_like();

-- Comment notifications: when someone comments on your post
CREATE OR REPLACE FUNCTION public.notify_on_comment()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
  v_post_owner uuid;
BEGIN
  SELECT user_id INTO v_post_owner FROM public.posts WHERE id = NEW.post_id;
  IF v_post_owner IS NULL OR v_post_owner = NEW.user_id THEN
    RETURN NEW; -- no notify if commenter is post owner
  END IF;

  PERFORM public.create_notification(
    v_post_owner,      -- recipient (post owner)
    NEW.user_id,       -- actor (commenter)
    'comment',
    NEW.post_id,
    NEW.id,
    jsonb_build_object('excerpt', left(coalesce(NEW.content,''), 140))
  );
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_notify_on_comment ON public.comments;
CREATE TRIGGER trg_notify_on_comment
AFTER INSERT ON public.comments
FOR EACH ROW EXECUTE FUNCTION public.notify_on_comment();

-- Optional: Comment likes table and notifications (creates table if not exists)
CREATE TABLE IF NOT EXISTS public.comment_likes (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  comment_id uuid NOT NULL REFERENCES public.comments(id) ON DELETE CASCADE,
  user_id uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  created_at timestamptz NOT NULL DEFAULT timezone('utc'::text, now()),
  UNIQUE(comment_id, user_id)
);

ALTER TABLE public.comment_likes ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can create their own comment likes" ON public.comment_likes;
CREATE POLICY "Users can create their own comment likes" ON public.comment_likes
  FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own comment likes" ON public.comment_likes;
CREATE POLICY "Users can delete their own comment likes" ON public.comment_likes
  FOR DELETE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can view comment likes" ON public.comment_likes;
CREATE POLICY "Users can view comment likes" ON public.comment_likes
  FOR SELECT USING (true);

CREATE INDEX IF NOT EXISTS comment_likes_comment_id_idx ON public.comment_likes(comment_id);
CREATE INDEX IF NOT EXISTS comment_likes_user_id_idx ON public.comment_likes(user_id);

-- Comment like notifications: when someone likes your comment
CREATE OR REPLACE FUNCTION public.notify_on_comment_like()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
  v_comment_owner uuid;
BEGIN
  SELECT user_id INTO v_comment_owner FROM public.comments WHERE id = NEW.comment_id;
  IF v_comment_owner IS NULL OR v_comment_owner = NEW.user_id THEN
    RETURN NEW; -- no notify if self-like
  END IF;

  PERFORM public.create_notification(
    v_comment_owner,   -- recipient (comment owner)
    NEW.user_id,       -- actor (liker)
    'comment_like',
    (SELECT post_id FROM public.comments WHERE id = NEW.comment_id),
    NEW.comment_id,
    NULL
  );
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_notify_on_comment_like ON public.comment_likes;
CREATE TRIGGER trg_notify_on_comment_like
AFTER INSERT ON public.comment_likes
FOR EACH ROW EXECUTE FUNCTION public.notify_on_comment_like();

-- 6) Verification helpers
-- Count unread notifications for a user
CREATE OR REPLACE FUNCTION public.get_unread_notification_count(p_user_id uuid)
RETURNS integer
LANGUAGE sql
STABLE
AS $$
  SELECT COUNT(*)::int FROM public.notifications n
  WHERE n.recipient_id = p_user_id AND n.read_at IS NULL;
$$;


