"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { 
  Heart, 
  MessageCircle, 
  Share2, 
  MoreHorizontal, 
  Edit, 
  Trash2,
  Reply,
  ThumbsUp,
  Briefcase,
  Save,
  X
} from "lucide-react"
import { getSupabaseClient } from "@/lib/supabaseClient";
import { useRouter } from "next/navigation"
import { useAccessibility } from "@/components/accessibility-provider";
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import dayjs from 'dayjs';
import Link from 'next/link';
import { ProfileCard } from "@/components/profile/profile-card";
import { moderateText } from "@/lib/moderation";

interface Comment {
  id: string
  content: string
  author_name: string
  created_at: string
  user_id: string
}

interface FeedPostProps {
  postId: string
  author: string
  title: string
  time: string
  content: string
  likes: number
  comments: number
  shares: number
  avatar?: string
  isVerified: boolean
  imageUrl?: string
  videoUrl?: string
  isJobPost?: boolean
  currentUserId?: string
  postUserId?: string
  onPostUpdate?: (postId: string, newTitle: string, newContent: string) => void
  onPostDelete?: (postId: string) => void
}

export function FeedPost({ 
  postId, 
  author, 
  title, 
  time, 
  content, 
  likes, 
  comments, 
  shares, 
  avatar, 
  isVerified,
  imageUrl,
  videoUrl,
  isJobPost = false,
  currentUserId,
  postUserId,
  onPostUpdate,
  onPostDelete
}: FeedPostProps) {
  const [likeInProgress, setLikeInProgress] = useState(false);
  const [userLiked, setUserLiked] = useState(false);
  const [likeCount, setLikeCount] = useState(likes);
  const [commentCount, setCommentCount] = useState(comments); // Use the comments prop as initial value
  const [showComments, setShowComments] = useState(false)
  const [commentText, setCommentText] = useState("")
  const [postComments, setPostComments] = useState<Comment[]>([])
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [isEditing, setIsEditing] = useState(false);
  const [editTitle, setEditTitle] = useState(title);
  const [editContent, setEditContent] = useState(content);
  const [editingCommentId, setEditingCommentId] = useState<string | null>(null)
  const [editingCommentText, setEditingCommentText] = useState("")
  const router = useRouter()
  const { settings } = useAccessibility();
  const highContrast = settings.highContrast;
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [editPostLoading, setEditPostLoading] = useState(false);
  const [editPostError, setEditPostError] = useState<string | null>(null);
  const [editCommentLoading, setEditCommentLoading] = useState(false);
  const [editCommentError, setEditCommentError] = useState<string | null>(null);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    }
    if (dropdownOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    } else {
      document.removeEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [dropdownOpen]);

  const isOwnPost = currentUserId === postUserId

  // Add a role/type for accent color (mocked for now, replace with real role if available)
  const userRole = isJobPost ? 'employer' : 'user';
  // Use a soft yellow top border and shadow accent
  const accentColor = 'border-t-4 border-yellow-400 shadow-md shadow-yellow-100';

  // Check if user liked this post on component mount
  useEffect(() => {
    console.log('Component mounted for post:', postId);
    console.log('Initial comments prop value:', comments);
    console.log('Initial commentCount state:', commentCount);
    
    if (currentUserId) {
      console.log('Checking like status for post:', postId, 'user:', currentUserId);
      checkUserLikeStatus();
      refreshLikeCount();
    }
  }, [currentUserId, postId]);

  // Update comment count when comments prop changes
  useEffect(() => {
    console.log('Comments prop changed for post:', postId, 'new value:', comments);
    setCommentCount(comments);
  }, [comments, postId]);

  const checkUserLikeStatus = async () => {
    if (!currentUserId) {
      console.log('No currentUserId, skipping like status check');
      return;
    }
    
    console.log('=== CHECKING LIKE STATUS ===');
    console.log('Post ID:', postId);
    console.log('User ID:', currentUserId);
    
    try {
      const supabase = getSupabaseClient();
      const { data, error } = await supabase
        .from('likes')
        .select('id')
        .eq('post_id', postId)
        .eq('user_id', currentUserId)
        .single();
      
      console.log('Like check result - data:', data, 'error:', error);
      
      if (!error && data) {
        console.log('✅ User has liked this post');
        setUserLiked(true);
      } else {
        console.log('❌ User has not liked this post or error:', error);
        setUserLiked(false);
      }
    } catch (error) {
      console.error('❌ Error checking like status:', error);
      setUserLiked(false);
    }
  };

  const refreshCommentCount = async () => {
    console.log('=== REFRESHING COMMENT COUNT ===');
    console.log('Post ID:', postId);
    
    try {
      const supabase = getSupabaseClient();
      console.log('Making database query for comments...');
      
      // Add a small delay to ensure database has updated
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const { count, error } = await supabase
        .from('comments')
        .select('*', { count: 'exact', head: true })
        .eq('post_id', postId);
      
      console.log('Comment count result - count:', count, 'error:', error);
      console.log('Count type:', typeof count, 'Count value:', count);
      
      if (!error && count !== null) {
        console.log('✅ Setting comment count to:', count);
        setCommentCount(count);
      } else if (error) {
        console.error('❌ Error refreshing comment count:', error);
        // Fallback: try to fetch comments and count them
        console.log('Trying fallback method...');
        const { data: commentsData, error: commentsError } = await supabase
          .from('comments')
          .select('id')
          .eq('post_id', postId);
        
        if (!commentsError && commentsData) {
          console.log('Fallback: Found', commentsData.length, 'comments');
          setCommentCount(commentsData.length);
        } else {
          console.error('Fallback also failed:', commentsError);
        }
      } else {
        console.log('Count is null, setting to 0');
        setCommentCount(0);
      }
    } catch (error) {
      console.error('❌ Error refreshing comment count:', error);
    }
  };

  const refreshLikeCount = async () => {
    console.log('=== REFRESHING LIKE COUNT ===');
    console.log('Post ID:', postId);
    
    try {
      const supabase = getSupabaseClient();
      const { count, error } = await supabase
        .from('likes')
        .select('*', { count: 'exact', head: true })
        .eq('post_id', postId);
      
      console.log('Like count result - count:', count, 'error:', error);
      
      if (!error && count !== null) {
        console.log('✅ Setting like count to:', count);
        setLikeCount(count);
      } else {
        console.error('❌ Error refreshing like count:', error);
      }
    } catch (error) {
      console.error('❌ Error refreshing like count:', error);
    }
  };

  const handleLike = async () => {
    if (likeInProgress || !currentUserId) {
      console.log('Like action blocked - in progress:', likeInProgress, 'no user:', !currentUserId);
      return;
    }
    
    console.log('=== HANDLING LIKE ACTION ===');
    console.log('Current like state:', userLiked);
    console.log('Post ID:', postId);
    console.log('User ID:', currentUserId);
    
    setLikeInProgress(true);
    
    try {
      const supabase = getSupabaseClient();
      
      if (!userLiked) {
        // Add like
        console.log('➕ Adding like...');
        const likeData = { post_id: postId, user_id: currentUserId };
        console.log('Like data to insert:', likeData);
        
        const { data, error } = await supabase
          .from('likes')
          .insert(likeData)
          .select()
          .single();
        
        console.log('Insert result - data:', data, 'error:', error);
        
        if (!error) {
          setLikeCount(prev => prev + 1);
          setUserLiked(true);
          console.log('✅ Like added successfully');
        } else {
          console.error('❌ Error adding like:', error);
        }
      } else {
        // Remove like
        console.log('➖ Removing like...');
        const { data, error } = await supabase
          .from('likes')
          .delete()
          .eq('post_id', postId)
          .eq('user_id', currentUserId)
          .select();
        
        console.log('Delete result - data:', data, 'error:', error);
        
        if (!error) {
          setLikeCount(prev => Math.max(0, prev - 1));
          setUserLiked(false);
          console.log('✅ Like removed successfully');
        } else {
          console.error('❌ Error removing like:', error);
        }
      }
    } catch (error) {
      console.error('❌ Error updating like:', error);
    } finally {
      setLikeInProgress(false);
    }
  };

  const handleComment = async () => {
    if (!commentText.trim()) {
      console.log('Comment text is empty');
      return;
    }
    
    if (!currentUserId) {
      console.log('No currentUserId, cannot add comment');
      return;
    }

    // AI moderation
    try {
      const moderation = await moderateText(commentText.trim());
      if (!moderation.allowed) {
        alert('Your comment was blocked for harmful or toxic content.');
        return;
      }
    } catch (e) {
      console.warn('Moderation failed, proceeding to local checks', e);
    }
    
    console.log('=== COMMENT CREATION DEBUG ===');
    console.log('Current User ID:', currentUserId);
    console.log('Post ID:', postId);
    console.log('Comment Text:', commentText.trim());
    
    try {
      const supabase = getSupabaseClient();
      
      // Get current user's profile for author name
      console.log('Fetching profile for user:', currentUserId);
      const { data: profile, error: profileError } = await supabase
        .from('profiles')
        .select('first_name, last_name')
        .eq('id', currentUserId)
        .single();
      
      if (profileError) {
        console.error('Error fetching profile:', profileError);
        return;
      }
      
      console.log('Profile data:', profile);
      const authorName = profile ? `${profile.first_name} ${profile.last_name}`.trim() : 'Anonymous';
      console.log('Final author name:', authorName);
      
      const commentData = {
        post_id: postId,
        user_id: currentUserId,
        content: commentText.trim(),
        author_name: authorName
      };
      
      console.log('Inserting comment data:', commentData);
      
      const { data: insertedComment, error } = await supabase
        .from('comments')
        .insert(commentData)
        .select()
        .single();
      
      if (!error) {
        console.log('Comment inserted successfully:', insertedComment);
        setCommentText('');
        fetchComments(); // Refresh comments
        refreshCommentCount(); // Update comment count
        console.log('Comment added successfully');
      } else {
        console.error('Error adding comment:', error);
      }
    } catch (error) {
      console.error('Error adding comment:', error);
    }
  };

  const fetchComments = async () => {
    try {
      console.log('=== FETCHING COMMENTS DEBUG ===');
      console.log('Fetching comments for post:', postId);
      
      const supabase = getSupabaseClient();
      const { data, error } = await supabase
        .from('comments')
        .select('*')
        .eq('post_id', postId)
        .order('created_at', { ascending: true });
      
      if (!error && data) {
        console.log('Comments fetched successfully:', data);
        setPostComments(data);
        setCommentCount(data.length); // Update comment count based on fetched comments
      } else {
        console.error('Error fetching comments:', error);
      }
    } catch (error) {
      console.error('Error fetching comments:', error);
    }
  };

  const handleDeleteComment = async (commentId: string, commentUserId: string) => {
    // Check if user can delete this comment (own comment or post owner)
    if (commentUserId !== currentUserId && !isOwnPost) return;
    
    try {
      const supabase = getSupabaseClient();
      const { error } = await supabase
        .from('comments')
        .delete()
        .eq('id', commentId);
      
      if (!error) {
        fetchComments(); // Refresh comments
        refreshCommentCount(); // Update comment count
      }
    } catch (error) {
      console.error('Error deleting comment:', error);
    }
  };

  const fetchPost = async () => {
    try {
      const supabase = getSupabaseClient();
      const { data, error } = await supabase
        .from('posts')
        .select('*')
        .eq('id', postId)
        .single();
      if (!error && data) {
        setEditTitle(data.title);
        setEditContent(data.content);
        // Optionally update other state if needed
      }
    } catch (error) {
      // Optionally handle fetch error
    }
  };

  const handleEditPost = async () => {
    if (!isOwnPost || !editTitle.trim() || !editContent.trim()) return;
    setEditPostLoading(true);
    setEditPostError(null);
    try {
      const supabase = getSupabaseClient();
      const { error } = await supabase
        .from('posts')
        .update({ 
          title: editTitle.trim(),
          content: editContent.trim(),
          updated_at: new Date().toISOString()
        })
        .eq('id', postId);
      if (!error) {
        await fetchPost(); // Re-fetch post from backend
        setIsEditing(false);
        onPostUpdate?.(postId, editTitle.trim(), editContent.trim());
      } else {
        setEditPostError(error.message || 'Error updating post.');
      }
    } catch (error: any) {
      setEditPostError(error.message || 'Error updating post.');
    } finally {
      setEditPostLoading(false);
    }
  };

  const handleEditComment = async (commentId: string) => {
    if (!editingCommentText.trim()) return;
    setEditCommentLoading(true);
    setEditCommentError(null);
    try {
      const supabase = getSupabaseClient();
      const { error } = await supabase
        .from('comments')
        .update({ content: editingCommentText })
        .eq('id', commentId);
      if (!error) {
        await fetchComments(); // Re-fetch comments from backend
        setEditingCommentId(null);
        setEditingCommentText('');
      } else {
        setEditCommentError(error.message || 'Error updating comment.');
      }
    } catch (err: any) {
      setEditCommentError(err.message || 'Error updating comment.');
    } finally {
      setEditCommentLoading(false);
    }
  };

  const handleDeletePost = async () => {
    if (!isOwnPost) return;
    
    try {
      const supabase = getSupabaseClient();
      const { error } = await supabase
        .from('posts')
        .delete()
        .eq('id', postId);
      
      if (!error) {
        onPostDelete?.(postId);
      }
    } catch (error) {
      console.error('Error deleting post:', error);
    }
  };

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: title,
          text: content,
          url: window.location.href
        });
      } catch (error) {
        console.error('Error sharing:', error);
      }
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(`${title}\n\n${content}\n\n${window.location.href}`);
      alert('Link copied to clipboard!');
    }
  };

  const toggleComments = () => {
    if (!showComments) {
      fetchComments();
    }
    setShowComments(!showComments);
  };

  const startEditComment = (comment: Comment) => {
    setEditingCommentId(comment.id);
    setEditingCommentText(comment.content);
  };

  const cancelEditComment = () => {
    setEditingCommentId(null);
    setEditingCommentText('');
  };

  return (
    <Card className="p-5 space-y-2 rounded-xl shadow-md bg-card hover:shadow-lg transition">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link href={`/profile/${postUserId}`} tabIndex={0} aria-label={`View ${author}'s profile`} className="focus:outline-none">
            <ProfileCard
              avatarUrl={avatar}
              name={author}
              subtitle={isJobPost ? "Job" : undefined}
              isEditing={false}
              compact={true}
            />
          </Link>
        </div>
        {/* Dropdown */}
        {isOwnPost && (
          <div className="relative ml-2" ref={dropdownRef}>
            <button
              aria-label="Post options"
              className="hover:bg-accent p-2 rounded-full transition focus:outline-none focus:ring-2 focus:ring-mustard"
              onClick={() => setDropdownOpen((open) => !open)}
              tabIndex={0}
            >
              <MoreHorizontal className="h-5 w-5 text-muted-foreground" aria-hidden="true" />
            </button>
            {dropdownOpen && (
              <div className="absolute right-0 mt-2 w-32 bg-card border border-border rounded-lg shadow-lg z-10">
                <button
                  onClick={() => { setIsEditing(true); setEditTitle(title); setEditContent(content); setDropdownOpen(false); }}
                  className="w-full text-left px-4 py-2 text-sm hover:bg-accent focus:bg-accent rounded-t-lg text-foreground"
                  aria-label="Edit post"
                >
                  <Edit className="h-4 w-4 mr-2 inline-block align-text-bottom" /> Edit
                </button>
                <button
                  onClick={() => { setShowDeleteConfirm(!showDeleteConfirm); setDropdownOpen(false); }}
                  className="w-full text-left px-4 py-2 text-sm text-destructive hover:bg-accent focus:bg-accent rounded-b-lg"
                  aria-label="Delete post"
                >
                  <Trash2 className="h-4 w-4 mr-2 inline-block align-text-bottom" /> Delete
                </button>
              </div>
            )}
          </div>
        )}
      </div>
      {/* Content */}
      <div>
        {isEditing ? (
          <div className="space-y-2">
            <input
              type="text"
              value={editTitle}
              onChange={e => setEditTitle(e.target.value)}
              className="w-full border border-input rounded px-3 py-2 text-base font-semibold mb-1 focus:outline-none focus:ring-2 focus:ring-mustard bg-background text-foreground"
              placeholder="Title"
              aria-label="Edit post title"
              disabled={editPostLoading}
            />
            <textarea
              value={editContent}
              onChange={e => setEditContent(e.target.value)}
              className="w-full border border-input rounded px-3 py-2 text-sm leading-relaxed focus:outline-none focus:ring-2 focus:ring-mustard bg-background text-foreground"
              placeholder="Content"
              aria-label="Edit post content"
              rows={3}
              disabled={editPostLoading}
            />
            {editPostError && <div className="text-destructive text-sm">{editPostError}</div>}
            <div className="flex gap-2 mt-2">
              <button
                onClick={handleEditPost}
                className="bg-mustard text-white px-4 py-1 rounded-full font-semibold hover:bg-yellow-500 transition focus:outline-none focus:ring-2 focus:ring-mustard"
                aria-label="Save post edits"
                disabled={editPostLoading}
              >
                {editPostLoading ? 'Saving...' : 'Save'}
              </button>
              <button
                onClick={() => { setIsEditing(false); setEditTitle(title); setEditContent(content); }}
                className="bg-muted text-muted-foreground px-4 py-1 rounded-full font-semibold hover:bg-accent transition focus:outline-none focus:ring-2 focus:ring-muted-foreground"
                aria-label="Cancel editing post"
                disabled={editPostLoading}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <>
            {title && <div className="font-semibold text-base mb-1 text-foreground">{title}</div>}
            <div className="text-sm leading-relaxed text-foreground">{content}</div>
            {imageUrl && (
              <div className="mb-3">
                <img 
                  src={imageUrl} 
                  alt={`Post image shared by ${author}`}
                  className="max-w-full h-auto rounded-lg"
                  onError={(e) => {
                    e.currentTarget.style.display = 'none';
                  }}
                />
              </div>
            )}
            {videoUrl && (
              <div className="mb-3">
                <video 
                  src={videoUrl} 
                  controls 
                  className="max-w-full h-auto rounded-lg"
                  aria-label={`Video shared by ${author}`}
                  onError={(e) => {
                    e.currentTarget.style.display = 'none';
                  }}
                />
              </div>
            )}
          </>
        )}
      </div>
      {/* Footer */}
      <div className="border-t border-border pt-2 mt-2 flex justify-between items-center text-sm text-muted-foreground">
        <button
          onClick={handleLike}
          disabled={likeInProgress}
          className={`flex items-center gap-1 text-sm px-2 py-1 hover:bg-accent rounded-full transition focus:outline-none focus:ring-2 focus:ring-mustard ${userLiked ? 'text-mustard' : 'text-muted-foreground'}`}
          aria-label={`${userLiked ? 'Unlike' : 'Like'} this post. ${likeCount} likes`}
          aria-pressed={userLiked}
        >
          <Heart className="h-5 w-5" aria-hidden="true" />
          <span className="ml-1">Like</span>
          <span className="ml-1 text-xs">{likeCount}</span>
        </button>
        <button
          onClick={toggleComments}
          className="flex items-center gap-1 text-sm text-muted-foreground px-2 py-1 hover:bg-accent rounded-full transition focus:outline-none focus:ring-2 focus:ring-mustard"
          aria-label={`${showComments ? 'Hide' : 'Show'} comments. ${commentCount} comments`}
          aria-expanded={showComments}
        >
          <MessageCircle className="h-5 w-5" aria-hidden="true" />
          <span className="ml-1">Comment</span>
          <span className="ml-1 text-xs">{commentCount}</span>
        </button>
        <button
          onClick={handleShare}
          className="flex items-center gap-1 text-sm text-muted-foreground px-2 py-1 hover:bg-accent rounded-full transition focus:outline-none focus:ring-2 focus:ring-mustard"
          aria-label="Share this post."
        >
          <Share2 className="h-5 w-5" aria-hidden="true" />
          <span className="ml-1">Share</span>
        </button>
      </div>
      {/* Comments Section */}
      {showComments && (
        <div className="mt-4 space-y-3">
          <div className="flex space-x-2">
            <Textarea
              value={commentText}
              onChange={(e) => setCommentText(e.target.value)}
              placeholder="Write a comment..."
              className="flex-1 bg-background border border-input text-foreground placeholder:text-muted-foreground"
              rows={2}
            />
            <Button onClick={handleComment} disabled={!commentText.trim()} aria-label="Post comment">
              Comment
            </Button>
          </div>
          <div className="space-y-2">
            {postComments.map((comment) => (
              <div key={comment.id} className="bg-muted p-3 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-sm text-foreground">{comment.author_name}</span>
                    <span className="text-xs text-muted-foreground">{new Date(comment.created_at).toLocaleString()}</span>
                  </div>
                  {(currentUserId === comment.user_id || isOwnPost) && (
                    <div className="flex items-center gap-1">
                      {currentUserId === comment.user_id && (
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => startEditComment(comment)}
                          aria-label="Edit this comment"
                          className="hover:bg-accent p-1 rounded-full transition focus:outline-none focus:ring-2 focus:ring-mustard text-muted-foreground"
                        >
                          <Edit className="h-4 w-4" aria-hidden="true" />
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleDeleteComment(comment.id, comment.user_id)}
                        className="hover:bg-accent p-1 rounded-full transition focus:outline-none focus:ring-2 focus:ring-red-400 text-destructive"
                        aria-label="Delete this comment"
                      >
                        <Trash2 className="h-4 w-4" aria-hidden="true" />
                      </Button>
                    </div>
                  )}
                </div>
                {editingCommentId === comment.id ? (
                  <div className="flex items-center gap-2">
                    <Input
                      value={editingCommentText}
                      onChange={(e) => setEditingCommentText(e.target.value)}
                      className="flex-1 bg-background border border-input text-foreground"
                    />
                    <Button
                      onClick={() => handleEditComment(comment.id)}
                      disabled={editCommentLoading}
                      className="bg-mustard text-white px-3 py-1 rounded text-sm"
                    >
                      {editCommentLoading ? 'Saving...' : 'Save'}
                    </Button>
                    <Button
                      onClick={cancelEditComment}
                      variant="outline"
                      className="px-3 py-1 rounded text-sm"
                    >
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <div className="text-sm text-foreground mt-2">{comment.content}</div>
                )}
                {editCommentError && <div className="text-destructive text-xs mt-1">{editCommentError}</div>}
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}
