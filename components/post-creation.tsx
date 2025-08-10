"use client"

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ImageIcon, Video, Mic, Send, Loader2 } from "lucide-react";
import { getSupabaseClient } from "@/lib/supabaseClient";
import { filterContent, hasBadWords } from "@/lib/content-filter";
import { useAccessibility } from "@/components/accessibility-provider";

export function PostCreation() {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const { settings } = useAccessibility();
  const highContrast = settings.highContrast;

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      setImageUrl(URL.createObjectURL(file));
      // Clear video if image is selected
      setVideoFile(null);
      setVideoUrl(null);
    }
  };

  const handleVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      // Clear image if video is selected
      setImageFile(null);
      setImageUrl(null);
    }
  };

  const handlePost = async () => {
    setLoading(true);
    const supabase = getSupabaseClient();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      alert("You must be logged in to post.");
      setLoading(false);
      return;
    }

    // Check for bad words
    if (hasBadWords(title) || hasBadWords(content)) {
      alert("Your post contains inappropriate language. Please review and edit your content.");
      setLoading(false);
      return;
    }

    // Filter content
    const filteredTitle = filterContent(title);
    const filteredContent = filterContent(content);

    let uploadedImageUrl = null;
    let uploadedVideoUrl = null;

    // Upload image if selected
    if (imageFile) {
      const imageFileName = `${user.id}/${Date.now()}_${imageFile.name}`;
      const { data: imageData, error: imageError } = await supabase.storage
        .from("post-media")
        .upload(imageFileName, imageFile);

      if (imageError) {
        console.error("Error uploading image:", imageError);
        alert("Error uploading image");
        setLoading(false);
        return;
      }

      const { data: imageUrlData } = supabase.storage
        .from("post-media")
        .getPublicUrl(imageFileName);
      uploadedImageUrl = imageUrlData.publicUrl;
    }

    // Upload video if selected
    if (videoFile) {
      const videoFileName = `${user.id}/${Date.now()}_${videoFile.name}`;
      const { data: videoData, error: videoError } = await supabase.storage
        .from("post-media")
        .upload(videoFileName, videoFile);

      if (videoError) {
        console.error("Error uploading video:", videoError);
        alert("Error uploading video");
        setLoading(false);
        return;
      }

      const { data: videoUrlData } = supabase.storage
        .from("post-media")
        .getPublicUrl(videoFileName);
      uploadedVideoUrl = videoUrlData.publicUrl;
    }

    const { error } = await supabase.from("posts").insert([{
      user_id: user.id,
      title: filteredTitle,
      content: filteredContent,
      image_url: uploadedImageUrl,
      video_url: uploadedVideoUrl,
      is_job_post: false, // Always false for general posts
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      likes: 0,
      comments: 0,
      shares: 0,
      tags: [],
    }]);

    if (error) {
      console.error("Error creating post:", error);
      alert("Error creating post");
    } else {
      setTitle("");
      setContent("");
      setImageFile(null);
      setVideoFile(null);
      setImageUrl(null);
      setVideoUrl(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      if (videoInputRef.current) videoInputRef.current.value = "";
      alert("Post created successfully!");
    }
    setLoading(false);
  };

  return (
    <div className="relative">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/70 z-50">
          <Loader2 className="animate-spin h-8 w-8 text-mustard" />
        </div>
      )}
      <div className="bg-card rounded-xl shadow-sm border border-border p-5 mb-6">
        <div className="flex items-start gap-4">
          {/* Profile Picture */}
          <div className="flex-shrink-0">
            <div className="w-10 h-10 bg-gradient-to-br from-mustard to-forest-green rounded-full flex items-center justify-center text-white font-semibold text-sm">
              ME
            </div>
          </div>
          {/* Post Input and Actions */}
          <div className="flex-1 min-w-0">
            <Textarea
              placeholder="Share an accessible post, article, or opportunity..."
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className="w-full bg-background border border-input rounded-xl px-4 py-3 text-sm min-h-[60px] resize-none focus:min-h-[100px] focus:py-4 focus:px-5 focus:shadow-md focus:outline-none focus:ring-2 focus:ring-yellow-400 transition-all text-foreground placeholder:text-muted-foreground"
              rows={3}
              aria-label="Post content"
            />
            {/* Media Preview */}
            {imageUrl && (
              <div className="mb-3">
                <img
                  src={imageUrl}
                  alt="Preview"
                  className="max-w-full h-auto rounded-lg max-h-48 object-cover"
                />
              </div>
            )}
            {videoUrl && (
              <div className="mb-3">
                <video
                  src={videoUrl}
                  controls
                  className="max-w-full h-auto rounded-lg max-h-48"
                />
              </div>
            )}
            {/* Actions Row */}
            <div className="flex items-center justify-between mt-2">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-2 text-sm text-muted-foreground hover:bg-accent px-3 py-2 rounded-md transition focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  aria-label="Add photo"
                >
                  <ImageIcon className="h-5 w-5" />
                  <span>Photo</span>
                  <input
                    type="file"
                    accept="image/*"
                    ref={fileInputRef}
                    style={{ display: "none" }}
                    onChange={handleImageChange}
                  />
                </button>
                <button
                  type="button"
                  onClick={() => videoInputRef.current?.click()}
                  className="flex items-center gap-2 text-sm text-muted-foreground hover:bg-accent px-3 py-2 rounded-md transition focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  aria-label="Add video"
                >
                  <Video className="h-5 w-5" />
                  <span>Video</span>
                  <input
                    type="file"
                    accept="video/*"
                    ref={videoInputRef}
                    style={{ display: "none" }}
                    onChange={handleVideoChange}
                  />
                </button>
                <button
                  type="button"
                  disabled
                  className="flex items-center gap-2 text-sm text-muted-foreground hover:bg-accent px-3 py-2 rounded-md transition focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  aria-label="Add audio (coming soon)"
                >
                  <Mic className="h-5 w-5" />
                  <span>Audio</span>
                </button>
              </div>
              <button
                onClick={handlePost}
                disabled={loading || !content.trim()}
                className="bg-yellow-400 hover:bg-yellow-500 text-white p-2 rounded-full transition focus:outline-none focus:ring-2 focus:ring-yellow-500"
                aria-label="Create post"
                type="button"
              >
                {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
