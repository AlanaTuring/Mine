import { Header } from "./components/header"
import { NGOProfileSidebar } from "./components/NGOProfileSidebar"
import { PostCreation } from "./components/post-creation"
import { FeedPost } from "./components/feed-post"
import { JobSidebar } from "./components/job-sidebar"

// This file is not directly used in the Next.js App Router structure.
// The main application entry points are in the `app` directory.
// This file might have been a previous entry point or a placeholder.
// It is safe to ignore or remove if not explicitly used.

export default function PWDLinkedIn() {
  const posts = [
    {
      author: "Sarah Johnson",
      title: "Disability Rights Advocate at Equal Access Foundation",
      time: "2 hours ago",
      content:
        "Just completed my certification in Web Accessibility! 🎉 It's amazing how small changes in design can make such a huge difference for people with disabilities. Looking forward to implementing these practices in my next project. Ready to make the web more inclusive for everyone!",
      likes: 24,
      comments: 8,
      shares: 3,
      isVerified: true,
      tags: ["Accessibility", "InclusiveDesign", "PWD", "WebDev"],
    },
    {
      author: "Michael Torres",
      title: "Software Engineer with Visual Impairment",
      time: "4 hours ago",
      content:
        "Thrilled to announce that I've joined the accessibility team at TechCorp! 🚀 As someone who uses screen readers daily, I'm excited to help build more inclusive technology. Remember: accessibility isn't just compliance, it's about creating equal opportunities for everyone. Let's build a better digital world together! 💪",
      likes: 67,
      comments: 15,
      shares: 12,
      isVerified: true,
      tags: ["NewJob", "Accessibility", "ScreenReader", "TechCorp"],
    },
    {
      author: "Emma Davis",
      title: "Occupational Therapist & Assistive Technology Specialist",
      time: "1 day ago",
      content:
        "Hosting a free webinar next week on 'Workplace Accommodations That Actually Work' 💼 Drawing from 10+ years of experience helping PWDs thrive in their careers. We'll cover practical strategies, real success stories, and actionable tips. Link in comments! Let's break down barriers together and create more inclusive workplaces. 🌟",
      likes: 89,
      comments: 23,
      shares: 34,
      isVerified: true,
      tags: ["WorkplaceInclusion", "AssistiveTechnology", "Webinar", "PWD"],
    },
    {
      author: "James Wilson",
      title: "Entrepreneur & Wheelchair User",
      time: "2 days ago",
      content:
        "My startup just secured Series A funding! 🎯 Building accessible e-commerce solutions has been my passion project for 3 years. To other PWD entrepreneurs: your unique perspective is your superpower. Don't let anyone tell you otherwise. The world needs what you're building! Keep pushing boundaries and creating solutions that matter. 🚀✨",
      likes: 156,
      comments: 42,
      shares: 28,
      isVerified: true,
      tags: ["Startup", "SeriesA", "Entrepreneur", "AccessibleTech", "PWD"],
    },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-red-50/30">
      <Header />

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Sidebar */}
          <div className="lg:col-span-3">
            <NGOProfileSidebar />
          </div>

          {/* Main Feed */}
          <div className="lg:col-span-6 space-y-6">
            <PostCreation />

            <div className="space-y-6">
              {posts.map((post, index) => (
                <FeedPost
                  key={index}
                  author={post.author}
                  title={post.title}
                  time={post.time}
                  content={post.content}
                  likes={post.likes}
                  comments={post.comments}
                  shares={post.shares}
                  isVerified={post.isVerified}
                  tags={post.tags}
                />
              ))}
            </div>
          </div>

          {/* Right Sidebar */}
          <div className="lg:col-span-3">
            <JobSidebar />
          </div>
        </div>
      </div>
    </div>
  )
}
