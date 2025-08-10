import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "../styles/optimized.css";
import "../styles/high-contrast.css";
import { ThemeProvider } from "@/components/theme-provider";
import { AccessibilityProvider } from "@/components/accessibility-provider";

// Optimize font loading
const inter = Inter({ 
  subsets: ["latin"],
  display: 'swap',
  preload: true,
  fallback: ['system-ui', 'arial'],
});

export const metadata: Metadata = {
  title: "AccessAble - Professional Network for People with Disabilities",
  description: "A professional networking platform designed specifically for people with disabilities, featuring accessibility-first design and inclusive features.",
  metadataBase: new URL('https://accessable.com'),
  openGraph: {
    title: "AccessAble - Professional Network for People with Disabilities",
    description: "A professional networking platform designed specifically for people with disabilities, featuring accessibility-first design and inclusive features.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "AccessAble - Professional Network for People with Disabilities",
    description: "A professional networking platform designed specifically for people with disabilities, featuring accessibility-first design and inclusive features.",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Preconnect to external domains */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link rel="preconnect" href="https://*.supabase.co" />
        
        {/* DNS prefetch for performance */}
        <link rel="dns-prefetch" href="https://*.supabase.co" />
        
        {/* Preload critical resources */}
        <link rel="preload" href="/placeholder-logo.svg" as="image" type="image/svg+xml" />
        
        {/* Resource hints */}
        <link rel="modulepreload" href="/_next/static/chunks/vendors.js" />
        <link rel="modulepreload" href="/_next/static/chunks/radix.js" />
      </head>
      <body className={inter.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="light"
          enableSystem
          disableTransitionOnChange
        >
          <AccessibilityProvider>
            {/* Skip to Content Link for Screen Readers */}
            <a
              href="#main-content"
              className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-mustard focus:text-white focus:rounded focus:outline-none focus:ring-2 focus:ring-forest-green"
            >
              Skip to main content
            </a>
            <main id="main-content">
              {children}
            </main>
          </AccessibilityProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
