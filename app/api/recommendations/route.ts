import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import { cookies } from 'next/headers'
import { computeRecommendationsForUser } from '@/lib/recommendations'

export async function GET(req: NextRequest) {
  try {
    const cookieStore = cookies()
    const incomingAuthHeader = req.headers.get('authorization') || ''
    const bearerFromHeader = incomingAuthHeader?.toLowerCase().startsWith('bearer ')
      ? incomingAuthHeader.slice(7)
      : ''
    const bearerFromCookie = cookieStore.get('sb-access-token')?.value || ''
    const accessToken = bearerFromHeader || bearerFromCookie
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        global: { headers: { Authorization: `Bearer ${accessToken}` } }
      }
    )

    const { data: { user }, error: userErr } = await supabase.auth.getUser()
    if (userErr) return NextResponse.json({ error: userErr.message }, { status: 401 })
    if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })

    const { searchParams } = new URL(req.url)
    const topN = Number(searchParams.get('n') || '5')
    const { results, usedPostsFallback } = await computeRecommendationsForUser(supabase, user.id, topN)

    // hydrate minimal job details for frontend
    const payload = results.map((r: any) => ({
      id: r.job.id,
      title: r.job.title || r.job.job_metadata?.title || 'Job',
      description: r.job.description || r.job.content || '',
      job_metadata: r.job.job_metadata || null,
      score: r.score,
      accessibility: r.job.job_metadata?.accessibility_features || r.job.accessibility_features || null,
    }))

    return NextResponse.json({ recommendations: payload, source: usedPostsFallback ? 'posts' : 'jobs' })
  } catch (e: any) {
    return NextResponse.json({ error: e?.message || 'Recommendation error' }, { status: 500 })
  }
}

export async function POST(req: NextRequest) {
  return GET(req)
}

