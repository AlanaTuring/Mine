import { createClient } from '@supabase/supabase-js'
import { embedTexts, cosineSimilarity } from './hfEmbeddings'

type SupabaseClient = ReturnType<typeof createClient>

type UserProfile = {
  id: string
  skills: string | string[] | null
  accessibility_needs?: string[] | null
}

function skillsToText(skills: string | string[] | null | undefined): string {
  if (Array.isArray(skills)) return skills.filter(Boolean).join(' ')
  if (typeof skills === 'string') return skills
  return ''
}

type JobRow = {
  id: string
  title?: string | null
  description?: string | null
  content?: string | null
  job_metadata?: any
}

function accessibilityNeedsToText(needs: string | string[] | null | undefined): string {
  if (Array.isArray(needs)) return needs.filter(Boolean).join(' ')
  if (typeof needs === 'string') return needs
  return ''
}

function jobAccessibilityToText(job: JobRow): string {
  const meta = (job && job.job_metadata) || {}
  const fromMetadata = meta?.accessibility_features
  const features: string[] = []
  if (fromMetadata && typeof fromMetadata === 'object') {
    for (const [key, value] of Object.entries(fromMetadata)) {
      if (value) features.push(key)
    }
  }
  // Attempt to read from possible jobs table column if present at runtime
  const anyJob = job as any
  const jobTableFeatures = anyJob?.accessibility_features
  if (jobTableFeatures && typeof jobTableFeatures === 'object') {
    for (const [key, value] of Object.entries(jobTableFeatures)) {
      if (value) features.push(key)
    }
  }
  return features.join(' ')
}

function computeAccessibilityBoost(needs: string[] | null | undefined, job: JobRow): number {
  if (!needs || needs.length === 0) return 0
  const enabled = new Set<string>(jobAccessibilityToText(job).toLowerCase().split(/\s+/).filter(Boolean))
  if (enabled.size === 0) return 0
  let matches = 0
  for (const needRaw of needs) {
    const need = (needRaw || '').toLowerCase()
    if (!need) continue
    if (need.includes('visual')) {
      if (enabled.has('screenreadersupport') || enabled.has('screen_reader_support') || enabled.has('assistivetechnology')) matches++
    } else if (need.includes('hearing')) {
      if (enabled.has('signlanguagesupport')) matches++
    } else if (need.includes('mobility')) {
      if (enabled.has('accessibleoffice') || enabled.has('remotework')) matches++
    } else if (need.includes('cognitive')) {
      if (enabled.has('flexiblehours') || enabled.has('remotework')) matches++
    } else {
      // Generic boost if any accessibility features exist
      if (enabled.size > 0) matches += 0.5
    }
  }
  // Cap the boost to keep cosine range sensible
  const boost = Math.min(0.2, matches * 0.05)
  return boost
}

export async function fetchProfilesAndJobs(supabase: SupabaseClient): Promise<{ profiles: UserProfile[]; jobs: JobRow[]; usedPostsFallback: boolean }>{
  const { data: profiles, error: profilesError } = await supabase
    .from('profiles')
    .select('id, skills, accessibility_needs')

  if (profilesError) throw profilesError

  // Try jobs table first
  let usedPostsFallback = false
  let jobs: JobRow[] = []

  const { data: jobsData, error: jobsError } = await supabase
    .from('jobs')
    .select('id, title, description')
    .limit(1000)

  if (!jobsError && Array.isArray(jobsData)) {
    jobs = jobsData
  } else {
    // Fallback to posts where is_job_post = true
    usedPostsFallback = true
    const { data: postsData, error: postsError } = await supabase
      .from('posts')
      .select('id, title, content, job_metadata')
      .eq('is_job_post', true)
      .limit(1000)
    if (postsError) throw postsError
    jobs = (postsData || []).map((p: any) => ({ id: p.id, title: p.title, content: p.content, job_metadata: p.job_metadata }))
  }

  return { profiles: (profiles || []) as UserProfile[], jobs, usedPostsFallback }
}

export async function computeRecommendationsForAllUsers(supabase: SupabaseClient, topN: number = 5) {
  const { profiles, jobs, usedPostsFallback } = await fetchProfilesAndJobs(supabase)

  const profileIds = profiles.map((p) => p.id)
  const profileTexts = profiles.map((p) => `${skillsToText(p.skills)} ${accessibilityNeedsToText(p.accessibility_needs)}`.trim())
  const jobIds = jobs.map((j) => j.id)
  const jobTexts = jobs.map((j) => `${j.description || j.content || j.title || ''} ${jobAccessibilityToText(j)}`.trim())

  // Embed all texts in two batches
  const [profileEmbeddings, jobEmbeddings] = await Promise.all([
    embedTexts(profileTexts),
    embedTexts(jobTexts),
  ])

  const results: Record<string, { jobId: string; score: number }[]> = {}

  for (let pIdx = 0; pIdx < profileEmbeddings.length; pIdx++) {
    const pVec = profileEmbeddings[pIdx]
    const scores: { jobId: string; score: number }[] = []
    for (let jIdx = 0; jIdx < jobEmbeddings.length; jIdx++) {
      const jVec = jobEmbeddings[jIdx]
      const base = cosineSimilarity(pVec, jVec)
      const boost = computeAccessibilityBoost(profiles[pIdx].accessibility_needs || null, jobs[jIdx])
      const score = base + boost
      scores.push({ jobId: jobIds[jIdx], score })
    }
    scores.sort((a, b) => b.score - a.score)
    results[profileIds[pIdx]] = scores.slice(0, topN)
  }

  return { results, usedPostsFallback }
}

export async function computeRecommendationsForUser(supabase: SupabaseClient, userId: string, topN: number = 5) {
  const { data: profile, error: profileError } = await supabase
    .from('profiles')
    .select('id, skills, accessibility_needs')
    .eq('id', userId)
    .maybeSingle()

  if (profileError) throw profileError
  if (!profile) return { results: [], usedPostsFallback: false }

  // Try jobs table first
  let usedPostsFallback = false
  let jobs: JobRow[] = []
  const { data: jobsData, error: jobsError } = await supabase
    .from('jobs')
    .select('id, title, description')
    .limit(1000)

  if (!jobsError && Array.isArray(jobsData)) {
    jobs = jobsData
  } else {
    usedPostsFallback = true
    const { data: postsData, error: postsError } = await supabase
      .from('posts')
      .select('id, title, content, job_metadata')
      .eq('is_job_post', true)
      .limit(1000)
    if (postsError) throw postsError
    jobs = (postsData || []).map((p: any) => ({ id: p.id, title: p.title, content: p.content, job_metadata: p.job_metadata }))
  }

  if (jobs.length === 0) return { results: [], usedPostsFallback }

  try {
    const [profileVecArr, jobEmbeddings] = await Promise.all([
      embedTexts([`${skillsToText(profile.skills)} ${accessibilityNeedsToText(profile.accessibility_needs)}`.trim()]),
      embedTexts(jobs.map((j) => `${j.description || j.content || j.title || ''} ${jobAccessibilityToText(j)}`.trim())),
    ])
    const profileVec = profileVecArr[0]

    const scored = jobs.map((j, idx) => {
      const base = cosineSimilarity(profileVec, jobEmbeddings[idx])
      const boost = computeAccessibilityBoost(profile.accessibility_needs || null, j)
      return { job: j, score: base + boost }
    })

    scored.sort((a, b) => b.score - a.score)
    return { results: scored.slice(0, topN), usedPostsFallback }
  } catch (err) {
    // Fallback: simple keyword matching on skills vs job text when embeddings fail
    const skills = (skillsToText(profile.skills) + ' ' + accessibilityNeedsToText(profile.accessibility_needs)).toLowerCase()
      .split(/[\s,;]+/)
      .filter(Boolean)
    const scored = jobs.map((j) => {
      const text = (
        (j.description || '') + ' ' +
        (j.content || '') + ' ' +
        (j.title || '') + ' ' +
        jobAccessibilityToText(j)
      ).toLowerCase()
      const base = skills.reduce((acc, word) => acc + (text.includes(word) ? 1 : 0), 0)
      const boost = computeAccessibilityBoost(profile.accessibility_needs || null, j) * 20 // scale to roughly match keyword counts
      const score = base + boost
      return { job: j, score }
    })
    scored.sort((a, b) => b.score - a.score)
    return { results: scored.slice(0, topN), usedPostsFallback }
  }
}