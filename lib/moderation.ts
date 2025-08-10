export type ModerationResult = {
  allowed: boolean
  reasons: string[]
  scores: Record<string, number>
}

export async function moderateText(text: string): Promise<ModerationResult> {
  if (!text || !text.trim()) {
    return { allowed: true, reasons: [], scores: {} }
  }

  const res = await fetch('/api/moderate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })

  if (!res.ok) {
    throw new Error(`Moderation request failed: ${res.status}`)
  }

  return res.json()
}