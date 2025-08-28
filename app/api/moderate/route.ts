import { NextRequest, NextResponse } from 'next/server'

export async function POST(req: NextRequest) {
  try {
    const { text } = await req.json()
    if (!text || typeof text !== 'string') {
      return NextResponse.json({ error: 'Invalid text' }, { status: 400 })
    }

    const apiKey = process.env.PERSPECTIVE_API_KEY
    if (!apiKey) {
      return NextResponse.json({ error: 'Missing PERSPECTIVE_API_KEY' }, { status: 500 })
    }

    const requestBody = {
      comment: { text },
      languages: ['en'],
      requestedAttributes: {
        TOXICITY: {},
        SEVERE_TOXICITY: {},
        INSULT: {},
        THREAT: {},
        IDENTITY_ATTACK: {},
        PROFANITY: {},
        SEXUALLY_EXPLICIT: {},
        OBSCENE: {},
      },
    }

    const response = await fetch(
      `https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      }
    )

    if (!response.ok) {
      const errText = await response.text()
      return NextResponse.json({ error: 'Perspective API error', details: errText }, { status: 502 })
    }

    const data = await response.json()
    const attrScores = data.attributeScores || {}
    const scores: Record<string, number> = {}
    let maxScore = 0

    Object.keys(attrScores).forEach((key) => {
      const value = attrScores[key]?.summaryScore?.value ?? 0
      scores[key] = value
      if (value > maxScore) maxScore = value
    })

    const BLOCK_THRESHOLD = 0.2
    const allowed = maxScore < BLOCK_THRESHOLD
    const reasons = Object.entries(scores)
      .filter(([_, v]) => v >= BLOCK_THRESHOLD)
      .map(([k]) => k)

    return NextResponse.json({ allowed, scores, reasons })
  } catch (e: any) {
    return NextResponse.json({ error: 'Moderatoin failed', details: e?.message }, { status: 500 })
  }
}