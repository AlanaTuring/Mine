import assert from 'node:assert'

type EmbeddingsResponse = {
  embeddings: number[][]
}

const DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

export async function embedTexts(texts: string[], model: string = DEFAULT_MODEL): Promise<number[][]> {
  assert(Array.isArray(texts), 'texts must be an array')
  const trimmed = texts.map((t) => (t || '').toString().trim())
  const inputs = trimmed.map((t) => (t.length === 0 ? ' ' : t))

  const apiKey = process.env.HF_API_KEY || process.env.HUGGINGFACE_API_KEY
  if (!apiKey) {
    throw new Error('Missing HF_API_KEY environment variable')
  }

  const res = await fetch(`https://api-inference.huggingface.co/pipeline/feature-extraction/${model}`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'x-wait-for-model': 'true',
    },
    body: JSON.stringify({ inputs })
  })

  if (!res.ok) {
    const txt = await res.text()
    throw new Error(`HuggingFace embeddings error: ${res.status} ${txt}`)
  }

  const data = (await res.json()) as EmbeddingsResponse | number[][]
  // HF can return directly number[][]
  const vectors: number[][] = Array.isArray(data) ? (data as number[][]) : (data as EmbeddingsResponse).embeddings

  if (!Array.isArray(vectors) || !Array.isArray(vectors[0])) {
    throw new Error('Unexpected embeddings response shape')
  }

  return vectors
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}

