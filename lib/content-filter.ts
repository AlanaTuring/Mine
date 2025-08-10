// Content Filter for AccessAble Platform
// This is a basic implementation - in production, use a more comprehensive solution

const badWords = [
  // Common profanity
  'fuck', 'shit', 'bitch', 'ass', 'damn', 'hell',
  // Slurs and offensive terms
  'retard', 'retarded', 'spastic', 'cripple', 'handicap',
  // Hate speech
  'nazi', 'fascist', 'racist', 'sexist', 'homophobic',
  // Add more as needed
]

const replacementWords = {
  'fuck': 'f***',
  'shit': 's***',
  'bitch': 'b****',
  'ass': 'a**',
  'damn': 'd***',
  'hell': 'h***',
  'retard': 'r*****',
  'retarded': 'r******',
  'spastic': 's******',
  'cripple': 'c******',
  'handicap': 'h*******',
  'nazi': 'n***',
  'fascist': 'f******',
  'racist': 'r*****',
  'sexist': 's*****',
  'homophobic': 'h********',
}

export function filterContent(text: string): string {
  if (!text) return text
  
  let filteredText = text.toLowerCase()
  
  // Replace bad words with asterisks
  badWords.forEach(badWord => {
    const regex = new RegExp(`\\b${badWord}\\b`, 'gi')
    const replacement = replacementWords[badWord as keyof typeof replacementWords] || '*'.repeat(badWord.length)
    filteredText = filteredText.replace(regex, replacement)
  })
  
  return filteredText
}

export function hasBadWords(text: string): boolean {
  if (!text) return false
  
  const lowerText = text.toLowerCase()
  return badWords.some(badWord => {
    const regex = new RegExp(`\\b${badWord}\\b`, 'gi')
    return regex.test(lowerText)
  })
}

export function getFilteredWordCount(text: string): number {
  if (!text) return 0
  
  const lowerText = text.toLowerCase()
  let count = 0
  
  badWords.forEach(badWord => {
    const regex = new RegExp(`\\b${badWord}\\b`, 'gi')
    const matches = lowerText.match(regex)
    if (matches) {
      count += matches.length
    }
  })
  
  return count
} 