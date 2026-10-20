const compactFmt = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 })
const fullFmt = new Intl.NumberFormat('en-US')

export const compact = (n: number) => compactFmt.format(n)
export const full = (n: number) => fullFmt.format(Math.round(n))

// Timestamps arrive as "YYYY-MM-DDTHH:00" in UTC.
const asDate = (t: string) => new Date(`${t}:00Z`)
export const hourLabel = (t: string) => `${t.slice(11, 13)}:00`
export const dayLabel = (t: string) =>
  asDate(t).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' })
export const stamp = (t: string) => `${dayLabel(t)}, ${hourLabel(t)} UTC`
