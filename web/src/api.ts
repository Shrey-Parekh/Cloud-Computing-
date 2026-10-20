// Talks to the FastAPI backend (api/main.py). In dev, Vite proxies /api to
// localhost:8000. On Vercel, set VITE_API_URL to the Render backend URL.
const BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')

export type Mode = 'one_step' | 'multi_step'

export interface Settings {
  p: number
  d: number
  q: number
  train_ratio: number
  mode: Mode
  horizon: number
}

export interface RunResult {
  data: { source: string; message: string; rows: number; imputed: number; mean: number; peak: number }
  order: [number, number, number]
  seasonal: [number, number, number, number]
  mode: Mode
  train_ratio: number
  aic: number
  history: { t: string[]; y: number[] }
  n_train: number
  predicted: number[]
  metrics: { MAE: number; RMSE: number; MAPE: number; Accuracy: number }
  forecast: { t: string[]; predicted: number[]; lower: number[]; upper: number[] }
}

export interface Suggestion {
  order: [number, number, number]
  candidates: { p: number; d: number; q: number; AIC: number }[]
}

async function post<T>(path: string, body: unknown): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    throw new Error('The forecasting server did not respond. Check that the API is running, then try again.')
  }
  const json = await res.json().catch(() => null)
  if (!res.ok) {
    const detail = json?.detail
    throw new Error(typeof detail === 'string' ? detail : `Server error (HTTP ${res.status}).`)
  }
  return json as T
}

export const runModel = (s: Settings, refresh = false) => post<RunResult>('/api/run', { ...s, refresh })
export const suggestOrder = (train_ratio: number) => post<Suggestion>('/api/suggest', { train_ratio })

// Mirrors src/vm_calculator.required_vms so capacity changes are instant.
export const vmsFor = (requests: number, capacity: number) => Math.max(1, Math.ceil(Math.max(requests, 0) / capacity))
