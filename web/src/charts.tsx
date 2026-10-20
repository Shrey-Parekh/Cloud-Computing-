import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from 'react'
import { compact, dayLabel, hourLabel } from './format'

type Values = (number | null)[]

export type Layer =
  | { kind: 'line'; values: Values; className: string; draw?: boolean }
  | { kind: 'step'; values: Values; className: string }
  | { kind: 'band'; lower: Values; upper: Values; className: string; stepUpper?: boolean }

export interface EndLabel {
  index: number
  value: number
  text: string
}

interface Props {
  t: string[]
  layers: Layer[]
  label: string
  tooltip: (i: number) => ReactNode
  height?: number
  fromZero?: boolean
  marker?: { index: number; label: string }
  endLabel?: EndLabel
}

const PAD = { top: 16, left: 52, bottom: 30 }

function useWidth() {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, width] as const
}

function niceTicks(min: number, max: number, count = 4) {
  const raw = (max - min) / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw
  const ticks: number[] = []
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(v)
  return ticks
}

export function TimeChart({ t, layers, label, tooltip, height = 320, fromZero = true, marker, endLabel }: Props) {
  const [ref, width] = useWidth()
  const [hover, setHover] = useState<number | null>(null)
  const n = t.length
  const padRight = endLabel && width > 640 ? 104 : 12
  const plotW = Math.max(width - PAD.left - padRight, 10)
  const plotH = height - PAD.top - PAD.bottom

  const all = layers.flatMap((l) => (l.kind === 'band' ? [...l.lower, ...l.upper] : l.values)).filter((v): v is number => v != null)
  const hi = Math.max(...all)
  const lo = fromZero ? 0 : Math.min(...all)
  const pad = (hi - lo) * 0.08
  const ticks = niceTicks(fromZero ? 0 : lo - pad, hi + pad)
  const y0 = ticks[0]
  const y1 = Math.max(ticks[ticks.length - 1], hi + pad)

  const x = (i: number) => PAD.left + (n > 1 ? (i / (n - 1)) * plotW : 0)
  const y = (v: number) => PAD.top + plotH - ((v - y0) / (y1 - y0)) * plotH
  const half = n > 1 ? plotW / (n - 1) / 2 : 0
  const clampX = (v: number) => Math.min(Math.max(v, PAD.left), PAD.left + plotW)

  const runs = (values: Values, step: boolean) => {
    const out: [number, number][][] = []
    let cur: [number, number][] = []
    values.forEach((v, i) => {
      if (v == null) {
        if (cur.length) out.push(cur)
        cur = []
        return
      }
      if (step) cur.push([clampX(x(i) - half), y(v)], [clampX(x(i) + half), y(v)])
      else cur.push([x(i), y(v)])
    })
    if (cur.length) out.push(cur)
    return out
  }
  const toPath = (pts: [number, number][]) => pts.map(([a, b], k) => `${k ? 'L' : 'M'}${a.toFixed(1)},${b.toFixed(1)}`).join('')

  const days = t.map((s, i) => (s.endsWith('T00:00') ? i : -1)).filter((i) => i >= 0)
  const dayStep = Math.ceil(days.length / Math.max(Math.floor(plotW / 70), 1))

  const pick = (clientX: number, el: SVGSVGElement) => {
    const px = clientX - el.getBoundingClientRect().left
    setHover(Math.min(Math.max(Math.round(((px - PAD.left) / plotW) * (n - 1)), 0), n - 1))
  }
  const onKey = (e: KeyboardEvent) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    e.preventDefault()
    const dir = e.key === 'ArrowLeft' ? -1 : 1
    setHover((h) => Math.min(Math.max((h ?? (dir > 0 ? -1 : n)) + dir, 0), n - 1))
  }

  return (
    <div className="chart" ref={ref} style={{ height }}>
      {width > 0 && n > 0 && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${label}. Use the left and right arrow keys to read values hour by hour.`}
          tabIndex={0}
          onPointerMove={(e: PointerEvent<SVGSVGElement>) => pick(e.clientX, e.currentTarget)}
          onPointerLeave={() => setHover(null)}
          onKeyDown={onKey}
          onBlur={() => setHover(null)}
        >
          {ticks.map((v) => (
            <g key={v}>
              <line className="grid" x1={PAD.left} x2={PAD.left + plotW} y1={y(v)} y2={y(v)} />
              <text className="tick" x={PAD.left - 10} y={y(v)} dy="0.32em" textAnchor="end">
                {compact(v)}
              </text>
            </g>
          ))}
          {days.filter((_, k) => k % dayStep === 0).map((i) => (
            <text key={i} className="tick" x={x(i)} y={height - 8} textAnchor="middle">
              {dayLabel(t[i])}
            </text>
          ))}

          {layers.map((l, k) => {
            if (l.kind === 'band') {
              const up = runs(l.upper, !!l.stepUpper)
              const low = runs(l.lower, false)
              return up.map((u, j) => (
                <path key={`${k}-${j}`} className={l.className} d={`${toPath(u)}L${[...(low[j] ?? [])].reverse().map(([a, b]) => `${a.toFixed(1)},${b.toFixed(1)}`).join('L')}Z`} />
              ))
            }
            return runs(l.values, l.kind === 'step').map((r, j) => (
              <path key={`${k}-${j}`} className={`${l.className}${l.kind === 'line' && l.draw ? ' draw' : ''}`} d={toPath(r)} pathLength={l.kind === 'line' && l.draw ? 1 : undefined} />
            ))
          })}

          {marker && (
            <g className="marker">
              <line x1={x(marker.index)} x2={x(marker.index)} y1={PAD.top} y2={PAD.top + plotH} />
              <text x={x(marker.index) + 6} y={PAD.top + 10}>{marker.label}</text>
            </g>
          )}

          {endLabel && padRight > 12 && (
            <text className="end-label" x={x(endLabel.index) + half + 8} y={y(endLabel.value)} dy="0.32em">
              {endLabel.text}
            </text>
          )}

          {hover != null && (
            <line className="crosshair" x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={PAD.top + plotH} />
          )}
        </svg>
      )}
      {hover != null && width > 0 && (
        <div
          className="tooltip"
          style={{
            left: x(hover) > width / 2 ? undefined : x(hover) + 14,
            right: x(hover) > width / 2 ? width - x(hover) + 14 : undefined,
          }}
        >
          <div className="tooltip-time">{`${dayLabel(t[hover])}, ${hourLabel(t[hover])} UTC`}</div>
          {tooltip(hover)}
        </div>
      )}
    </div>
  )
}

export function TipRow({ value, name, swatch }: { value: string; name: string; swatch?: string }) {
  return (
    <div className="tip-row">
      <span className={`key ${swatch ?? ''}`} />
      <strong>{value}</strong>
      <span>{name}</span>
    </div>
  )
}

// One cell per test hour: rows are days, columns are hours of the day (UTC).
export type Verdict = 'under' | 'over' | 'exact'

export function HourGrid({ t, verdicts, detail }: { t: string[]; verdicts: Verdict[]; detail: (i: number) => string }) {
  const [hover, setHover] = useState<number | null>(null)
  const rows: { day: string; cells: (number | null)[] }[] = []
  t.forEach((s, i) => {
    const day = s.slice(0, 10)
    let row = rows[rows.length - 1]
    if (!row || row.day !== day) {
      row = { day, cells: Array(24).fill(null) }
      rows.push(row)
    }
    row.cells[Number(s.slice(11, 13))] = i
  })

  return (
    <div className="hourgrid">
      <div className="hourgrid-head" aria-hidden="true">
        {[0, 6, 12, 18].map((h) => (
          <span key={h} style={{ gridColumn: `${h + 2} / span 6` }}>{`${String(h).padStart(2, '0')}:00`}</span>
        ))}
      </div>
      {rows.map((r) => (
        <div className="hourgrid-row" key={r.day}>
          <span className="hourgrid-day">{dayLabel(`${r.day}T00:00`)}</span>
          {r.cells.map((i, h) =>
            i == null ? (
              <span key={h} className="cell empty" />
            ) : (
              <button
                key={h}
                type="button"
                className={`cell ${verdicts[i]}${hover === i ? ' on' : ''}`}
                aria-label={detail(i)}
                onPointerEnter={() => setHover(i)}
                onPointerLeave={() => setHover(null)}
                onFocus={() => setHover(i)}
                onBlur={() => setHover(null)}
              />
            ),
          )}
        </div>
      ))}
      <p className="hourgrid-readout" aria-live="polite">
        {hover != null ? detail(hover) : 'Point at an hour to compare the VMs the forecast asked for with the VMs that hour needed.'}
      </p>
    </div>
  )
}
