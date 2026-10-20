import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ArrowDownIcon,
  ArrowUpIcon,
  ArrowsClockwiseIcon,
  DownloadSimpleIcon,
  EqualsIcon,
  MinusIcon,
  MoonIcon,
  PlusIcon,
  SunIcon,
} from '@phosphor-icons/react'
import { runModel, suggestOrder, vmsFor, type Mode, type RunResult, type Settings, type Suggestion } from './api'
import { HourGrid, TimeChart, TipRow, type Verdict } from './charts'
import { compact, full, hourLabel, stamp } from './format'

const DEFAULTS: Settings = { p: 2, d: 1, q: 2, train_ratio: 0.8, mode: 'one_step', horizon: 24 }
const HISTORY_SHOWN = 72

type Theme = 'light' | 'dark'
const store = {
  get: (k: string) => { try { return localStorage.getItem(k) } catch { return null } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v) } catch { /* private mode */ } },
}

function useTheme() {
  const [theme, setTheme] = useState<Theme | null>(() => store.get('theme') as Theme | null)
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme
  }, [theme])
  const effective = theme ?? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
  const toggle = () => {
    const next = effective === 'dark' ? 'light' : 'dark'
    store.set('theme', next)
    setTheme(next)
  }
  return { effective, toggle }
}

export default function App() {
  const { effective, toggle } = useTheme()
  const [draft, setDraft] = useState<Settings>(DEFAULTS)
  const [applied, setApplied] = useState<Settings>(DEFAULTS)
  const [result, setResult] = useState<RunResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [capacity, setCapacity] = useState(500_000)
  const [running, setRunning] = useState(0)
  const [suggestion, setSuggestion] = useState<Suggestion | null>(null)
  const [suggesting, setSuggesting] = useState(false)
  const [suggestError, setSuggestError] = useState('')

  const load = useCallback((s: Settings, refresh = false) => {
    setLoading(true)
    setError('')
    runModel(s, refresh)
      .then(setResult)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => load(applied), [applied, load])

  const dirty = (Object.keys(draft) as (keyof Settings)[]).some((k) => draft[k] !== applied[k])
  const fit = () => (dirty ? setApplied(draft) : load(applied))

  const suggest = () => {
    setSuggesting(true)
    setSuggestError('')
    suggestOrder(draft.train_ratio)
      .then((s) => {
        setSuggestion(s)
        const [p, d, q] = s.order
        setDraft((x) => ({ ...x, p, d, q }))
      })
      .catch((e: Error) => setSuggestError(e.message))
      .finally(() => setSuggesting(false))
  }

  const source = result?.data.source
  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">Headroom</span>
          <span className="brand-note">ARIMA workload forecasting for VM provisioning</span>
        </div>
        <div className="topbar-actions">
          {result && (
            <span className="data-status">
              {result.data.rows} hours of data, {source === 'download' ? 'fresh from Wikimedia' : source === 'cache' ? 'from local cache' : 'from an older cache'}
            </span>
          )}
          <button type="button" className="ghost" onClick={() => load(applied, true)} disabled={loading}>
            <ArrowsClockwiseIcon size={16} weight="bold" aria-hidden="true" /> Refresh data
          </button>
          <button type="button" className="ghost icon" onClick={toggle} aria-label={`Switch to ${effective === 'dark' ? 'light' : 'dark'} theme`}>
            {effective === 'dark' ? <SunIcon size={18} aria-hidden="true" /> : <MoonIcon size={18} aria-hidden="true" />}
          </button>
        </div>
      </header>

      <div className="layout">
        <Console
          draft={draft}
          setDraft={setDraft}
          dirty={dirty}
          loading={loading}
          onFit={fit}
          capacity={capacity}
          setCapacity={setCapacity}
          running={running}
          setRunning={setRunning}
          suggestion={suggestion}
          suggesting={suggesting}
          suggestError={suggestError}
          onSuggest={suggest}
          pickOrder={(p, d, q) => setDraft((x) => ({ ...x, p, d, q }))}
        />

        <main className={loading && result ? 'busy' : undefined} aria-busy={loading}>
          <h1 className="sr-only">Workload forecast and VM provisioning plan</h1>
          {error && (
            <div className="alert" role="alert">
              <p>{error}</p>
              <button type="button" className="ghost" onClick={() => load(applied)}>Try again</button>
            </div>
          )}
          {result ? <Report result={result} capacity={capacity} running={running} /> : !error && <Skeleton />}
        </main>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------------ */

interface ConsoleProps {
  draft: Settings
  setDraft: (f: (s: Settings) => Settings) => void
  dirty: boolean
  loading: boolean
  onFit: () => void
  capacity: number
  setCapacity: (n: number) => void
  running: number
  setRunning: (n: number) => void
  suggestion: Suggestion | null
  suggesting: boolean
  suggestError: string
  onSuggest: () => void
  pickOrder: (p: number, d: number, q: number) => void
}

function Console(c: ConsoleProps) {
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => c.setDraft((s) => ({ ...s, [k]: v }))
  return (
    <aside className="console" aria-label="Model settings">
      <fieldset>
        <legend>Model order</legend>
        <Stepper label="AR terms" sym="p" value={c.draft.p} min={0} max={5} onChange={(v) => set('p', v)} />
        <Stepper label="Differencing" sym="d" value={c.draft.d} min={0} max={2} onChange={(v) => set('d', v)} />
        <Stepper label="MA terms" sym="q" value={c.draft.q} min={0} max={5} onChange={(v) => set('q', v)} />
        <p className="help">Daily seasonality (1, 0, 1, 24) is always on.</p>
        <button type="button" className="ghost wide" onClick={c.onSuggest} disabled={c.suggesting}>
          {c.suggesting ? 'Searching 12 models...' : 'Suggest an order'}
        </button>
        {c.suggesting && <p className="help">ADF test for d, then the lowest AIC. This can take a minute.</p>}
        {c.suggestError && <p className="field-error">{c.suggestError}</p>}
        {c.suggestion && !c.suggesting && (
          <table className="aic">
            <caption>Lowest AIC on training data</caption>
            <tbody>
              {c.suggestion.candidates.slice(0, 4).map((r) => (
                <tr key={`${r.p}${r.d}${r.q}`}>
                  <td>
                    <button type="button" className="link" onClick={() => c.pickOrder(r.p, r.d, r.q)}>
                      ({r.p}, {r.d}, {r.q})
                    </button>
                  </td>
                  <td>{full(r.AIC)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </fieldset>

      <fieldset>
        <legend>Evaluation</legend>
        <Segmented
          label="Training share"
          value={c.draft.train_ratio}
          options={[0.6, 0.7, 0.8, 0.9].map((v) => ({ value: v, text: `${Math.round(v * 100)}%` }))}
          onChange={(v) => set('train_ratio', v)}
        />
        <Segmented<Mode>
          label="Test-period prediction"
          value={c.draft.mode}
          options={[{ value: 'one_step', text: 'Hour by hour' }, { value: 'multi_step', text: 'All at once' }]}
          onChange={(v) => set('mode', v)}
        />
        <p className="help">
          {c.draft.mode === 'one_step'
            ? 'Each hour is predicted once the hour before it is known, the way a provisioner runs.'
            : 'The whole test period is predicted without seeing any of it. Harder, so errors grow.'}
        </p>
      </fieldset>

      <fieldset>
        <legend>Forecast</legend>
        <Segmented
          label="Hours ahead"
          value={c.draft.horizon}
          options={[12, 24, 48, 72].map((v) => ({ value: v, text: `${v} h` }))}
          onChange={(v) => set('horizon', v)}
        />
      </fieldset>

      <button type="button" className="primary wide" onClick={c.onFit} disabled={c.loading}>
        {c.loading ? 'Fitting model...' : c.dirty ? 'Fit with new settings' : 'Fit model'}
      </button>

      <fieldset>
        <legend>Capacity</legend>
        <div className="field">
          <label htmlFor="cap">Requests per VM per hour</label>
          <input id="cap" type="number" min={1000} step={50000} value={c.capacity}
            onChange={(e) => c.setCapacity(Math.max(1000, Number(e.target.value) || 1000))} />
          <p className="help">A project assumption, not an AWS instance spec. Applies instantly.</p>
        </div>
        <div className="field">
          <label htmlFor="running">VMs running now</label>
          <input id="running" type="number" min={0} step={1} value={c.running}
            onChange={(e) => c.setRunning(Math.max(0, Math.floor(Number(e.target.value) || 0)))} />
          <p className="help">0 uses the VMs the last observed hour needed.</p>
        </div>
      </fieldset>
    </aside>
  )
}

function Stepper({ label, sym, value, min, max, onChange }: { label: string; sym: string; value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <div className="stepper">
      <span className="stepper-label">{label} <i>{sym}</i></span>
      <div className="stepper-ctl">
        <button type="button" aria-label={`Decrease ${sym}`} disabled={value <= min} onClick={() => onChange(value - 1)}>
          <MinusIcon size={14} weight="bold" aria-hidden="true" />
        </button>
        <output aria-label={`${sym} value`}>{value}</output>
        <button type="button" aria-label={`Increase ${sym}`} disabled={value >= max} onClick={() => onChange(value + 1)}>
          <PlusIcon size={14} weight="bold" aria-hidden="true" />
        </button>
      </div>
    </div>
  )
}

function Segmented<T extends string | number>({ label, value, options, onChange }: { label: string; value: T; options: { value: T; text: string }[]; onChange: (v: T) => void }) {
  const id = `seg-${label.replace(/\W+/g, '-').toLowerCase()}`
  return (
    <div className="field">
      <span className="field-label" id={id}>{label}</span>
      <div className="segmented" role="radiogroup" aria-labelledby={id}>
        {options.map((o) => (
          <label key={String(o.value)} className={o.value === value ? 'on' : undefined}>
            <input type="radio" name={id} checked={o.value === value} onChange={() => onChange(o.value)} />
            {o.text}
          </label>
        ))}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------------ */

function Report({ result: r, capacity, running }: { result: RunResult; capacity: number; running: number }) {
  const { history, forecast } = r
  const lastActual = history.y[history.y.length - 1]
  const current = running > 0 ? running : vmsFor(lastActual, capacity)
  const plan = forecast.predicted.map((v) => vmsFor(v, capacity))
  const next = plan[0]
  const peak = Math.max(...plan)

  // Headroom chart: last 72 observed hours, then the forecast horizon.
  const h0 = Math.max(history.t.length - HISTORY_SHOWN, 0)
  const pastT = history.t.slice(h0)
  const pastY = history.y.slice(h0)
  const nPast = pastT.length
  const pad = <T,>(k: number) => Array<T | null>(k).fill(null)
  const span = [...pastT, ...forecast.t]
  const demand = [...pastY, ...pad<number>(forecast.t.length)]
  const fcLine = [...pad<number>(nPast - 1), pastY[nPast - 1], ...forecast.predicted]
  const fcOnly = [...pad<number>(nPast), ...forecast.predicted]
  const capLine = [...pad<number>(nPast), ...plan.map((v) => v * capacity)]
  const lower = [...pad<number>(nPast), ...forecast.lower]
  const upper = [...pad<number>(nPast), ...forecast.upper]

  // Backtest on the held-out hours.
  const testT = history.t.slice(r.n_train)
  const testY = history.y.slice(r.n_train)
  const verdicts: Verdict[] = testY.map((a, i) => {
    const need = vmsFor(a, capacity)
    const got = vmsFor(r.predicted[i], capacity)
    return got < need ? 'under' : got > need ? 'over' : 'exact'
  })
  const count = (v: Verdict) => verdicts.filter((x) => x === v).length

  const action = next > current ? 'up' : next < current ? 'down' : 'hold'
  const diff = Math.abs(next - current)
  const vmWord = diff === 1 ? 'VM' : 'VMs'
  const nextAt = hourLabel(forecast.t[0])
  const advice =
    action === 'up' ? `Add ${diff} ${vmWord} before ${nextAt} UTC, taking ${current} to ${next}.`
      : action === 'down' ? `Release ${diff} ${vmWord} at ${nextAt} UTC, taking ${current} to ${next}.`
        : `Keep the ${current} VMs already running.`

  return (
    <>
      <section className="hero">
        <div className="hero-main">
          <p className="kicker">Needed for {stamp(forecast.t[0])}</p>
          <p className="figure">
            {next}<span> VMs</span>
          </p>
          <p className="lede">
            <ActionIcon action={action} />
            <span>{advice} Forecast demand is {compact(forecast.predicted[0])} requests that hour.</span>
          </p>
        </div>
        <dl className="hero-stats">
          <div><dt>Peak in the next {forecast.t.length} h</dt><dd>{peak} VMs</dd></div>
          <div><dt>Accuracy on unseen hours</dt><dd>{r.metrics.Accuracy.toFixed(2)}%</dd></div>
          <div><dt>Model</dt><dd className="mono">ARIMA({r.order.join(',')})({r.seasonal.slice(0, 3).join(',')})<sub>24</sub></dd></div>
        </dl>
      </section>

      <section className="block">
        <h2>Demand and the capacity that covers it</h2>
        <p className="sub">The last {nPast} hours of English Wikipedia traffic, then the forecast. Each step of the capacity line is one VM, {capacity.toLocaleString('en-US')} requests an hour.</p>
        <Legend items={[['line muted', 'Observed'], ['line accent', 'Forecast'], ['area accent', '95% interval'], ['line ink', 'Provisioned capacity'], ['area ink', 'Headroom']]} />
        <TimeChart
          t={span}
          height={360}
          label="Observed demand, forecast demand with its 95% interval, and provisioned VM capacity"
          marker={{ index: nPast - 1, label: 'now' }}
          endLabel={{ index: span.length - 1, value: capLine[capLine.length - 1] as number, text: `${plan[plan.length - 1]} VMs` }}
          layers={[
            { kind: 'band', lower: fcOnly, upper: capLine, stepUpper: true, className: 'headroom' },
            { kind: 'band', lower, upper, className: 'interval' },
            { kind: 'line', values: demand, className: 's-muted' },
            { kind: 'line', values: fcLine, className: 's-accent', draw: true },
            { kind: 'step', values: capLine, className: 's-capacity' },
          ]}
          tooltip={(i) => {
            if (i < nPast) {
              return (
                <>
                  <TipRow value={full(pastY[i])} name="observed requests" swatch="muted" />
                  <TipRow value={`${vmsFor(pastY[i], capacity)} VMs`} name="would have been needed" />
                </>
              )
            }
            const k = i - nPast
            return (
              <>
                <TipRow value={full(forecast.predicted[k])} name="forecast requests" swatch="accent" />
                <TipRow value={`${compact(forecast.lower[k])} to ${compact(forecast.upper[k])}`} name="95% interval" />
                <TipRow value={`${plan[k]} VMs`} name={`${Math.round((forecast.predicted[k] / (plan[k] * capacity)) * 100)}% utilised`} swatch="ink" />
              </>
            )
          }}
        />
      </section>

      <section className="block">
        <h2>How the model did on hours it never saw</h2>
        <p className="sub">
          Trained on the first {r.n_train} hours, then predicted the last {testT.length} hours {r.mode === 'one_step' ? 'one at a time' : 'all at once'}.
        </p>
        <dl className="metrics">
          <div><dt>MAPE</dt><dd>{r.metrics.MAPE.toFixed(2)}%</dd></div>
          <div><dt>Accuracy, 100 minus MAPE</dt><dd>{r.metrics.Accuracy.toFixed(2)}%</dd></div>
          <div><dt>MAE, requests</dt><dd>{compact(r.metrics.MAE)}</dd></div>
          <div><dt>RMSE, requests</dt><dd>{compact(r.metrics.RMSE)}</dd></div>
          <div><dt>Training AIC</dt><dd>{full(r.aic)}</dd></div>
        </dl>
        <Legend items={[['line muted', 'Actual'], ['line accent', 'Predicted']]} />
        <TimeChart
          t={testT}
          height={300}
          fromZero={false}
          label="Actual versus predicted requests per hour in the test period"
          layers={[
            { kind: 'line', values: testY, className: 's-muted' },
            { kind: 'line', values: r.predicted, className: 's-accent', draw: true },
          ]}
          tooltip={(i) => (
            <>
              <TipRow value={full(testY[i])} name="actual" swatch="muted" />
              <TipRow value={full(r.predicted[i])} name="predicted" swatch="accent" />
              <TipRow value={`${(((r.predicted[i] - testY[i]) / testY[i]) * 100).toFixed(1)}%`} name="error" />
            </>
          )}
        />
      </section>

      <section className="block split">
        <div>
          <h2>Would the forecast have provisioned enough?</h2>
          <p className="sub">Each square is one test hour: the VMs the prediction asked for against the VMs the real traffic needed.</p>
          <ul className="tally">
            <li><span className="cell exact" aria-hidden="true" /><strong>{count('exact')}</strong> exactly right</li>
            <li><span className="cell over" aria-hidden="true" /><strong>{count('over')}</strong> too many, wasted cost</li>
            <li><span className="cell under" aria-hidden="true" /><strong>{count('under')}</strong> too few, QoS at risk</li>
          </ul>
        </div>
        <HourGrid
          t={testT}
          verdicts={verdicts}
          detail={(i) => `${stamp(testT[i])}: forecast asked for ${vmsFor(r.predicted[i], capacity)} VMs, traffic needed ${vmsFor(testY[i], capacity)}.`}
        />
      </section>

      <PlanTable t={forecast.t} predicted={forecast.predicted} plan={plan} current={current} capacity={capacity} />

      <footer className="method">
        <p>
          Implements the prediction module of Calheiros, Masoumi, Ranjan and Buyya,{' '}
          <a href="https://doi.org/10.1109/TCC.2014.2350475" target="_blank" rel="noreferrer">
            Workload Prediction Using ARIMA Model and Its Impact on Cloud Applications' QoS
          </a>
          , IEEE Transactions on Cloud Computing 3(4), 2015.
        </p>
        <p>
          Required VMs = ceil(predicted requests / capacity). Data: hourly English Wikipedia pageviews from the Wikimedia API, bots excluded. This page only calculates; it never starts or stops a VM.
        </p>
      </footer>
    </>
  )
}

function ActionIcon({ action }: { action: 'up' | 'down' | 'hold' }) {
  const Icon = action === 'up' ? ArrowUpIcon : action === 'down' ? ArrowDownIcon : EqualsIcon
  return <Icon className="action-icon" size={18} weight="bold" aria-hidden="true" />
}

function Legend({ items }: { items: [string, string][] }) {
  return (
    <ul className="legend">
      {items.map(([k, text]) => (
        <li key={text}><span className={`key ${k}`} aria-hidden="true" />{text}</li>
      ))}
    </ul>
  )
}

function PlanTable({ t, predicted, plan, current, capacity }: { t: string[]; predicted: number[]; plan: number[]; current: number; capacity: number }) {
  const rows = useMemo(
    () => plan.map((v, i) => ({
      t: t[i],
      requests: predicted[i],
      vms: v,
      change: v - (i === 0 ? current : plan[i - 1]),
      util: Math.round((predicted[i] / (v * capacity)) * 1000) / 10,
    })),
    [t, predicted, plan, current, capacity],
  )

  const download = () => {
    const head = 'hour_utc,predicted_requests,required_vms,change_vs_previous,action,utilization_pct'
    const body = rows.map((r) => [r.t, r.requests, r.vms, r.change, r.change > 0 ? 'Scale Up' : r.change < 0 ? 'Scale Down' : 'Maintain', r.util].join(','))
    const url = URL.createObjectURL(new Blob([[head, ...body].join('\n')], { type: 'text/csv' }))
    const a = Object.assign(document.createElement('a'), { href: url, download: 'vm_provisioning_plan.csv' })
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <section className="block">
      <div className="block-head">
        <h2>Hour-by-hour plan</h2>
        <button type="button" className="ghost" onClick={download}>
          <DownloadSimpleIcon size={16} weight="bold" aria-hidden="true" /> Download CSV
        </button>
      </div>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Hourly provisioning plan">
        <table className="plan">
          <thead>
            <tr><th>Hour, UTC</th><th>Forecast requests</th><th>VMs</th><th>Change</th><th>Utilisation</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.t}>
                <td>{stamp(r.t).replace(' UTC', '')}</td>
                <td>{full(r.requests)}</td>
                <td>{r.vms}</td>
                <td className={r.change > 0 ? 'up' : r.change < 0 ? 'down' : 'hold'}>
                  {r.change === 0 ? 'hold' : r.change > 0 ? `+${r.change}` : r.change}
                </td>
                <td>{r.util.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function Skeleton() {
  return (
    <div className="skeleton">
      <div className="sk sk-kicker" />
      <div className="sk sk-figure" />
      <div className="sk sk-line" />
      <div className="sk sk-chart" />
      <p className="sk-note">Fitting the seasonal ARIMA model. If the server was asleep, the first fit can take about a minute.</p>
    </div>
  )
}
