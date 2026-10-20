# Cloud Workload Prediction with ARIMA

Forecasts hourly cloud workload with a seasonal ARIMA model and turns the forecast into a
VM provisioning plan. A FastAPI backend runs the model; a React web app ("Headroom") shows
the forecast, the model's accuracy on unseen hours, and how many VMs to run each hour.

> **Scope note.** This project implements the *ARIMA-based workload prediction and VM
> provisioning concept* from the paper below. It does **not** reproduce the paper's exact
> results (different data, a fixed-capacity VM model instead of a queueing model, and no
> CloudSim QoS experiments).

## 1. Project objective

Cloud applications need enough VMs to meet demand (QoS) without paying for idle VMs.
The project:

1. takes **real** hourly web traffic,
2. forecasts the next hours with **seasonal ARIMA**,
3. measures forecast accuracy on held-out data (MAE, RMSE, MAPE),
4. translates the forecast into the **number of VMs** needed, and
5. recommends **Scale Up / Scale Down / Maintain** for every forecast hour.

No VM is launched anywhere: the project only *calculates* the recommendation.

## 2. Paper

R. N. Calheiros, E. Masoumi, R. Ranjan, R. Buyya, *"Workload Prediction Using ARIMA Model
and Its Impact on Cloud Applications' QoS"*, IEEE Transactions on Cloud Computing 3(4),
pp. 449-458, 2015. DOI: [10.1109/TCC.2014.2350475](https://doi.org/10.1109/TCC.2014.2350475).
Free PDF: <https://eprints.ncl.ac.uk/226309>.

The paper predicts future request load with ARIMA so resources can be provisioned *before*
the load arrives, and studies how prediction accuracy affects QoS and cost. This project
reproduces that idea end to end with real Wikipedia traffic (the paper also used Wikipedia
traces, but not this exact dataset).

## 3. Architecture

```
Wikimedia API -> src/data_loader.py -> src/preprocessing.py -> data/workload.csv
                                                                     |
                                                                     v
web/ (React)  <--JSON--  api/main.py (FastAPI)  <--  src/arima_model.py + src/evaluation.py
     |
     +-- VM plan: ceil(predicted / capacity), same rule as src/vm_calculator.py
```

| Path | Responsibility |
|------|----------------|
| `src/data_loader.py` | Download hourly pageviews, retry on errors, cache in `data/workload.csv` |
| `src/preprocessing.py` | Sort, de-duplicate, fill missing hours, validate |
| `src/arima_model.py` | 80/20 split, seasonal ARIMA fit, test prediction, future forecast, order suggestion |
| `src/evaluation.py` | MAE, RMSE, MAPE, project accuracy |
| `src/vm_calculator.py` | VM count, scaling action, provisioning quality |
| `api/main.py` | FastAPI app: JSON endpoints, result caching, serves the built web app |
| `web/` | React + TypeScript frontend (Vite) |
| `app.py` | Original Streamlit dashboard (still works; optional) |
| `Dockerfile`, `render.yaml` | One container with API + web app; Render blueprint |

### API

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/api/health` | | `{"ok": true}` |
| POST | `/api/run` | `p, d, q, train_ratio, mode ("one_step" or "multi_step"), horizon, refresh` | history, test predictions, metrics, future forecast with 95% interval |
| POST | `/api/suggest` | `train_ratio` | best `(p, d, q)` by ADF test + AIC, top candidates |

## 4. Data source

Official Wikimedia Analytics (pageviews) REST API, no key required:

```
https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/
    en.wikipedia/all-access/user/hourly/{YYYYMMDDHH}/{YYYYMMDDHH}
```

* project `en.wikipedia`, access `all-access`, agent `user` (human traffic, bots excluded)
* hourly granularity, UTC timestamps, the last 30 days (about 720 rows)
* the cleaned result is saved to `data/workload.csv` (`timestamp, requests, imputed`)
* a CSV younger than 6 hours is reused; an older one is replaced by a fresh download, and
  if the download fails the older CSV is used instead

Because data is refreshed automatically, results change over time. Keep a copy of
`data/workload.csv` if you need to reproduce a specific set of numbers.

Set the `WIKI_USER_AGENT` environment variable to identify your client to Wikimedia.

## 5. ARIMA methodology

ARIMA(p, d, q) combines **AR** (p past values), **I** (d rounds of differencing) and
**MA** (q past forecast errors). A seasonal part `(1, 0, 1, 24)` models the daily cycle.

1. **Clean**: sort, drop duplicates, reindex to a complete hourly range, treat zero counts as
   missing, interpolate gaps (at most 20% missing is accepted).
2. **Split**: first 80% of hours for training, last 20% for testing (chronological).
3. **Train**: fit `statsmodels` ARIMA with the chosen `(p, d, q)` and seasonal `(1, 0, 1, 24)`.
4. **Predict the test period** in one of two modes:
   * *One-step (hour by hour)*, default: each hour is predicted after the real previous
     hours are appended, with parameters fixed from training (`append(refit=False)`).
     This matches how a provisioner works.
   * *Multi-step (all at once)*: the whole test period is forecast without seeing it.
5. **Forecast the future**: refit the same configuration on all data and forecast the next
   12-72 hours with a 95% interval.
6. **Order suggestion** (optional): choose `d` with the Augmented Dickey-Fuller test, then
   `(p, q)` with the lowest AIC on the training data.

## 6. Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| MAE | mean(abs(actual - predicted)) | average miss, in requests/hour |
| RMSE | sqrt(mean((actual - predicted)^2)) | like MAE but penalises large misses |
| MAPE | mean(abs(actual - predicted) / actual) x 100 | average miss in percent |
| Accuracy | 100 - MAPE | project's own headline figure (not clipped) |
| AIC | from statsmodels | model fit vs. complexity; only for comparing orders |

The provisioning check counts test hours where the VMs from the prediction were exactly
right, too many (wasted cost) or too few (QoS risk).

Example results with ARIMA(2,1,2)(1,0,1,24) on one 30-day snapshot: one-step MAPE 2.65%
(accuracy 97.35%), multi-step MAPE 3.75% (96.25%). Results vary with the data window.

## 7. VM calculation

```
required_vms = max(1, ceil(predicted_requests / vm_capacity))
```

* **Scale Up** if required VMs exceed the current VMs, **Scale Down** if fewer, else **Maintain**.
* "VMs running now" is set in the web app; 0 means "what the last observed hour needed".
* VM capacity (default 500,000 requests/hour) is a configurable project assumption, not an
  AWS instance specification.

## 8. Run locally

Requires Python 3.10+, Node.js 20.19+ and internet access to `wikimedia.org`.

```bash
# backend
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# frontend (build once; the API then serves it)
cd web
npm install
npm run build
cd ..

uvicorn api.main:app --port 8000    # open http://localhost:8000
```

For frontend development with hot reload, keep uvicorn running and in another terminal run
`npm run dev` inside `web/` (Vite proxies `/api` to port 8000, opens on port 5173).

The original Streamlit dashboard still works: `streamlit run app.py` (port 8501).

## 9. Deployment

### Render (API + web app in one service)

1. Push the repo to GitHub.
2. On render.com: **New > Blueprint**, select the repo. Render reads `render.yaml` and
   builds the `Dockerfile`.
3. Open `https://<service>.onrender.com`. Health check: `/api/health`.

The free plan sleeps after about 15 minutes idle, so the first request can take about a
minute.

### Vercel (frontend only, API on Render)

The Python/statsmodels backend is too large and slow for Vercel functions, so Vercel hosts
only `web/`:

1. Import the repo in Vercel and set **Root Directory** to `web` (framework: Vite).
2. Add the environment variable `VITE_API_URL=https://<service>.onrender.com` and deploy.
   It is read at build time, so redeploy after changing it.
3. On Render, set `ALLOWED_ORIGINS=https://<project>.vercel.app` (comma-separated list).

### AWS EC2

On Ubuntu 24.04 (`t3.micro` works with 2 GB swap; `t3.small` is more comfortable), open
inbound **TCP 8000** in the security group, then either:

```bash
# Docker
sudo apt update && sudo apt install -y docker.io
sudo docker build -t headroom .
sudo docker run -d --restart unless-stopped -p 8000:8000 headroom
```

or run `uvicorn api.main:app --host 0.0.0.0 --port 8000` from a virtual environment
(build `web/` first so the site is served). Open `http://<EC2_PUBLIC_IP>:8000`.

## 10. Environment variables

| Variable | Where | Default | Purpose |
|----------|-------|---------|---------|
| `PORT` | API | `8000` | Port uvicorn listens on (set by Render) |
| `ALLOWED_ORIGINS` | API | `*` | CORS origins allowed to call the API |
| `WIKI_USER_AGENT` | API | project string | User-Agent sent to Wikimedia |
| `VITE_API_URL` | web build | empty (same origin) | API base URL when the frontend is hosted separately |
