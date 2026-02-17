# Robinhood Crypto AI Trader - Execution Plan

## 0) Mission
Build a production-minded Python trading system for Robinhood crypto that:
- Scans eligible coins daily
- Pulls market + news context
- Uses a local Ollama model to score trends and trade ideas
- Executes trades with strict risk controls
- Logs every decision, outcome, and post-trade analysis so the strategy can improve over time
- Provides a modern, highly configurable local UI dashboard for monitoring and control

> This project is for software engineering and strategy research. Live trading has real financial risk.

---

## 1) Scope (MVP first)
### In scope
- Robinhood authentication + account state fetch
- Crypto watchlist generation and filtering
- Daily news aggregation per candidate coin
- Trend scoring via local Ollama model
- Trade signal generation (long/flat for MVP)
- Position sizing + risk guardrails
- Paper-trading mode first, then optional live mode
- Full audit log + “why this trade” notes

### Out of scope (for initial MVP)
- Leverage / margin
- High-frequency intraday scalping
- Multi-broker routing
- Complex derivatives/options

---

## 2) Proposed stack
- **Language:** Python 3.11+
- **Broker API:** `robin_stocks` (community Robinhood library)
- **Data/TA:** `pandas`, `numpy`, `ta`, `yfinance` (backup market context), optional `ccxt` for cross-exchange price checks
- **News:** RSS + API source(s) (e.g., NewsAPI/CryptoPanic if needed)
- **AI local model runtime:** Ollama
- **Model client:** direct Ollama HTTP API (`http://localhost:11434`) and optional OpenAI API backend
- **UI:** Streamlit dashboard (dark modern theme, live refresh, controls panel)
- **Storage:** SQLite for logs + features + outcomes
- **Scheduler:** APScheduler (or Windows Task Scheduler integration)
- **Config:** `.env` + `pydantic-settings`
- **Observability:** structured logs (`loguru` or stdlib logging + JSON)

---

## 3) Target architecture (phased)
1. **Collector Layer**
   - Robinhood market/account data
   - News ingest per symbol
2. **Feature Layer**
   - Technical indicators (momentum, volatility, trend)
   - News sentiment & narrative tags
3. **AI Decision Layer**
   - Ollama model analyzes combined context
   - Outputs structured decision JSON (score, confidence, rationale)
4. **Execution Layer**
   - Policy checks (risk limits, cooldowns, max positions)
   - Paper/live order placement
5. **Learning Layer**
   - Persist trade decisions and outcomes
   - Daily review job generates “what worked / failed” notes
6. **UI Layer**
   - Real-time dashboard with strategy controls and model/settings panels
   - Manual safety actions (pause, resume, kill switch, mode lock)

---

## 4) Phase plan

## Phase 1 - Foundation & safety rails
**Goal:** Runnable skeleton with secure config and strict operational boundaries.

### Deliverables
- Project structure initialized
- Env/config system (`.env`, settings model)
- Secrets handling policy
- Global risk policy file (`max_daily_loss`, `max_position_pct`, etc.)
- Mode switch: `paper` vs `live` (default `paper`)
- UI shell with modern layout + configurable strategy/risk controls

### Exit criteria
- App starts cleanly and validates config
- Refuses to run live mode without explicit confirmation flags

---

## Phase 2 - Robinhood integration + market data
**Goal:** Connect account and fetch tradable crypto data reliably.

### Deliverables
- Login/session manager
- Account balance + buying power fetch
- Crypto quote + historical candle fetch wrappers
- Symbol universe discovery and filtering
- Retry + backoff + error classification

### Exit criteria
- We can fetch account state and candles for selected symbols repeatedly without manual fixes

---

## Phase 3 - News ingestion & enrichment
**Goal:** Daily news context for each candidate coin.

### Deliverables
- News connectors (RSS/API)
- Deduplication + freshness scoring
- Coin-symbol/entity mapping
- Persisted daily news snapshots in SQLite

### Exit criteria
- Daily run produces a clean news bundle per symbol with timestamps and source links

---

## Phase 4 - AI trend engine (Ollama)
**Goal:** Local model produces structured trend/trade recommendations.

### Deliverables
- Ollama model selector utility:
  - Discovers installed models
  - Picks strongest available by policy (quality-first, resource-aware fallback)
- Prompt contract producing strict JSON
- Output validator (schema + bounds checks)
- Decision score + confidence + rationale extraction

### Exit criteria
- For each symbol, model returns valid structured output
- Invalid/non-JSON outputs are retried and safely handled

---

## Phase 5 - Strategy + risk engine
**Goal:** Turn signals into safe, explainable orders.

### Deliverables
- Signal fusion (technical + news + AI)
- Position sizing algorithm
- Trade gating rules (max open positions, cooldowns, drawdown stop)
- Stop-loss / take-profit policy support

### Exit criteria
- Every proposed order includes a full rationale and passes policy checks

---

## Phase 6 - Paper trading loop + evaluation
**Goal:** End-to-end automated simulation before any live trades.

### Deliverables
- Scheduled daily workflow
- Paper execution simulator with slippage/fees assumptions
- Daily and weekly performance reports
- Trade journal entries with reason and post-mortem fields
- Dashboard views for PnL, open positions, signal rationale, and model decisions

### Exit criteria
- At least 2-4 weeks of stable paper-trading metrics and logs

---

## Phase 7 - Controlled live rollout
**Goal:** Small, constrained live deployment.

### Deliverables
- Live guardrails (small caps, kill switch, daily loss stop)
- Alerting (email/Discord/Telegram optional)
- Runbook for failures and manual override

### Exit criteria
- Live mode runs with no policy violations and complete audit trail

---

## 5) Model selection policy (Ollama)
- Prefer best-performing local model available for reasoning + structured output.
- Use NVIDIA RTX 3070 acceleration through Ollama's GPU runtime when available.
- Start with this priority order if installed:
  1. `deepseek-r1` (strong reasoning)
  2. `qwen2.5:32b` or highest-size `qwen2.5` available
  3. `llama3.3:70b` (if hardware supports)
  4. `llama3.1:8b` fallback
- Final selection must also satisfy runtime limits (latency/memory).
- For RTX 3070 VRAM limits, prefer quantized variants for larger models when needed.
- If top model is too slow, use a two-stage approach:
  - Fast model for candidate filtering
  - Best model for final trade decision on short-list

## 5.1) Model selection policy (OpenAI optional)
- Optional cloud reasoning backend for higher-quality trend interpretation.
- In `AI_PROVIDER=auto`, prefer OpenAI when API key is configured and policy flag is enabled.
- If OpenAI backend errors in `auto`, fall back to Ollama and log provider fallback telemetry.
- Default OpenAI model target is `gpt-5`, configurable via environment.

---

## 6) Data schema (initial)
### Tables
- `market_snapshots` - candles/features by symbol/time
- `news_items` - source/title/url/published_at/symbol tags
- `ai_decisions` - prompt hash/model/raw + parsed JSON output
- `orders` - proposed vs executed details
- `trade_outcomes` - PnL, hold time, drawdown
- `daily_reviews` - summary of wins/losses + lessons

---

## 7) Risk policy baseline (initial defaults)
- Max risk per trade: `1%` of account
- Max daily realized loss: `2%`
- Max concurrent positions: `3`
- No trade if confidence below threshold (to be tuned)
- Circuit breaker on API/model failures above threshold

---

## 8) Automation plan
- Scheduler trigger at defined daily time (pre-market and/or crypto session window)
- Bot service designed to run continuously on local PC with heartbeat monitoring and auto-restart hooks.
- Runtime resource policy: low-memory Ollama defaults, bounded context window, and clean process lifecycle scripts.
- Pipeline steps:
  1. Refresh symbols
  2. Pull market data
  3. Pull news
  4. Run AI scoring
  5. Generate trade candidates
  6. Apply risk checks
  7. Execute paper/live
  8. Persist logs + send summary

---

## 9) Execution tracker (update this every session)

### Current phase
- **Phase:** 7 - Controlled live rollout
- **Status:** In Progress (guardrails implementation)
- **Next milestone:** Run staged live dry-run with preflight passing under constrained live caps

### Session log
| Date | What we built | Issues encountered | Decisions made | Next action |
|---|---|---|---|---|
| 2026-02-14 | Created initial phased execution plan | None | Paper-trading first, then controlled live rollout | Scaffold project structure |
| 2026-02-14 | Added UI + GPU requirements into plan | None | Build modern Streamlit dashboard; target RTX 3070 accelerated Ollama | Scaffold Phase 1 codebase |
| 2026-02-14 | Added low-resource runtime defaults and process hygiene | Tooling diagnostics glitch; validated via direct Python checks | Default to single-process service, manual dashboard refresh, capped symbols per cycle | Implement Robinhood auth + news ingestion (Phase 2/3) |
| 2026-02-14 | Added low-RAM runtime controls and process cleanup flow | Dependency install was interrupted | Prioritize system stability; keep bot/UI at below-normal priority | Continue with dependency install when system load allows |
| 2026-02-14 | Implemented adaptive AI fallback (latency/error based) | None | Auto-downgrade inference profile/model before failing cycle | Validate in live Ollama environment |
| 2026-02-14 | Implemented Robinhood Phase 2 wrappers | None | Added session/auth, account, quote, candles, and retry/backoff with paper-safe fallback | Improve news ingestion and strategy quality in Phase 3/5 |
| 2026-02-14 | Implemented Phase 3 RSS news ingestion | None | Added symbol-aware feed filtering and persisted snapshots for daily context | Build strategy fusion and risk gating for order proposals |
| 2026-02-14 | Implemented risk-gated order proposal workflow | None | Only BUY signals above confidence threshold generate proposals; live proposals blocked by policy by default | Add execution simulation and performance analytics |
| 2026-02-14 | Added hybrid AI provider support (OpenAI + Ollama) | None | In auto mode prefer OpenAI for reasoning and fail over to Ollama on backend failure | Validate provider behavior with real API key |
| 2026-02-14 | Implemented Phase 6 paper execution simulation | None | Added paper fills/closes with slippage+fees and persisted outcomes/reviews | Add weekly performance dashboard and execution quality alerts |
| 2026-02-14 | Added execution-quality analytics and alerting | None | Added weekly report card, symbol hit rate, confidence calibration, and drawdown/quality alerts | Begin Phase 7 guarded live rollout prep |
| 2026-02-14 | Added daily/weekly earnings goal controls | None | Goals now track progress and can pause new proposals when targets are hit | Tune goals versus risk settings during paper run |
| 2026-02-14 | Added dynamic goal pacing strategy mode | None | Risk budget now scales within min/max bounds based on pace-to-goal ratios | Calibrate pacing sensitivity during paper trial |
| 2026-02-14 | Added pacing presets and tracking updates | None | Added conservative/balanced/aggressive/custom presets and surfaced active preset in dashboard telemetry | Choose preset and tune based on paper results |
| 2026-02-14 | Started Phase 7 live guardrail rollout | None | Added external kill switch scripts/checks, live notional caps, and webhook alert routing hooks | Prepare operator runbook and dry-run live safety checks |
| 2026-02-14 | Added live runbook + preflight gate script | Initial preflight script had result aggregation bug (fixed) | Live enablement now requires passing scripted checks; verified failure in paper mode as expected | Configure a constrained live dry-run profile and validate a full preflight pass |
| 2026-02-14 | Added week-over-week paper learning context in AI prompts | None | Decision prompts now include last-lookback performance stats and latest review lessons for adaptive paper behavior | Run one-week paper trial and compare week-2 outcomes against week-1 baseline |
| 2026-02-14 | Launched dashboard and added weekly paper review checklist | Streamlit CLI hit Python 3.14 event-loop issue | Added compatibility launcher `ui/run_dashboard.py`; created repeatable weekly review checklist doc | Complete week-1 run and execute checklist before tuning week-2 settings |
| 2026-02-14 | Fixed dashboard import/path startup issues | `ModuleNotFoundError: src` and Python 3.14 event-loop startup friction | Added robust project-root path setup in dashboard and non-deprecated event-loop compat shim; added `scripts/start_dashboard.ps1` one-click launcher | Keep paper run active for 7 days and perform weekly checklist review |
| 2026-02-14 | Added capped paper bankroll mode + selected Ollama model | None | Added `PAPER_WORKING_CAPITAL_USD` sizing cap so paper trading can run as a `$500 grow` challenge; selected best installed model `llama3.2:latest` | Restart services and monitor bankroll growth metrics for week-1 baseline |
| 2026-02-14 | Enforced real-data-only paper practice mode | None | Added `PAPER_REQUIRE_LIVE_MARKET_DATA=true` guard so symbols are skipped unless authenticated Robinhood quotes are available | Add Robinhood credentials and verify authenticated snapshots before week-long practice run |
| 2026-02-15 | Built historical backtesting system + stock support | None | Added: `watchlist.py` (20 volatile crypto+stock tickers), `historical_data.py` (yfinance fetcher), `backtester.py` (full replay engine with AI, sizing, slippage/fees), stock methods in `robinhood_client.py`, backtest DB tables, dashboard section with equity curve, `run_backtest.ps1` script. $500 sim capital, GPU via Ollama. | Run first backtest: `scripts/run_backtest.ps1 -Symbols DOGE` then review results on dashboard |

### Open questions
- Which news APIs do you want to rely on first (free-only vs paid)?
- Preferred alert channel (none/email/Discord/Telegram)?
- Minimum paper-trading duration before live (2 vs 4 weeks)?

---

## 10) Definition of done (project)
- Daily autonomous run is stable
- Every trade has explainable rationale + stored context
- Risk limits are enforced in code (not just config text)
- Paper performance reviewed and accepted
- Live mode can be disabled instantly (kill switch)

---

## 11) Build notes template (copy per day)
### YYYY-MM-DD Notes
- **Goal:**
- **Changes made:**
- **Tested:**
- **Problems:**
- **Fixes/decisions:**
- **Next step:**
