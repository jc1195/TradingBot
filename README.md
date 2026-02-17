# TradingBot (Robinhood Crypto + Hybrid AI)

Local-first Python trading bot with:
- Robinhood crypto integration (paper mode first)
- News + trend scoring pipeline
- Hybrid AI backend: Ollama local or OpenAI cloud reasoning
- Risk policy enforcement
- FastAPI dashboard (`/dashboard`) with live controls and monitoring

## 1) Quick start
1. Create a virtual environment
2. Install dependencies
3. Copy `.env.example` to `.env` and fill values
4. Run database/bootstrap and start app

### PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m src.trading_bot.main
```

### Run Dashboard (FastAPI)
```powershell
python ui/run_dashboard.py
```

Then open: `http://localhost:8501/dashboard`

### Legacy Streamlit UI
```powershell
streamlit run ui/dashboard.py
```

### Run UI (Python 3.14-safe launcher)
```powershell
./scripts/start_dashboard.ps1
```

### Low-resource startup (recommended)
```powershell
./scripts/start_all.ps1
```

### New laptop setup (after git clone)
Clone from the correct repo:
```powershell
git clone https://github.com/jc1195/TradingBot.git
cd TradingBot
```

Then run setup:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
./scripts/start_all.ps1 -InstallDeps
./scripts/start_dashboard.ps1
```

Optional CUDA check (Torch):
```powershell
.\.venv\Scripts\python.exe -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('cuda_device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Notes:
- GTX 1050 Ti Max-Q supports CUDA, but inference can be slower due to lower VRAM/throughput.
- The same model names can still be used; if a model is too heavy for the laptop, inference latency increases and fallbacks may trigger.

### Data migration vs fresh start
By default, the new laptop starts mostly fresh because local runtime/data artifacts are intentionally git-ignored.

What comes from GitHub:
- Source code, scripts, configs, docs.
- Adaptive logic/code paths and model code (same behavior/config capabilities).

What does NOT come from GitHub by default:
- SQLite runtime DB (for example `data/trading_bot.sqlite3`).
- Adaptive runtime memory (for example `runtime/special_circumstances.json`).
- Runtime snapshots/reports (`runtime/*.json`, `.pid`, local logs/caches).

If you want continuity (technician exceptions/history/ML training context), copy these from old machine to new machine after clone:
```powershell
# while bot/dashboard are stopped
Copy-Item "<old>\TradingBot\data\trading_bot.sqlite3" ".\data\trading_bot.sqlite3" -Force
Copy-Item "<old>\TradingBot\runtime\special_circumstances.json" ".\runtime\special_circumstances.json" -Force
```

Recommended:
- Start fresh first in paper mode to validate environment.
- Then copy DB/runtime files if you want to resume learned history.

One-command migration helpers:
```powershell
# on old machine (creates zip under runtime/)
./scripts/export_state.ps1

# on new machine (while bot/dashboard are stopped)
./scripts/import_state.ps1 -ArchivePath ".\runtime\state_export_YYYYMMDD_HHMMSS.zip"

# optional: overwrite existing local state files
./scripts/import_state.ps1 -ArchivePath ".\runtime\state_export_YYYYMMDD_HHMMSS.zip" -Overwrite
```

### Optional split-terminal startup
```powershell
./scripts/start_all.ps1 -SplitTerminals -StartDashboard
```

### Clean shutdown / cleanup
```powershell
./scripts/stop_all.ps1
```

### Emergency kill switch
```powershell
./scripts/kill_switch_on.ps1
./scripts/kill_switch_off.ps1
```

## 2) Safety
- Default mode is `paper`.
- Live mode requires both `BOT_MODE=live` and `LIVE_MODE_UNLOCK=true`.
- Risk limits are read from `config/risk_policy.yaml`.

## 3) Ollama + RTX 3070
- Ensure NVIDIA drivers are current.
- Run Ollama locally and pull model(s), for example:
```powershell
ollama pull deepseek-r1
ollama pull qwen2.5:14b
```
- Model auto-selection prioritizes strongest available model and falls back if needed.
- For all-day operation on mixed workloads, keep low-memory settings enabled in `.env`:
	- `OLLAMA_LOW_MEMORY_MODE=true`
	- `OLLAMA_GPU_LAYERS=24` (tune down if VRAM pressure)
	- `OLLAMA_NUM_CTX=2048`
	- `OLLAMA_KEEP_ALIVE=5m`
	- `OLLAMA_LATENCY_THRESHOLD_SECONDS=20`
	- `OLLAMA_MAX_RETRIES_PER_SYMBOL=3`
- Process startup script sets bot/UI to `BelowNormal` priority to reduce system contention.
- Inference fallback is automatic: on slow/error responses, the bot reduces context/predict budget and can switch to a lighter model.

## 3.1) OpenAI provider option
- Set `AI_PROVIDER=openai` to force OpenAI usage.
- Set `AI_PROVIDER=auto` to let the bot choose automatically.
- In `auto`, if `OPENAI_API_KEY` is configured and `PREFER_OPENAI_FOR_REASONING=true`, the bot prefers OpenAI for trend reasoning.
- If OpenAI is unavailable in `auto`, the engine falls back to Ollama and records an `ai_provider_fallback` event.
- Default OpenAI model is `gpt-5` (override with `OPENAI_MODEL`).

## 4) Current status
Phase 1 scaffold is in place and Phase 2 integration is active for:
- Robinhood session/auth handling with paper-safe fallback behavior
- Account snapshot fetch wrapper
- Crypto symbols, quote, and historical candle wrappers
- Retry/backoff and API error classification

Phase 3 ingestion is now active for:
- RSS feed ingestion from configurable crypto news sources
- Symbol/alias filtering per coin candidate
- Lightweight keyword sentiment scoring
- Persistence into `news_items` and `system_events`

Strategy/risk proposal layer is now active for:
- AI action + confidence gating before proposals
- Position sizing from buying power and configured risk caps
- Policy-based live blocking (`blocked_policy`) when live orders are disallowed
- Order proposals persisted to `orders` and displayed in dashboard

Paper execution simulation is now active for:
- Automatic paper fills from proposed orders
- Configurable slippage/fee assumptions
- Auto-close after minimum hold window
- `trade_outcomes` PnL tracking and rolling `daily_reviews` summary

Execution-quality analytics are now active for:
- Weekly report card metrics (trades, PnL, win rate, hold time)
- Per-symbol hit-rate table
- Confidence calibration buckets (confidence vs realized outcomes)
- Drawdown and quality alerts logged to `system_events`

Earnings goals are now supported:
- Set `DAILY_PROFIT_GOAL_USD` and `WEEKLY_PROFIT_GOAL_USD` in `.env`
- Goal progress is shown in dashboard with daily/weekly progress bars
- If `PAUSE_NEW_ORDERS_ON_GOAL_HIT=true`, new order proposals pause after target is reached
- Optional goal pacing mode dynamically scales risk multiplier within bounds based on pace-to-target ratios
- Set `GOAL_PACING_PRESET` to `conservative`, `balanced`, `aggressive`, or `custom`
- Use `custom` to apply your own multiplier bounds/weights/sensitivity env settings

Paper bankroll cap is now supported:
- Set `PAPER_WORKING_CAPITAL_USD` to cap paper-mode capital (example: `500`)
- In paper mode, position sizing uses this capped bankroll and grows/shrinks from realized paper PnL
- Dashboard shows configured paper capital, current bankroll, and growth vs start

Paper mode with real data only:
- Set `PAPER_REQUIRE_LIVE_MARKET_DATA=true` (default)
- When enabled, symbols are skipped unless Robinhood quote data is authenticated and has a valid mark price
- This prevents paper training on fallback/mock quote values

Phase 7 live guardrails now include:
- External kill-switch file enforcement (`EXTERNAL_KILL_SWITCH_PATH`)
- Live max per-order notional cap (`LIVE_MAX_ORDER_NOTIONAL_USD`)
- Live max daily notional cap (`LIVE_MAX_DAILY_NOTIONAL_USD`)
- Live max order count per day (`LIVE_MAX_ORDERS_PER_DAY`)
- Optional webhook alert routing for critical events (`ALERT_WEBHOOK_URL`)

Phase 7 operator workflow:
- Run preflight before any live start: `./scripts/live_preflight.ps1`
- Follow live runbook: [docs/LIVE_RUNBOOK.md](docs/LIVE_RUNBOOK.md)

Trade execution placement and deeper post-trade learning enhancements are the next phases.
