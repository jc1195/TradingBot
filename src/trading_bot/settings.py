from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bot_mode: Literal["paper", "live"] = "paper"
    live_mode_unlock: bool = False
    require_explicit_live_confirm: bool = True
    ai_provider: Literal["auto", "ollama", "openai"] = "auto"
    prefer_openai_for_reasoning: bool = True

    robinhood_username: str = ""
    robinhood_password: str = ""
    robinhood_mfa_code: str = ""
    robinhood_max_retries: int = Field(default=3, ge=1, le=10)
    robinhood_retry_base_delay_seconds: float = Field(default=1.0, ge=0.1, le=30)
    news_feed_urls_csv: str = "https://www.coindesk.com/arc/outboundfeeds/rss/,https://cointelegraph.com/rss"
    news_request_timeout_seconds: int = Field(default=12, ge=3, le=60)
    news_max_items_per_feed: int = Field(default=25, ge=5, le=200)
    news_max_items_per_symbol: int = Field(default=8, ge=1, le=50)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = ""
    ollama_timeout_seconds: int = 120
    prefer_gpu: bool = True
    ollama_low_memory_mode: bool = True
    ollama_gpu_layers: int = Field(default=24, ge=0, le=120)
    ollama_num_ctx: int = Field(default=2048, ge=256, le=16384)
    ollama_num_predict: int = Field(default=512, ge=64, le=4096)
    ollama_keep_alive: str = "5m"
    ollama_latency_threshold_seconds: int = Field(default=20, ge=2, le=300)
    ollama_max_retries_per_symbol: int = Field(default=3, ge=1, le=10)

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-5"
    openai_timeout_seconds: int = Field(default=90, ge=5, le=300)
    openai_max_output_tokens: int = Field(default=600, ge=64, le=4096)
    openai_max_retries_per_symbol: int = Field(default=3, ge=1, le=10)

    paper_trading_enabled: bool = True
    paper_auto_close_enabled: bool = True
    paper_min_hold_minutes: int = Field(default=180, ge=1, le=10080)
    paper_early_climb_exit_enabled: bool = True
    paper_early_climb_take_profit_pct: float = Field(default=0.05, ge=0.005, le=1.0)
    paper_early_climb_stop_loss_pct: float = Field(default=0.025, ge=0.001, le=1.0)
    paper_early_climb_trailing_stop_pct: float = Field(default=0.015, ge=0.001, le=1.0)
    paper_early_climb_max_hold_minutes: int = Field(default=90, ge=1, le=10080)
    paper_slippage_bps: float = Field(default=8.0, ge=0, le=500)
    paper_fee_bps: float = Field(default=5.0, ge=0, le=500)
    paper_require_live_market_data: bool = True
    review_lookback_days: int = Field(default=7, ge=1, le=90)
    analytics_drawdown_alert_pct: float = Field(default=0.05, ge=0.001, le=1)
    analytics_min_trades_for_quality_alert: int = Field(default=8, ge=1, le=500)
    analytics_min_win_rate: float = Field(default=0.45, ge=0, le=1)
    paper_starting_equity: float = Field(default=10000, ge=100, le=10_000_000)
    paper_working_capital_usd: float = Field(default=2000, ge=0, le=10_000_000)
    paper_extra_play_cash_usd: float = Field(default=200, ge=0, le=1_000_000)
    daily_profit_goal_usd: float = Field(default=0, ge=0, le=1_000_000)
    weekly_profit_goal_usd: float = Field(default=0, ge=0, le=10_000_000)
    pause_new_orders_on_goal_hit: bool = True
    goal_pacing_enabled: bool = True
    goal_pacing_preset: Literal["conservative", "balanced", "aggressive", "custom"] = "balanced"
    goal_pacing_min_multiplier: float = Field(default=0.6, ge=0.1, le=2.0)
    goal_pacing_max_multiplier: float = Field(default=1.4, ge=0.5, le=3.0)
    goal_pacing_weight_daily: float = Field(default=0.5, ge=0, le=1)
    goal_pacing_weight_weekly: float = Field(default=0.5, ge=0, le=1)
    goal_pacing_sensitivity: float = Field(default=0.35, ge=0.01, le=2.0)

    timezone: str = "America/New_York"
    daily_run_hour: int = 8
    daily_run_minute: int = 0
    service_heartbeat_seconds: int = Field(default=15, ge=5, le=300)
    max_symbols_per_cycle: int = Field(default=4, ge=1, le=50)
    dynamic_discovery_enabled: bool = True
    scan_min_score: int = Field(default=15, ge=-100, le=100)
    scan_top_n: int = Field(default=10, ge=1, le=50)
    scan_min_avg_daily_move_stock_pct: float = Field(default=2.0, ge=0, le=100)
    scan_min_avg_daily_move_crypto_pct: float = Field(default=3.0, ge=0, le=100)
    scan_min_last_day_move_pct: float = Field(default=1.0, ge=0, le=100)
    scan_min_volume_vs_avg: float = Field(default=0.7, ge=0, le=20)
    scan_volatility_weight: float = Field(default=0.35, ge=0, le=2.0)
    scan_activity_weight: float = Field(default=0.25, ge=0, le=2.0)
    scan_dynamic_discovery_limit: int = Field(default=25, ge=0, le=200)
    discovery_enable_yahoo_most_active: bool = True
    discovery_enable_coingecko_trending: bool = True
    discovery_enable_stocktwits_trending: bool = True
    discovery_enable_reddit_wsb: bool = True
    discovery_enable_robinhood_top100: bool = True
    discovery_enable_robinhood_top_movers: bool = True
    discovery_reddit_post_limit: int = Field(default=80, ge=10, le=500)
    discovery_enable_reddit_extra: bool = True
    discovery_reddit_extra_subreddits_csv: str = "stocks,options,CryptoCurrency"
    daytrade_only_mode: bool = True
    daytrade_min_avg_bar_move_stock_pct: float = Field(default=0.30, ge=0, le=100)
    daytrade_min_avg_bar_move_crypto_pct: float = Field(default=0.45, ge=0, le=100)
    daytrade_min_avg_bar_range_stock_pct: float = Field(default=0.60, ge=0, le=100)
    daytrade_min_avg_bar_range_crypto_pct: float = Field(default=0.90, ge=0, le=100)
    daytrade_min_recent_bar_move_pct: float = Field(default=0.40, ge=0, le=100)
    daytrade_min_bar_volume_ratio: float = Field(default=0.80, ge=0, le=50)
    daytrade_min_score: float = Field(default=45.0, ge=0, le=100)
    daytrade_early_climb_enabled: bool = True
    daytrade_early_climb_min_net_climb_pct: float = Field(default=1.2, ge=0, le=100)
    daytrade_early_climb_min_up_step_ratio: float = Field(default=0.60, ge=0, le=1)
    daytrade_early_climb_min_acceleration_pct: float = Field(default=0.10, ge=0, le=100)
    daytrade_early_climb_max_distance_from_low_pct: float = Field(default=8.0, ge=0, le=100)
    daytrade_early_climb_max_pullback_pct: float = Field(default=1.2, ge=0, le=100)
    daytrade_early_climb_min_volume_ratio: float = Field(default=0.90, ge=0, le=50)
    daytrade_early_climb_min_score: float = Field(default=50.0, ge=0, le=100)
    adaptive_special_circumstances_enabled: bool = False
    adaptive_special_circumstances_paper_enabled: bool = True
    adaptive_special_circumstances_live_enabled: bool = False
    adaptive_technician_enabled: bool = True
    adaptive_exception_scope: Literal["symbol", "asset_class", "global"] = "asset_class"
    adaptive_special_circumstances_path: Path = Path("runtime/special_circumstances.json")
    adaptive_hourly_rule_generation_enabled: bool = False
    adaptive_max_new_rule_pairs_per_hour: int = Field(default=2, ge=0, le=20)
    adaptive_min_candidate_daytrade_score_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
    adaptive_rule_pair_max_losses: int = Field(default=2, ge=1, le=20)
    adaptive_rule_pair_max_net_loss_usd: float = Field(default=20.0, ge=0.0, le=100000.0)
    low_resource_mode: bool = True
    log_level: str = "INFO"
    db_path: Path = Path("data/trading_bot.sqlite3")

    max_risk_per_trade_pct: float = Field(default=0.01, ge=0, le=1)
    max_daily_realized_loss_pct: float = Field(default=0.02, ge=0, le=1)
    max_concurrent_positions: int = Field(default=3, ge=1, le=50)
    min_confidence_to_trade: float = Field(default=0.62, ge=0, le=1)
    live_max_order_notional_usd: float = Field(default=75, ge=1, le=1_000_000)
    live_max_daily_notional_usd: float = Field(default=300, ge=1, le=10_000_000)
    live_max_orders_per_day: int = Field(default=5, ge=1, le=500)
    external_kill_switch_path: str = "runtime/kill_switch.flag"
    alert_webhook_url: str = ""
    alert_webhook_timeout_seconds: int = Field(default=10, ge=2, le=60)
    alert_event_types_csv: str = "pipeline_error,kill_switch,drawdown_alert,quality_alert,ai_provider_fallback"

    # Backtest defaults
    backtest_default_period: str = "6mo"
    backtest_default_interval: str = "1d"
    backtest_default_capital: float = Field(default=500, ge=10, le=10_000_000)
    backtest_lookback_window: int = Field(default=24, ge=4, le=200)
    backtest_hold_candles: int = Field(default=6, ge=1, le=100)
    backtest_min_confidence: float = Field(default=0.55, ge=0, le=1)

    # Council (3-agent AI voting)
    use_council: bool = True
    council_require_unanimous_buy: bool = True

    @model_validator(mode="after")
    def validate_live_mode(self) -> "Settings":
        if self.bot_mode == "live" and self.require_explicit_live_confirm and not self.live_mode_unlock:
            raise ValueError(
                "Live mode is blocked. Set LIVE_MODE_UNLOCK=true to explicitly permit live execution."
            )
        return self


settings = Settings()
