from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from .engine import TradingEngine
from .risk import load_risk_policy
from .settings import settings


class BotScheduler:
    def __init__(self) -> None:
        self.scheduler = BackgroundScheduler(timezone=settings.timezone)
        self.engine = TradingEngine(load_risk_policy())

    def start(self) -> None:
        trigger = CronTrigger(hour=settings.daily_run_hour, minute=settings.daily_run_minute)
        self.scheduler.add_job(
            self.engine.run_daily_cycle,
            trigger=trigger,
            id="daily_trading_cycle",
            replace_existing=True,
        )
        self.scheduler.start()
        logger.info(
            "Scheduler started. Daily cycle at {:02d}:{:02d} {}",
            settings.daily_run_hour,
            settings.daily_run_minute,
            settings.timezone,
        )

    def stop(self) -> None:
        self.scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
