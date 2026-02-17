from __future__ import annotations

import time

from loguru import logger

from .db import initialize_database
from .engine import validate_live_mode_guardrails
from .logging_config import configure_logging
from .scheduler import BotScheduler
from .settings import settings



def run_service() -> None:
    configure_logging(settings.log_level)
    validate_live_mode_guardrails()
    initialize_database()

    scheduler = BotScheduler()
    scheduler.start()

    logger.info("Trading bot service running in {} mode", settings.bot_mode)
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(settings.service_heartbeat_seconds)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        scheduler.stop()


if __name__ == "__main__":
    run_service()
