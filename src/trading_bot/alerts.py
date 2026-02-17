from __future__ import annotations

import requests

from .settings import settings


class AlertRouter:
    def __init__(self) -> None:
        self.webhook_url = settings.alert_webhook_url.strip()
        self.timeout = settings.alert_webhook_timeout_seconds
        self.allowed_event_types = {
            item.strip()
            for item in settings.alert_event_types_csv.split(",")
            if item.strip()
        }

    def should_send(self, event_type: str) -> bool:
        if not self.webhook_url:
            return False
        return event_type in self.allowed_event_types

    def send(self, event_type: str, message: str, metadata: dict) -> None:
        if not self.should_send(event_type):
            return

        payload = {
            "event_type": event_type,
            "message": message,
            "metadata": metadata,
        }
        requests.post(self.webhook_url, json=payload, timeout=self.timeout)
