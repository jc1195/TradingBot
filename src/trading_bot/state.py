from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RuntimeState:
    is_running: bool = True
    kill_switch: bool = False
    last_run_started_at: datetime | None = None
    last_run_finished_at: datetime | None = None
    consecutive_failures: int = 0
    notes: list[str] = field(default_factory=list)

    def mark_start(self) -> None:
        self.last_run_started_at = datetime.now(timezone.utc)

    def mark_finish(self) -> None:
        self.last_run_finished_at = datetime.now(timezone.utc)

    def add_note(self, note: str) -> None:
        self.notes.append(note)


runtime_state = RuntimeState()
