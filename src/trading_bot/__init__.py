# ── Suppress curl_cffi KeyboardInterrupt callback noise ──────────
# yfinance uses curl_cffi for HTTP, and its CFFI C-callbacks can't
# handle KeyboardInterrupt cleanly — they print ugly tracebacks to
# stderr even though they're harmless.  We install a stderr filter
# to silently discard those messages.
import sys as _sys
import warnings as _warnings

_warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


class _CffiStderrFilter:
    """Wraps stderr to swallow curl_cffi callback exception noise.

    Duck-types the TextIO interface so loguru / logging / print all work.
    """

    def __init__(self, wrapped):  # type: ignore[no-untyped-def]
        self._wrapped = wrapped
        self._suppressing = False

    # ── Core write/flush ──
    def write(self, s: str) -> int:
        if "cffi callback" in s or "buffer_callback" in s:
            self._suppressing = True
            return len(s)
        if self._suppressing:
            if s.strip() == "" or "KeyboardInterrupt" in s:
                if "KeyboardInterrupt" in s:
                    self._suppressing = False
                return len(s)
            if not s.startswith((" ", "\t", "Traceback", "File")):
                self._suppressing = False
                return self._wrapped.write(s)
            return len(s)
        return self._wrapped.write(s)

    def flush(self) -> None:
        self._wrapped.flush()

    # ── Required TextIO attributes ──
    @property
    def name(self) -> str:
        return getattr(self._wrapped, "name", "<stderr>")

    def fileno(self) -> int:
        return self._wrapped.fileno()

    def isatty(self) -> bool:
        return self._wrapped.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self._wrapped, "encoding", "utf-8")

    @property
    def errors(self) -> str | None:
        return getattr(self._wrapped, "errors", None)

    @property
    def closed(self) -> bool:
        return getattr(self._wrapped, "closed", False)

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        pass  # never close stderr

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._wrapped, name)


# Install the filter on stderr
try:
    if not isinstance(_sys.stderr, _CffiStderrFilter):
        _sys.stderr = _CffiStderrFilter(_sys.stderr)  # type: ignore[assignment]
except Exception:
    pass  # Don't crash if stderr is weird (e.g. pytest capture)


__all__ = [
    "settings",
    "engine",
]
