"""Configuration and credential loading.

Everything the pipeline needs to run offline is derived from function arguments; this
module only handles the *optional* live path (Alpaca credentials) and the location of the
local Parquet cache. Credentials are read from the environment so nothing is hard-coded.

A tiny built-in ``.env`` reader is included so the project has no runtime dependency on
python-dotenv. It is best-effort: it never raises, and real environment variables always
win over values in the file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ENV_LOADED = False


def load_dotenv(path: str | os.PathLike[str] | None = None) -> None:
    """Populate ``os.environ`` from a ``.env`` file if present (real env vars take priority).

    Idempotent and dependency-free. Lines are ``KEY=VALUE``; ``#`` comments and blank lines
    are ignored. Surrounding quotes on values are stripped.
    """
    global _ENV_LOADED
    env_path = Path(path) if path is not None else Path.cwd() / ".env"
    if not env_path.is_file():
        _ENV_LOADED = True
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    _ENV_LOADED = True


def cache_dir() -> Path:
    """Directory for the per-symbol-year Parquet cache. Override with ``FINBERT_NEWS_CACHE``."""
    if not _ENV_LOADED:
        load_dotenv()
    root = os.getenv("FINBERT_NEWS_CACHE") or (Path.cwd() / "data_cache")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class AlpacaCredentials:
    """Alpaca API credentials for the optional live fetch path."""

    api_key: str
    secret_key: str
    data_url: str = "https://data.alpaca.markets"

    @classmethod
    def from_env(cls) -> AlpacaCredentials:
        """Build credentials from the environment (loading ``.env`` first if needed).

        Raises:
            RuntimeError: if ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` are not set. The
            offline pipeline never calls this, so a missing key only affects live fetches.
        """
        if not _ENV_LOADED:
            load_dotenv()
        key = os.getenv("ALPACA_API_KEY", "").strip()
        secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
        if not key or not secret:
            raise RuntimeError(
                "Alpaca credentials missing. Copy .env.example to .env and set "
                "ALPACA_API_KEY / ALPACA_SECRET_KEY, or run the offline pipeline on the "
                "bundled fixtures (no credentials required)."
            )
        data_url = os.getenv("ALPACA_DATA_URL", "").strip() or "https://data.alpaca.markets"
        return cls(api_key=key, secret_key=secret, data_url=data_url)
