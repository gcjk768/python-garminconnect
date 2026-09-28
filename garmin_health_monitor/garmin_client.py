"""Thin wrapper around :mod:`garminconnect` with token persistence and per-endpoint error isolation."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

from .config import ProfileConfig
from .normalize import (
    RAW_ACTIVITIES,
    RAW_BB_EVENTS,
    RAW_DEVICE,
    RAW_HEART_RATES,
    RAW_HRV,
    RAW_READINESS,
    RAW_RESPIRATION,
    RAW_SLEEP,
    RAW_SPO2,
    RAW_STEPS,
    RAW_STRESS,
    RAW_SUMMARY,
)

logger = logging.getLogger(__name__)

# Endpoints fetched on every poll (cheap, change during the day)
LIGHT_ENDPOINTS = (RAW_SUMMARY, RAW_HEART_RATES, RAW_STEPS, RAW_STRESS, RAW_ACTIVITIES)
# Extra endpoints fetched for the morning/evening jobs and history backfill
FULL_ENDPOINTS = LIGHT_ENDPOINTS + (
    RAW_SLEEP,
    RAW_HRV,
    RAW_BB_EVENTS,
    RAW_RESPIRATION,
    RAW_SPO2,
    RAW_READINESS,
    RAW_DEVICE,
)


class GarminUnavailable(RuntimeError):
    """Raised when Garmin cannot be reached / is rate limiting; the caller should skip this cycle."""


class GarminAuthRequired(RuntimeError):
    """Raised when stored tokens are invalid and no credentials / MFA are available."""


class GarminSession:
    """One authenticated Garmin Connect session per profile.

    ``mfa_prompt`` is called (from the worker thread) when Garmin asks for a
    one-time code.  The CLI passes ``input``; the bot passes a callable that
    asks over Telegram and blocks until ``/mfa <code>`` arrives.
    """

    def __init__(self, profile: ProfileConfig, mfa_prompt: Callable[[], str] | None = None):
        self.profile = profile
        self.mfa_prompt = mfa_prompt
        self._lock = threading.RLock()
        self._api: Garmin | None = None

    # -- auth ---------------------------------------------------------------

    @property
    def api(self) -> Garmin:
        if self._api is None:
            self.login()
        assert self._api is not None
        return self._api

    def has_tokens(self) -> bool:
        p = Path(self.profile.garmin.tokenstore).expanduser()
        return p.exists() and any(p.iterdir()) if p.is_dir() else p.exists()

    def login(self) -> Garmin:
        with self._lock:
            g = self.profile.garmin
            Path(g.tokenstore).expanduser().mkdir(parents=True, exist_ok=True)
            api = Garmin(
                email=g.email,
                password=g.password,
                is_cn=g.is_cn,
                prompt_mfa=self.mfa_prompt,
            )
            try:
                api.login(g.tokenstore)
            except GarminConnectTooManyRequestsError as exc:
                raise GarminUnavailable(f"Garmin rate limit during login: {exc}") from exc
            except GarminConnectAuthenticationError as exc:
                raise GarminAuthRequired(
                    f"Garmin login failed for profile {self.profile.name}: {exc}. "
                    "Run `garmin-monitor login --profile <name>` (or check email/password)."
                ) from exc
            except GarminConnectConnectionError as exc:
                raise GarminUnavailable(f"Garmin unreachable during login: {exc}") from exc
            self._api = api
            logger.info("Garmin login OK for profile %s (%s)", self.profile.name, api.get_full_name() or "?")
            return api

    def logout(self) -> None:
        with self._lock:
            self._api = None

    # -- fetch --------------------------------------------------------------

    def _call(self, name: str, fn: Callable[[], Any], errors: dict[str, str], retry_auth: bool = True) -> Any:
        try:
            return fn()
        except GarminConnectTooManyRequestsError as exc:
            raise GarminUnavailable(f"Garmin rate limit on {name}: {exc}") from exc
        except GarminConnectAuthenticationError as exc:
            if retry_auth:
                logger.warning("Auth error on %s for %s; re-logging in", name, self.profile.name)
                self.logout()
                self.login()
                return self._call(name, fn, errors, retry_auth=False)
            raise GarminAuthRequired(str(exc)) from exc
        except GarminConnectConnectionError as exc:
            errors[name] = f"connection: {exc}"
            logger.warning("%s: %s failed: %s", self.profile.name, name, exc)
        except Exception as exc:  # noqa: BLE001 - one bad endpoint must not sink the day
            errors[name] = f"{type(exc).__name__}: {exc}"
            logger.warning("%s: %s failed: %s", self.profile.name, name, exc)
        return None

    def fetch_day(self, day: date, light: bool = False) -> tuple[dict[str, Any], dict[str, str]]:
        """Fetch the raw payloads for ``day``.

        Returns ``(raw, errors)`` where ``raw`` is keyed by the ``RAW_*`` names in
        :mod:`normalize`.  ``light=True`` fetches only the endpoints that change
        during the day (used by the frequent poll).
        """
        api = self.api
        d = day.isoformat()
        raw: dict[str, Any] = {}
        errors: dict[str, str] = {}
        fetchers: dict[str, Callable[[], Any]] = {
            RAW_SUMMARY: lambda: api.get_user_summary(d),
            RAW_HEART_RATES: lambda: api.get_heart_rates(d),
            RAW_STEPS: lambda: api.get_steps_data(d),
            RAW_STRESS: lambda: api.get_stress_data(d),
            RAW_ACTIVITIES: lambda: api.get_activities_by_date(d, d),
            RAW_SLEEP: lambda: api.get_sleep_data(d),
            RAW_HRV: lambda: api.get_hrv_data(d),
            RAW_BB_EVENTS: lambda: api.get_body_battery_events(d),
            RAW_RESPIRATION: lambda: api.get_respiration_data(d),
            RAW_SPO2: lambda: api.get_spo2_data(d),
            RAW_READINESS: lambda: api.get_training_readiness(d),
            RAW_DEVICE: lambda: api.get_device_last_used(),
        }
        for name in LIGHT_ENDPOINTS if light else FULL_ENDPOINTS:
            value = self._call(name, fetchers[name], errors)
            if value is not None:
                raw[name] = value
        # A day with *no* data at all usually means the API tier is refusing us.
        if not raw and errors:
            raise GarminUnavailable(f"All Garmin endpoints failed for {self.profile.name} on {d}: {errors}")
        return raw, errors

    def fetch_device(self) -> dict[str, Any] | None:
        errors: dict[str, str] = {}
        return self._call(RAW_DEVICE, lambda: self.api.get_device_last_used(), errors)

    def whoami(self) -> str:
        return self.api.get_full_name() or self.profile.name
