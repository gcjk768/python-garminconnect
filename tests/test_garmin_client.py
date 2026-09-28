"""GarminSession tests with the garminconnect.Garmin class replaced by a fake."""

from __future__ import annotations

from datetime import date

import pytest
from garminconnect import (
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

from garmin_health_monitor import garmin_client
from garmin_health_monitor.garmin_client import (
    FULL_ENDPOINTS,
    LIGHT_ENDPOINTS,
    GarminAuthRequired,
    GarminSession,
    GarminUnavailable,
)
from garmin_health_monitor.normalize import RAW_HEART_RATES, RAW_SLEEP, RAW_STEPS, RAW_SUMMARY
from tests.conftest import make_profile

DAY = date(2026, 9, 27)


class FakeGarmin:
    instances: list[FakeGarmin] = []
    login_error: Exception | None = None
    fail_endpoints: dict[str, Exception] = {}
    auth_fail_once: set[str] = set()

    def __init__(self, email=None, password=None, is_cn=False, prompt_mfa=None):
        self.email, self.password, self.is_cn, self.prompt_mfa = email, password, is_cn, prompt_mfa
        self.logins = 0
        self.calls: list[str] = []
        FakeGarmin.instances.append(self)

    def login(self, tokenstore=None):
        self.logins += 1
        self.tokenstore = tokenstore
        if FakeGarmin.login_error:
            raise FakeGarmin.login_error
        return None, None

    def get_full_name(self):
        return "Test Person"

    def _ep(self, name, value):
        self.calls.append(name)
        if name in FakeGarmin.auth_fail_once:
            FakeGarmin.auth_fail_once.discard(name)
            raise GarminConnectAuthenticationError("expired")
        if name in FakeGarmin.fail_endpoints:
            raise FakeGarmin.fail_endpoints[name]
        return value

    def get_user_summary(self, d):
        return self._ep(RAW_SUMMARY, {"totalSteps": 1})

    def get_heart_rates(self, d):
        return self._ep(RAW_HEART_RATES, {"heartRateValues": []})

    def get_steps_data(self, d):
        return self._ep(RAW_STEPS, [])

    def get_stress_data(self, d):
        return self._ep("stress", {})

    def get_activities_by_date(self, a, b):
        return self._ep("activities", [])

    def get_sleep_data(self, d):
        return self._ep(RAW_SLEEP, {"dailySleepDTO": {}})

    def get_hrv_data(self, d):
        return self._ep("hrv", None)

    def get_body_battery_events(self, d):
        return self._ep("body_battery_events", [])

    def get_respiration_data(self, d):
        return self._ep("respiration", {})

    def get_spo2_data(self, d):
        return self._ep("spo2", {})

    def get_training_readiness(self, d):
        return self._ep("training_readiness", [])

    def get_device_last_used(self):
        return self._ep("device_last_used", {"lastUsedDeviceName": "Venu"})


@pytest.fixture(autouse=True)
def fake_garmin(monkeypatch, tmp_path):
    FakeGarmin.instances = []
    FakeGarmin.login_error = None
    FakeGarmin.fail_endpoints = {}
    FakeGarmin.auth_fail_once = set()
    monkeypatch.setattr(garmin_client, "Garmin", FakeGarmin)
    yield FakeGarmin


def _session(tmp_path, mfa=None):
    p = make_profile()
    p.garmin.tokenstore = str(tmp_path / "tokens")
    return GarminSession(p, mfa_prompt=mfa)


def test_login_creates_tokenstore_and_passes_mfa(tmp_path):
    prompt = lambda: "123456"  # noqa: E731
    sess = _session(tmp_path, mfa=prompt)
    api = sess.login()
    assert api.prompt_mfa is prompt
    assert api.tokenstore == str(tmp_path / "tokens")
    assert (tmp_path / "tokens").is_dir()
    assert sess.whoami() == "Test Person"
    assert sess.has_tokens() is False  # directory exists but is empty
    (tmp_path / "tokens" / "oauth1_token.json").write_text("{}")
    assert sess.has_tokens() is True


def test_login_errors_are_translated(tmp_path):
    FakeGarmin.login_error = GarminConnectAuthenticationError("bad password")
    with pytest.raises(GarminAuthRequired, match="garmin-monitor login"):
        _session(tmp_path).login()
    FakeGarmin.login_error = GarminConnectTooManyRequestsError("429")
    with pytest.raises(GarminUnavailable):
        _session(tmp_path).login()
    FakeGarmin.login_error = GarminConnectConnectionError("down")
    with pytest.raises(GarminUnavailable):
        _session(tmp_path).login()


def test_fetch_day_light_vs_full(tmp_path):
    sess = _session(tmp_path)
    raw, errors = sess.fetch_day(DAY, light=True)
    assert set(raw) <= set(LIGHT_ENDPOINTS) and RAW_SUMMARY in raw and errors == {}
    assert RAW_SLEEP not in raw
    raw, errors = sess.fetch_day(DAY, light=False)
    assert RAW_SLEEP in raw and "device_last_used" in raw
    assert "hrv" not in raw  # None results are simply absent
    assert set(FakeGarmin.instances[0].calls) >= set(FULL_ENDPOINTS)


def test_one_failing_endpoint_does_not_sink_the_day(tmp_path):
    FakeGarmin.fail_endpoints = {RAW_STEPS: GarminConnectConnectionError("boom"), RAW_SLEEP: ValueError("weird")}
    sess = _session(tmp_path)
    raw, errors = sess.fetch_day(DAY, light=False)
    assert RAW_STEPS not in raw and RAW_SUMMARY in raw
    assert "connection" in errors[RAW_STEPS] and "ValueError" in errors[RAW_SLEEP]


def test_rate_limit_raises_unavailable(tmp_path):
    FakeGarmin.fail_endpoints = {RAW_SUMMARY: GarminConnectTooManyRequestsError("429")}
    with pytest.raises(GarminUnavailable):
        _session(tmp_path).fetch_day(DAY, light=True)


def test_auth_error_triggers_single_relogin(tmp_path):
    FakeGarmin.auth_fail_once = {RAW_HEART_RATES}
    sess = _session(tmp_path)
    raw, errors = sess.fetch_day(DAY, light=True)
    assert RAW_HEART_RATES in raw and errors == {}
    assert len(FakeGarmin.instances) == 2  # re-login created a fresh client


def test_persistent_auth_error_raises(tmp_path):
    FakeGarmin.fail_endpoints = {RAW_SUMMARY: GarminConnectAuthenticationError("nope")}
    with pytest.raises(GarminAuthRequired):
        _session(tmp_path).fetch_day(DAY, light=True)


def test_all_endpoints_failing_is_unavailable(tmp_path):
    FakeGarmin.fail_endpoints = {name: GarminConnectConnectionError("x") for name in LIGHT_ENDPOINTS}
    with pytest.raises(GarminUnavailable):
        _session(tmp_path).fetch_day(DAY, light=True)
