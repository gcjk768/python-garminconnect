"""``garmin-monitor`` command line interface."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import logging
import sys
from collections.abc import Callable
from datetime import date, datetime, timedelta

from . import __version__
from .config import AppConfig, ConfigError, ProfileConfig, load_config
from .llm import LLMClient, LLMError, describe_backend, make_llm_client
from .storage import Storage
from .utils import parse_date

logger = logging.getLogger("garmin_health_monitor")


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def _config(args: argparse.Namespace) -> AppConfig:
    try:
        return load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(2)


def _profile(cfg: AppConfig, name: str | None) -> ProfileConfig:
    if name:
        return cfg.profile(name)
    if len(cfg.profiles) == 1:
        return cfg.profiles[0]
    print("Several profiles configured; pass --profile " + "|".join(p.name for p in cfg.profiles), file=sys.stderr)
    sys.exit(2)


def _day(value: str | None, profile: ProfileConfig, service) -> date:
    if not value or value == "today":
        return service.today(profile)
    if value == "yesterday":
        return service.today(profile) - timedelta(days=1)
    return parse_date(value)


def _service(cfg: AppConfig, llm: LLMClient | None = None, mfa: Callable[[], str] | None = None):
    from .service import MonitorService

    return MonitorService(cfg, Storage(cfg.database), llm=llm, mfa_prompt_factory=(lambda p: mfa) if mfa else None)


def _terminal_mfa() -> str:
    return input("Garmin MFA code: ").strip()


# ------------------------------------------------------------------ commands


def cmd_login(args: argparse.Namespace) -> int:
    cfg = _config(args)
    profile = _profile(cfg, args.profile)
    from .garmin_client import GarminSession

    if not profile.garmin.email:
        profile.garmin.email = input("Garmin email: ").strip()
    if not profile.garmin.password:
        profile.garmin.password = getpass.getpass("Garmin password: ")
    sess = GarminSession(profile, mfa_prompt=_terminal_mfa)
    api = sess.login()
    print(f"Logged in as {api.get_full_name()} — tokens saved to {profile.garmin.tokenstore}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config(args)
    from .scheduler import Scheduler
    from .service import MonitorService
    from .telegram_bot import HealthBot

    llm = make_llm_client(cfg)
    logger.info("LLM backend: %s", describe_backend(llm))
    if llm is not None and not llm.is_available():
        logger.warning("LLM backend is not available right now; rule-based fallbacks will be used until it is")
    elif llm is not None and hasattr(llm, "ensure_model"):
        try:
            llm.ensure_model()
        except LLMError as exc:
            logger.warning("LLM model check failed: %s", exc)

    storage = Storage(cfg.database)
    bot_holder: dict[str, HealthBot] = {}

    def mfa_factory(profile: ProfileConfig) -> Callable[[], str]:
        def prompt() -> str:
            bot = bot_holder.get("bot")
            if bot is None or getattr(bot, "loop", None) is None:
                return _terminal_mfa()
            from . import messages as _m

            ids = list(cfg.telegram.admin_chat_ids) or list(profile.telegram_chat_ids)
            asyncio.run_coroutine_threadsafe(bot.send_text(ids, _m.mfa_request(profile)), bot.loop)
            return bot.mfa.wait_for_code(profile.name, timeout=600)

        return prompt

    service = MonitorService(cfg, storage, llm=llm, mfa_prompt_factory=mfa_factory)
    bot = HealthBot(cfg, service)
    bot_holder["bot"] = bot
    Scheduler(cfg, service, bot).register()
    logger.info("Starting Telegram bot (%d profile(s))", len(cfg.profiles))
    bot.run()
    return 0


def cmd_poll(args: argparse.Namespace) -> int:
    cfg = _config(args)
    service = _service(cfg, make_llm_client(cfg) if not args.no_llm else None, mfa=_terminal_mfa)
    profiles = [_profile(cfg, args.profile)] if args.profile else cfg.profiles
    for p in profiles:
        day = _day(args.date, p, service)
        result = service.poll(p, day=day, light=not args.full)
        if result.error:
            print(f"[{p.name}] error: {result.error}")
            continue
        s = result.snapshot
        assert s is not None
        print(f"[{p.name}] {s.date_str}: steps={s.summary.total_steps} rhr={s.summary.resting_hr} hr_samples={len(s.hr)} errors={s.errors or 'none'}")
        for ep in result.new_episodes:
            print(f"  new episode: {ep.kind} {ep.start.isoformat()} {ep.duration_min}min peak {ep.peak_hr} base {ep.baseline_hr} conf {ep.confidence}")
        for ep in result.to_notify:
            print(f"  would alert: episode #{ep.id} ({ep.llm_assessment or 'unassessed'})")
        for a in result.alerts:
            print(f"  alert [{a.severity}] {a.title}: {a.body}")
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    cfg = _config(args)
    service = _service(cfg, None, mfa=_terminal_mfa)
    profiles = [_profile(cfg, args.profile)] if args.profile else cfg.profiles
    for p in profiles:
        n = service.backfill(p, args.days)
        print(f"[{p.name}] fetched {n} day(s)")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    cfg = _config(args)
    service = _service(cfg, None)
    p = _profile(cfg, args.profile)
    day = _day(args.date, p, service)
    snap = service.load_snapshot(p, day)
    if snap is None:
        print(f"No stored data for {p.name} on {day}; run `poll --date {day}` first")
        return 1
    from .palpitations import detect_episodes

    eps = detect_episodes(snap, p.palpitations, extra_sleep_windows=service._next_day_sleep_window(p, day))
    if not eps:
        print("No candidate episodes")
    for ep in eps:
        print(
            f"{ep.kind:9s} {ep.start.astimezone().strftime('%H:%M')} {ep.duration_min:5.1f} min "
            f"peak {ep.peak_hr:3d} base {ep.baseline_hr:5.1f} Δ{ep.delta_hr:+5.1f} jump {ep.max_jump_bpm:3d} "
            f"steps {ep.steps_in_window:4d} {ep.activity_level:9s} asleep={ep.asleep} conf={ep.confidence}"
        )
    return 0


def _send_or_print(cfg: AppConfig, chat_ids: list[int], text: str, send: bool) -> None:
    if not send:
        print(text)
        return
    from .telegram_bot import HealthBot

    bot = HealthBot(cfg, service=None)  # type: ignore[arg-type]

    async def _go() -> None:
        async with bot.app:
            await bot.send_text(chat_ids, text)

    asyncio.run(_go())
    print("sent")


def cmd_brief(args: argparse.Namespace) -> int:
    cfg = _config(args)
    service = _service(cfg, make_llm_client(cfg) if not args.no_llm else None, mfa=_terminal_mfa)
    p = _profile(cfg, args.profile)
    fn = {
        "morning": service.morning_brief_text,
        "evening": service.evening_summary_text,
        "weekly": service.weekly_review_text,
        "today": service.today_text,
    }[args.kind]
    _send_or_print(cfg, p.telegram_chat_ids, fn(p), args.send)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    cfg = _config(args)
    llm = make_llm_client(cfg)
    if llm is None:
        print("llm.backend is 'none'; nothing to analyse with")
        return 1
    service = _service(cfg, llm, mfa=_terminal_mfa)
    p = _profile(cfg, args.profile)
    _send_or_print(cfg, p.telegram_chat_ids, service.analyze_now(p), args.send)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = _config(args)
    service = _service(cfg, make_llm_client(cfg) if not args.no_llm else None)
    p = _profile(cfg, args.profile)
    files = service.doctor_report(p, args.days)
    print(files.summary)
    print(f"PDF: {files.pdf_path}\nCSV: {files.csv_path}")
    if args.send:
        from . import messages
        from .telegram_bot import HealthBot

        bot = HealthBot(cfg, service)

        async def _go() -> None:
            async with bot.app:
                await bot.send_document(p.telegram_chat_ids, files.pdf_path, messages.doctor_report_caption(p, args.days, files.n_episodes, files.n_symptoms))
                await bot.send_document(p.telegram_chat_ids, files.csv_path, "📄 CSV export of the same diary")

        asyncio.run(_go())
        print("sent")
    return 0


def cmd_test_telegram(args: argparse.Namespace) -> int:
    cfg = _config(args)
    ids = cfg.telegram.admin_chat_ids or [cid for p in cfg.profiles for cid in p.telegram_chat_ids]
    if not ids:
        print("No chat ids configured")
        return 1
    _send_or_print(cfg, ids, f"✅ Garmin Health Monitor {__version__} can reach this chat.", send=True)
    return 0


def cmd_test_llm(args: argparse.Namespace) -> int:
    cfg = _config(args)
    llm = make_llm_client(cfg)
    print(f"Backend: {describe_backend(llm)}")
    if llm is None:
        return 0
    if not llm.is_available():
        print("NOT available (is the CLI installed / logged in, or the Ollama server running?)")
        return 1
    out = llm.chat_json(
        "You are a test harness. Answer only with JSON.",
        "Return {\"ok\": true, \"note\": \"one short sentence\"}",
        {"type": "object", "properties": {"ok": {"type": "boolean"}, "note": {"type": "string"}}, "required": ["ok"]},
    )
    print(f"OK: {out}")
    return 0


def cmd_symptom(args: argparse.Namespace) -> int:
    cfg = _config(args)
    service = _service(cfg, make_llm_client(cfg))  # the LLM extracts symptoms/triggers from the note
    p = _profile(cfg, args.profile)
    when = datetime.fromisoformat(args.time) if args.time else None
    print(service.log_symptom(p, when, args.note or "", None))
    return 0


# ------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="garmin-monitor", description="Garmin Connect -> Telegram health monitor")
    parser.add_argument("--config", "-c", default=None, help="Path to config.yaml (default: $GHM_CONFIG or ./config.yaml)")
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--version", action="version", version=f"garmin-monitor {__version__}")
    sub = parser.add_subparsers(dest="command")

    s = sub.add_parser("login", help="Interactive Garmin login (asks for MFA) and token storage")
    s.add_argument("--profile", "-p")
    s.set_defaults(func=cmd_login)

    s = sub.add_parser("run", help="Run the Telegram bot and the scheduler (default)")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("poll", help="One polling cycle: fetch, detect, evaluate alerts (prints, does not send)")
    s.add_argument("--profile", "-p")
    s.add_argument("--date", "-d", help="YYYY-MM-DD | today | yesterday")
    s.add_argument("--full", action="store_true", help="fetch all endpoints, not just the intraday ones")
    s.add_argument("--no-llm", action="store_true")
    s.set_defaults(func=cmd_poll)

    s = sub.add_parser("backfill", help="Fetch history without sending alerts")
    s.add_argument("--profile", "-p")
    s.add_argument("--days", type=int, default=7)
    s.set_defaults(func=cmd_backfill)

    s = sub.add_parser("detect", help="Run the palpitation detector on stored data")
    s.add_argument("--profile", "-p")
    s.add_argument("--date", "-d", default="yesterday")
    s.set_defaults(func=cmd_detect)

    s = sub.add_parser("brief", help="Print (or send) a scheduled message now")
    s.add_argument("--profile", "-p")
    s.add_argument("--kind", "-k", choices=["morning", "evening", "weekly", "today"], default="evening")
    s.add_argument("--send", action="store_true")
    s.add_argument("--no-llm", action="store_true")
    s.set_defaults(func=cmd_brief)

    s = sub.add_parser("analyze", help="Run the coaching analysis now")
    s.add_argument("--profile", "-p")
    s.add_argument("--send", action="store_true")
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("report", help="Generate the doctor report (PDF + CSV)")
    s.add_argument("--profile", "-p")
    s.add_argument("--days", type=int, default=30)
    s.add_argument("--send", action="store_true")
    s.add_argument("--no-llm", action="store_true")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("symptom", help="Log a palpitation the person felt")
    s.add_argument("--profile", "-p")
    s.add_argument("--time", "-t", help="ISO time, e.g. 2026-09-28T14:30+08:00 (default now)")
    s.add_argument("--note", "-n", default="")
    s.set_defaults(func=cmd_symptom)

    s = sub.add_parser("test-telegram", help="Send a test message to the admin chat(s)")
    s.set_defaults(func=cmd_test_telegram)

    s = sub.add_parser("test-llm", help="Check the configured LLM backend (claude -p / Ollama)")
    s.set_defaults(func=cmd_test_llm)
    s = sub.add_parser("test-ollama", help="Alias of test-llm")
    s.set_defaults(func=cmd_test_llm)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = args.log_level
    if level is None:
        try:
            level = load_config(args.config).log_level
        except Exception:  # noqa: BLE001
            level = "INFO"
    _setup_logging(level)
    if not args.command:
        args.func = cmd_run
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
