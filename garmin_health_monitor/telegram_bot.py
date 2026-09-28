"""Telegram front end: interactive commands, alert delivery and the MFA relay.

Built on python-telegram-bot v22 (async).  Every handler:

* ignores chats that ``AppConfig.is_authorised`` does not know (logged at INFO),
* resolves which profile the chat is talking about (``/today Dad`` when a chat
  can see several people),
* runs the blocking :class:`~garmin_health_monitor.service.MonitorService` call in
  a worker thread with :func:`asyncio.to_thread`,
* and replies with a short, calm error line instead of a stack trace when
  something goes wrong.

Outgoing helpers (:meth:`HealthBot.send_text`, :meth:`HealthBot.send_photo`,
:meth:`HealthBot.send_document`, :meth:`HealthBot.notify_episode`) are used by
the scheduler and the CLI.  They chunk long texts, fall back to plain text when
Telegram rejects the HTML, and never let one failing chat stop the others.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
import threading
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    Message,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest, Forbidden, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from . import messages
from .config import AppConfig, ProfileConfig
from .models import Episode
from .utils import get_tz, now_utc, parse_date, to_local, to_utc

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .service import MonitorService

logger = logging.getLogger(__name__)

#: Telegram allows 4096 characters per text message; keep headroom for tags we re-open.
MAX_TEXT_LEN = 4000
#: Telegram allows 1024 characters in a photo/document caption.
MAX_CAPTION_LEN = 1024
#: Callback-query toast text limit.
MAX_CALLBACK_ANSWER_LEN = 200

FELT_CALLBACK_RE = re.compile(r"^felt:(\d+):(yes|no)$")
_TIME_ARG_RE = re.compile(r"^(\d{1,2})[:.h](\d{2})$|^(\d{2})(\d{2})$")
_TAG_RE = re.compile(r"<(/?)(b|i|u|s|code|pre|strong|em)>", re.IGNORECASE)

DEFAULT_EPISODE_DAYS = 7
MAX_EPISODE_DAYS = 90
DEFAULT_REPORT_DAYS = 30
MAX_REPORT_DAYS = 365

_BOT_COMMANDS: list[tuple[str, str]] = [
    ("today", "Live snapshot: steps, body battery, heart rate"),
    ("yesterday", "Yesterday's summary"),
    ("sleep", "Last night's sleep"),
    ("hr", "Heart-rate chart (optional YYYY-MM-DD)"),
    ("steps", "Steps today vs goal"),
    ("episodes", "Palpitation diary (optional days)"),
    ("palp", "Log a palpitation: /palp [HH:MM] [note]"),
    ("note", "Add a note to the diary"),
    ("report", "Doctor report PDF + CSV (optional days)"),
    ("analyze", "Ask the local model for coaching now"),
    ("profiles", "Who this chat can see"),
    ("status", "Service health"),
    ("help", "Command list"),
]


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without Telegram)
# ---------------------------------------------------------------------------


def _open_tags(fragment: str) -> list[str]:
    """Return the HTML tags still open at the end of ``fragment`` (outermost first)."""
    stack: list[str] = []
    for m in _TAG_RE.finditer(fragment):
        closing, name = m.group(1) == "/", m.group(2).lower()
        if closing:
            if name in stack:
                # pop up to and including the matching tag
                while stack:
                    top = stack.pop()
                    if top == name:
                        break
        else:
            stack.append(name)
    return stack


def chunk_text(text: str, limit: int = MAX_TEXT_LEN) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` characters.

    Splits on paragraph/line boundaries when possible, then on spaces, then
    hard.  Simple HTML tags (``<b>``, ``<i>``, ``<code>``, ``<pre>``...) that
    would straddle a boundary are closed at the end of one chunk and re-opened
    at the start of the next so Telegram still accepts each piece.
    """
    text = text or ""
    if len(text) <= limit:
        return [text] if text else []
    reserve = min(64, max(0, limit // 4))  # room for the closing tags we may append
    chunks: list[str] = []
    carry: list[str] = []  # tags to re-open at the start of the next chunk
    rest = text
    while rest:
        prefix = "".join(f"<{t}>" for t in carry)
        budget = max(1, limit - len(prefix) - reserve)
        if len(rest) <= budget:
            body, rest = rest, ""
        else:
            window = rest[:budget]
            cut = -1
            for sep in ("\n\n", "\n", " "):
                pos = window.rfind(sep)
                if pos > budget // 4:  # avoid pathological tiny chunks
                    cut = pos
                    break
            if cut <= 0:
                cut = budget
            body = rest[:cut]
            rest = rest[cut:].lstrip("\n ")
        open_tags = _open_tags(prefix + body)
        suffix = "".join(f"</{t}>" for t in reversed(open_tags))
        chunk = prefix + body + suffix
        if chunk.strip():
            chunks.append(chunk)
        carry = open_tags
    return chunks


def parse_symptom_args(
    args: Sequence[str],
    tz: str,
    now: datetime | None = None,
) -> tuple[datetime | None, str]:
    """Parse ``/palp [HH:MM] [note...]`` arguments.

    Returns ``(event_time_utc, note)``.  ``event_time_utc`` is ``None`` when no
    time was given (meaning "now").  A time later than *now* in the profile's
    zone is taken to mean yesterday.  Accepts ``HH:MM``, ``HH.MM`` and ``HHMM``.

    Raises :class:`ValueError` when the first argument looks like a time but is
    out of range (e.g. ``25:70``).
    """
    args = [a for a in args if a is not None]
    if not args:
        return None, ""
    first = args[0].strip()
    m = _TIME_ARG_RE.match(first)
    if not m:
        return None, " ".join(a.strip() for a in args).strip()
    hour = int(m.group(1) or m.group(3))
    minute = int(m.group(2) or m.group(4))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"{first!r} is not a valid time; use HH:MM (24-hour)")
    zone = get_tz(tz)
    now_local = to_local(now or now_utc(), zone)
    event_local = datetime.combine(now_local.date(), time(hour, minute), tzinfo=zone)
    if event_local > now_local + timedelta(minutes=2):
        event_local -= timedelta(days=1)
    note = " ".join(a.strip() for a in args[1:]).strip()
    return to_utc(event_local), note


def _parse_days(args: Sequence[str], default: int, cap: int) -> int:
    """First integer argument clamped to ``1..cap``; ``default`` when absent/invalid."""
    for a in args:
        try:
            value = int(str(a).strip())
        except (TypeError, ValueError):
            continue
        return max(1, min(value, cap))
    return default


def build_felt_keyboard(episode_id: int) -> InlineKeyboardMarkup:
    """Inline "I felt it" / "Didn't notice" buttons for an episode alert."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("❤️ I felt it", callback_data=f"felt:{int(episode_id)}:yes"),
                InlineKeyboardButton(
                    "🙂 Didn't notice", callback_data=f"felt:{int(episode_id)}:no"
                ),
            ]
        ]
    )


def _is_parse_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "parse" in msg or "entit" in msg or "tag" in msg


def _plain(text: str) -> str:
    """Strip the simple HTML tags we use so a fallback message still reads well."""
    return re.sub(r"</?(b|i|u|s|code|pre|strong|em|a)(\s[^>]*)?>", "", text or "")


# ---------------------------------------------------------------------------
# MFA relay
# ---------------------------------------------------------------------------


class MfaBroker:
    """Hands Garmin one-time codes from ``/mfa <code>`` to the worker thread that is logging in.

    The Garmin login runs in a worker thread (:func:`asyncio.to_thread`); when
    Garmin asks for a code that thread calls :meth:`wait_for_code`, which blocks
    until an admin sends ``/mfa 123456`` in Telegram (handled on the event loop,
    calling :meth:`submit`) or the timeout passes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._codes: dict[str, str] = {}
        self._names: dict[str, str] = {}  # key -> display name

    @staticmethod
    def _key(profile_name: str) -> str:
        return str(profile_name).strip().lower()

    def pending(self) -> list[str]:
        """Profile names currently waiting for a code (insertion order)."""
        with self._lock:
            return [self._names[k] for k in self._events]

    def wait_for_code(self, profile_name: str, timeout: float = 600) -> str:
        """Block (worker thread) until a code for ``profile_name`` arrives.

        Raises :class:`TimeoutError` when nothing arrives within ``timeout`` seconds.
        """
        key = self._key(profile_name)
        with self._lock:
            event = threading.Event()
            self._events[key] = event
            self._names[key] = profile_name
            self._codes.pop(key, None)
        logger.info("Waiting up to %.0fs for an MFA code for %s", timeout, profile_name)
        got = event.wait(timeout)
        with self._lock:
            self._events.pop(key, None)
            self._names.pop(key, None)
            code = self._codes.pop(key, None)
        if not got or code is None:
            raise TimeoutError(f"No MFA code received for {profile_name} within {timeout:.0f}s")
        return code

    def submit(self, profile_name: str | None, code: str) -> bool:
        """Deliver ``code``; returns ``False`` when nothing is waiting for it.

        When only one profile is pending, any ``profile_name`` (including ``None``
        or an unknown name) is accepted for it.
        """
        code = str(code or "").strip()
        if not code:
            return False
        with self._lock:
            key = self._key(profile_name) if profile_name else ""
            if key not in self._events:
                if len(self._events) == 1:
                    key = next(iter(self._events))
                else:
                    return False
            self._codes[key] = code
            self._events[key].set()
            name = self._names.get(key, key)
        logger.info("MFA code received for %s", name)
        return True


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Ctx:
    """Everything a command handler needs after authorisation and profile resolution."""

    chat_id: int
    message: Message
    profile: ProfileConfig
    args: list[str]
    is_admin: bool


class HealthBot:
    """python-telegram-bot application wrapper: commands, alerts, files."""

    def __init__(self, config: AppConfig, service: MonitorService | None):
        self.config = config
        self.service = service
        self.mfa = MfaBroker()
        #: Running event loop, set once the application starts (used by worker threads
        #: through ``asyncio.run_coroutine_threadsafe``).
        self.loop: asyncio.AbstractEventLoop | None = None
        self.app: Application = (
            ApplicationBuilder().token(config.telegram.bot_token).post_init(self._post_init).build()
        )
        self._register_handlers()

    # ------------------------------------------------------------------ setup

    def _register_handlers(self) -> None:
        app = self.app
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("profiles", self.cmd_profiles))
        app.add_handler(CommandHandler("today", self.cmd_today))
        app.add_handler(CommandHandler("yesterday", self.cmd_yesterday))
        app.add_handler(CommandHandler("sleep", self.cmd_sleep))
        app.add_handler(CommandHandler("steps", self.cmd_steps))
        app.add_handler(CommandHandler("hr", self.cmd_hr))
        app.add_handler(CommandHandler("episodes", self.cmd_episodes))
        app.add_handler(CommandHandler("palp", self.cmd_palp))
        app.add_handler(CommandHandler("note", self.cmd_note))
        app.add_handler(CommandHandler("report", self.cmd_report))
        app.add_handler(CommandHandler("analyze", self.cmd_analyze))
        app.add_handler(CommandHandler("analyse", self.cmd_analyze))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("mfa", self.cmd_mfa))
        app.add_handler(CallbackQueryHandler(self.cb_felt, pattern=FELT_CALLBACK_RE))
        app.add_handler(MessageHandler(filters.COMMAND, self.cmd_unknown))
        app.add_error_handler(self.on_error)

    async def _post_init(self, app: Application) -> None:
        """Runs inside ``run_polling`` once the application is initialised."""
        self.loop = asyncio.get_running_loop()
        try:
            await app.bot.set_my_commands([BotCommand(c, d) for c, d in _BOT_COMMANDS])
        except TelegramError as exc:
            logger.warning("Could not publish the command menu: %s", exc)

    def run(self) -> None:
        """Block and poll Telegram until interrupted (the scheduler registers jobs before this)."""
        self.app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("Unhandled error in Telegram handler", exc_info=context.error)

    # ---------------------------------------------------------- authorisation

    def _chat_id(self, update: Update) -> int | None:
        chat = getattr(update, "effective_chat", None)
        cid = getattr(chat, "id", None)
        return cid if isinstance(cid, int) else None

    def _authorised_chat(self, update: Update) -> int | None:
        """Return the chat id when the chat may talk to the bot, else ``None`` (logged)."""
        chat_id = self._chat_id(update)
        if chat_id is None:
            return None
        if not self.config.is_authorised(chat_id):
            user = getattr(getattr(update, "effective_user", None), "id", None)
            logger.info("Ignoring update from unauthorised chat %s (user %s)", chat_id, user)
            return None
        return chat_id

    def _is_admin(self, chat_id: int) -> bool:
        return chat_id in self.config.telegram.admin_chat_ids

    @staticmethod
    def _message(update: Update) -> Message | None:
        msg = getattr(update, "effective_message", None)
        return msg if msg is not None else getattr(update, "message", None)

    @staticmethod
    def _args(context: ContextTypes.DEFAULT_TYPE) -> list[str]:
        args = getattr(context, "args", None)
        if not args:
            return []
        return [str(a) for a in args if a is not None and str(a).strip()]

    def _resolve_profile(
        self, chat_id: int, args: Sequence[str]
    ) -> tuple[ProfileConfig | None, list[str], str | None]:
        """Pick the profile a command refers to.

        Returns ``(profile, remaining_args, error_text)``.  The first argument may
        name a profile (case-insensitive name or slug); otherwise the chat must
        see exactly one profile.
        """
        profiles = self.config.profiles_for_chat(chat_id)
        args = list(args)
        if not profiles:
            return None, args, "This chat is not linked to anyone yet."
        if args:
            first = args[0].strip().lstrip("@").lower()
            for p in profiles:
                if p.name.lower() == first or p.slug == first:
                    return p, args[1:], None
        if len(profiles) == 1:
            return profiles[0], args, None
        names = ", ".join(messages.esc(p.name) for p in profiles)
        return (
            None,
            args,
            f"Which person? Add the name first, e.g. <code>/today {messages.esc(profiles[0].name)}</code>.\n"
            f"This chat can see: {names}",
        )

    async def _begin(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> _Ctx | None:
        """Authorise, locate the message and resolve the profile; reply and return None on failure."""
        chat_id = self._authorised_chat(update)
        message = self._message(update)
        if chat_id is None or message is None:
            return None
        if self.service is None:
            await self._reply(message, "The monitor service is not running; try again later.")
            return None
        profile, args, err = self._resolve_profile(chat_id, self._args(context))
        if profile is None:
            await self._reply(message, err or "No profile available.")
            return None
        return _Ctx(chat_id, message, profile, args, self._is_admin(chat_id))

    async def _run(self, ctx: _Ctx, what: str, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a blocking service call in a thread; on failure reply a short line and return None."""
        try:
            return await asyncio.to_thread(fn, *args)
        except Exception as exc:  # noqa: BLE001 - user-facing boundary
            logger.exception("%s: %s failed", ctx.profile.name, what)
            await self._reply(
                ctx.message,
                f"Sorry, I could not {messages.esc(what)} right now ({messages.esc(type(exc).__name__)}). "
                "Please try again in a few minutes.",
            )
            return None

    # ----------------------------------------------------------- outgoing I/O

    async def _send_chunks(
        self,
        send: Callable[..., Awaitable[Any]],
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
        **kwargs: Any,
    ) -> None:
        """Send ``text`` through ``send(text=..., **kwargs)`` in <= 4000-char pieces.

        Falls back to plain text when Telegram rejects the HTML.  ``reply_markup``
        goes on the last piece only.
        """
        chunks = chunk_text(text, MAX_TEXT_LEN) or ["n/a"]
        for i, chunk in enumerate(chunks):
            markup = reply_markup if i == len(chunks) - 1 else None
            try:
                await send(text=chunk, parse_mode=ParseMode.HTML, reply_markup=markup, **kwargs)
            except BadRequest as exc:
                if not _is_parse_error(exc):
                    raise
                logger.warning("Telegram rejected HTML (%s); resending as plain text", exc)
                await send(text=_plain(chunk), parse_mode=None, reply_markup=markup, **kwargs)

    async def _reply(
        self, message: Message, text: str, reply_markup: InlineKeyboardMarkup | None = None
    ) -> None:
        try:
            await self._send_chunks(message.reply_text, text, reply_markup)
        except TelegramError as exc:
            logger.error("Reply failed: %s", exc)

    def _thread_kwargs(self, chat_id: int, profile: ProfileConfig | None) -> dict[str, Any]:
        thread = self.config.thread_for(chat_id, profile)
        return {"message_thread_id": thread} if thread is not None else {}

    @staticmethod
    def _chat_list(chat_ids: Iterable[int] | int | None) -> list[int]:
        if chat_ids is None:
            return []
        if isinstance(chat_ids, int):
            return [chat_ids]
        out: list[int] = []
        for cid in chat_ids:
            try:
                value = int(cid)
            except (TypeError, ValueError):
                continue
            if value not in out:
                out.append(value)
        return out

    async def send_text(
        self,
        chat_ids: Iterable[int] | int | None,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
        profile: ProfileConfig | None = None,
    ) -> None:
        """Send an HTML message to every chat; long texts are chunked; failures are logged per chat."""
        for cid in self._chat_list(chat_ids):
            try:
                await self._send_chunks(
                    functools.partial(self.app.bot.send_message, chat_id=cid),
                    text,
                    reply_markup,
                    **self._thread_kwargs(cid, profile),
                )
            except Forbidden as exc:
                logger.warning("Chat %s blocked the bot or is unreachable: %s", cid, exc)
            except TelegramError as exc:
                logger.error("send_text to %s failed: %s", cid, exc)
            except Exception:  # noqa: BLE001
                logger.exception("send_text to %s failed", cid)

    async def _send_photo_one(
        self,
        chat_id: int,
        png: bytes,
        caption: str,
        reply_markup: InlineKeyboardMarkup | None,
        profile: ProfileConfig | None,
    ) -> None:
        extra = self._thread_kwargs(chat_id, profile)
        caption = caption or ""
        if len(caption) <= MAX_CAPTION_LEN:
            try:
                await self.app.bot.send_photo(
                    chat_id=chat_id,
                    photo=InputFile(png, filename="chart.png"),
                    caption=caption or None,
                    parse_mode=ParseMode.HTML if caption else None,
                    reply_markup=reply_markup,
                    **extra,
                )
                return
            except BadRequest as exc:
                if not _is_parse_error(exc):
                    raise
                logger.warning("Telegram rejected caption HTML (%s); resending plain", exc)
                await self.app.bot.send_photo(
                    chat_id=chat_id,
                    photo=InputFile(png, filename="chart.png"),
                    caption=_plain(caption) or None,
                    reply_markup=reply_markup,
                    **extra,
                )
                return
        # Caption too long: photo first, then the text (with the buttons) as a message.
        await self.app.bot.send_photo(
            chat_id=chat_id, photo=InputFile(png, filename="chart.png"), **extra
        )
        await self._send_chunks(
            functools.partial(self.app.bot.send_message, chat_id=chat_id),
            caption,
            reply_markup,
            **extra,
        )

    async def send_photo(
        self,
        chat_ids: Iterable[int] | int | None,
        png: bytes,
        caption: str,
        reply_markup: InlineKeyboardMarkup | None = None,
        profile: ProfileConfig | None = None,
    ) -> None:
        """Send a PNG with an HTML caption to every chat (falls back to text when ``png`` is empty)."""
        if not png:
            await self.send_text(chat_ids, caption, reply_markup, profile)
            return
        for cid in self._chat_list(chat_ids):
            try:
                await self._send_photo_one(cid, png, caption, reply_markup, profile)
            except Forbidden as exc:
                logger.warning("Chat %s blocked the bot or is unreachable: %s", cid, exc)
            except TelegramError as exc:
                logger.error("send_photo to %s failed: %s", cid, exc)
            except Exception:  # noqa: BLE001
                logger.exception("send_photo to %s failed", cid)

    async def send_document(
        self,
        chat_ids: Iterable[int] | int | None,
        path: str | Path,
        caption: str,
        profile: ProfileConfig | None = None,
    ) -> None:
        """Send a file (PDF/CSV) with an HTML caption to every chat."""
        p = Path(path)
        try:
            data = p.read_bytes()
        except OSError as exc:
            logger.error("Cannot read %s: %s", p, exc)
            await self.send_text(
                chat_ids,
                f"Sorry, the file {messages.esc(p.name)} could not be read.",
                None,
                profile,
            )
            return
        caption = caption or ""
        if len(caption) > MAX_CAPTION_LEN:
            caption = caption[: MAX_CAPTION_LEN - 1] + "…"
        for cid in self._chat_list(chat_ids):
            extra = self._thread_kwargs(cid, profile)
            try:
                try:
                    await self.app.bot.send_document(
                        chat_id=cid,
                        document=InputFile(data, filename=p.name),
                        caption=caption or None,
                        parse_mode=ParseMode.HTML if caption else None,
                        **extra,
                    )
                except BadRequest as exc:
                    if not _is_parse_error(exc):
                        raise
                    await self.app.bot.send_document(
                        chat_id=cid,
                        document=InputFile(data, filename=p.name),
                        caption=_plain(caption) or None,
                        **extra,
                    )
            except Forbidden as exc:
                logger.warning("Chat %s blocked the bot or is unreachable: %s", cid, exc)
            except TelegramError as exc:
                logger.error("send_document to %s failed: %s", cid, exc)
            except Exception:  # noqa: BLE001
                logger.exception("send_document to %s failed", cid)

    async def notify_episode(
        self, profile: ProfileConfig, episode: Episode, text: str, png: bytes | None
    ) -> None:
        """Deliver an episode alert (chart + caption or text) with the felt/not-felt buttons."""
        markup = build_felt_keyboard(episode.id) if episode.id is not None else None
        chat_ids = list(profile.telegram_chat_ids) or list(self.config.telegram.admin_chat_ids)
        if not chat_ids:
            logger.warning("%s: no chat to notify about episode %s", profile.name, episode.id)
            return
        if png:
            await self.send_photo(chat_ids, png, text, markup, profile)
        else:
            await self.send_text(chat_ids, text, markup, profile)

    # ---------------------------------------------------------- reply helpers

    async def _reply_photo(self, message: Message, png: bytes, caption: str) -> None:
        """Reply in-thread with a photo; long captions become a separate text message."""
        try:
            if len(caption or "") <= MAX_CAPTION_LEN:
                try:
                    await message.reply_photo(
                        photo=InputFile(png, filename="chart.png"),
                        caption=caption or None,
                        parse_mode=ParseMode.HTML if caption else None,
                    )
                except BadRequest as exc:
                    if not _is_parse_error(exc):
                        raise
                    await message.reply_photo(
                        photo=InputFile(png, filename="chart.png"), caption=_plain(caption) or None
                    )
                return
            await message.reply_photo(photo=InputFile(png, filename="chart.png"))
            await self._reply(message, caption)
        except TelegramError as exc:
            logger.error("reply_photo failed: %s", exc)
            await self._reply(message, caption)

    async def _reply_document(self, message: Message, path: str | Path, caption: str) -> bool:
        p = Path(path)
        try:
            data = p.read_bytes()
        except OSError as exc:
            logger.error("Cannot read %s: %s", p, exc)
            await self._reply(message, f"Sorry, the file {messages.esc(p.name)} could not be read.")
            return False
        caption = caption or ""
        if len(caption) > MAX_CAPTION_LEN:
            caption = caption[: MAX_CAPTION_LEN - 1] + "…"
        try:
            try:
                await message.reply_document(
                    document=InputFile(data, filename=p.name),
                    caption=caption or None,
                    parse_mode=ParseMode.HTML if caption else None,
                )
            except BadRequest as exc:
                if not _is_parse_error(exc):
                    raise
                await message.reply_document(
                    document=InputFile(data, filename=p.name), caption=_plain(caption) or None
                )
            return True
        except TelegramError as exc:
            logger.error("reply_document failed: %s", exc)
            await self._reply(
                message, f"Sorry, sending {messages.esc(p.name)} failed: {messages.esc(str(exc))}"
            )
            return False

    # ------------------------------------------------------------- commands

    def _greeting(self, chat_id: int) -> str:
        profiles = self.config.profiles_for_chat(chat_id)
        names = ", ".join(f"<b>{messages.esc(p.name)}</b>" for p in profiles) or "nobody yet"
        lines = [
            "👋 Hello! I watch a Garmin watch and send simple health updates here.",
            f"This chat can see: {names}.",
        ]
        if any(p.features.palpitations for p in profiles):
            lines.append(
                "If you feel your heart racing or fluttering, send /palp (add the time and a note if you like). "
                "It goes into a diary you can show your doctor; it is not a diagnosis."
            )
        return "\n".join(lines)

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = self._authorised_chat(update)
        message = self._message(update)
        if chat_id is None or message is None:
            return
        text = self._greeting(chat_id) + "\n\n" + messages.help_text(self._is_admin(chat_id))
        await self._reply(message, text)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = self._authorised_chat(update)
        message = self._message(update)
        if chat_id is None or message is None:
            return
        await self._reply(message, messages.help_text(self._is_admin(chat_id)))

    async def cmd_profiles(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = self._authorised_chat(update)
        message = self._message(update)
        if chat_id is None or message is None:
            return
        profiles = self.config.profiles_for_chat(chat_id)
        lines = ["<b>Profiles this chat can see</b>"]
        for p in profiles:
            feats = []
            if p.features.palpitations:
                feats.append("palpitation diary")
            if p.features.daily_coaching:
                feats.append("coaching")
            if p.features.doctor_report:
                feats.append("doctor report")
            extra = f" — {', '.join(feats)}" if feats else ""
            lines.append(
                f"• <b>{messages.esc(p.name)}</b> ({messages.esc(p.timezone or self.config.timezone)}){extra}"
            )
        if len(profiles) > 1:
            lines.append("")
            lines.append(
                f"Put the name first to pick one, e.g. <code>/today {messages.esc(profiles[0].name)}</code>."
            )
        await self._reply(message, "\n".join(lines))

    async def _simple(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, what: str, method: str
    ) -> None:
        ctx = await self._begin(update, context)
        if ctx is None:
            return
        fn = getattr(self.service, method)
        text = await self._run(ctx, what, fn, ctx.profile)
        if text is not None:
            await self._reply(ctx.message, str(text) or "n/a")

    async def cmd_today(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._simple(update, context, "fetch today's data", "today_text")

    async def cmd_yesterday(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._simple(update, context, "fetch yesterday's data", "yesterday_text")

    async def cmd_sleep(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._simple(update, context, "fetch the sleep data", "sleep_text")

    async def cmd_steps(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._simple(update, context, "fetch the step data", "steps_text")

    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ctx = await self._begin(update, context)
        if ctx is None:
            return
        await self._reply(ctx.message, "🧠 Asking the local model… this can take a minute.")
        text = await self._run(ctx, "run the analysis", self.service.analyze_now, ctx.profile)
        if text is not None:
            await self._reply(ctx.message, str(text) or "n/a")

    async def cmd_hr(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ctx = await self._begin(update, context)
        if ctx is None:
            return
        day = None
        if ctx.args:
            raw = ctx.args[0].strip().lower()
            if raw in {"today", "now"}:
                day = None
            elif raw == "yesterday":
                day = to_local(now_utc(), ctx.profile.timezone).date() - timedelta(days=1)
            else:
                try:
                    day = parse_date(ctx.args[0])
                except ValueError:
                    await self._reply(
                        ctx.message,
                        "Please give the day as YYYY-MM-DD, e.g. <code>/hr 2026-09-27</code>.",
                    )
                    return
        result = await self._run(
            ctx, "draw the heart-rate chart", self.service.hr_chart, ctx.profile, day
        )
        if result is None:
            return
        try:
            text, png = result
        except (TypeError, ValueError):
            text, png = str(result), None
        if png:
            await self._reply_photo(ctx.message, png, text or "")
        else:
            await self._reply(ctx.message, text or "n/a")

    async def cmd_episodes(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ctx = await self._begin(update, context)
        if ctx is None:
            return
        days = _parse_days(ctx.args, DEFAULT_EPISODE_DAYS, MAX_EPISODE_DAYS)
        text = await self._run(ctx, "read the diary", self.service.episodes_text, ctx.profile, days)
        if text is not None:
            await self._reply(ctx.message, str(text) or "n/a")

    async def _log_symptom(self, ctx: _Ctx, event_time: datetime | None, note: str) -> None:
        text = await self._run(
            ctx,
            "save the note",
            self.service.log_symptom,
            ctx.profile,
            event_time,
            note,
            ctx.chat_id,
        )
        if text is not None:
            await self._reply(ctx.message, str(text) or "Recorded.")
            # copy to the family (profile chats + admins) so a log from Dad's chat reaches you too
            others = [
                c
                for c in dict.fromkeys([*ctx.profile.telegram_chat_ids, *self.config.telegram.admin_chat_ids])
                if c != ctx.chat_id
            ]
            if others:
                await self.send_text(others, str(text), profile=ctx.profile)

    async def cmd_palp(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ctx = await self._begin(update, context)
        if ctx is None:
            return
        try:
            event_time, note = parse_symptom_args(
                ctx.args, ctx.profile.timezone or self.config.timezone
            )
        except ValueError as exc:
            await self._reply(
                ctx.message,
                f"{messages.esc(str(exc))}\nExample: <code>/palp 14:30 felt fluttering after lunch</code>",
            )
            return
        await self._log_symptom(ctx, event_time, note)

    async def cmd_note(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ctx = await self._begin(update, context)
        if ctx is None:
            return
        note = " ".join(a.strip() for a in ctx.args).strip()
        if not note:
            await self._reply(
                ctx.message,
                "What should I note down? Example: <code>/note dizzy after standing up</code>",
            )
            return
        await self._log_symptom(ctx, None, note)

    async def cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ctx = await self._begin(update, context)
        if ctx is None:
            return
        days = _parse_days(ctx.args, DEFAULT_REPORT_DAYS, MAX_REPORT_DAYS)
        await self._reply(
            ctx.message,
            f"🩺 Generating the {days}-day doctor report for {messages.esc(ctx.profile.name)}… this can take a minute.",
        )
        files = await self._run(
            ctx, "generate the doctor report", self.service.doctor_report, ctx.profile, days
        )
        if files is None:
            return
        n_eps = getattr(files, "n_episodes", None)
        n_sym = getattr(files, "n_symptoms", None)
        caption = messages.doctor_report_caption(
            ctx.profile, days, n_eps if n_eps is not None else 0, n_sym if n_sym is not None else 0
        )
        pdf = getattr(files, "pdf_path", None)
        csv = getattr(files, "csv_path", None)
        if pdf:
            await self._reply_document(ctx.message, pdf, caption)
        if csv:
            await self._reply_document(
                ctx.message, csv, "CSV export of the same diary (one row per episode and per note)"
            )
        if not pdf and not csv:
            await self._reply(ctx.message, "The report came back empty; nothing to send.")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = self._authorised_chat(update)
        message = self._message(update)
        if chat_id is None or message is None:
            return
        if self.service is None:
            await self._reply(message, "The monitor service is not running.")
            return
        if self._is_admin(chat_id):
            try:
                text = await asyncio.to_thread(self.service.status_text)
            except Exception as exc:  # noqa: BLE001
                logger.exception("status_text failed")
                text = f"Status is unavailable right now ({messages.esc(type(exc).__name__)})."
            pending = self.mfa.pending()
            if pending:
                text += "\n🔐 Waiting for /mfa code: " + ", ".join(messages.esc(n) for n in pending)
            await self._reply(message, text)
            return
        lines = ["<b>Status</b>"]
        for p in self.config.profiles_for_chat(chat_id):
            diary = "on" if p.features.palpitations else "off"
            lines.append(
                f"• <b>{messages.esc(p.name)}</b>: monitoring active, palpitation diary {diary}, "
                f"times shown in {messages.esc(p.timezone or self.config.timezone)}"
            )
        lines.append("Use /today for the latest numbers.")
        await self._reply(message, "\n".join(lines))

    async def cmd_mfa(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = self._authorised_chat(update)
        message = self._message(update)
        if chat_id is None or message is None:
            return
        args = self._args(context)
        if not args:
            await self._reply(message, "Send the code like this: <code>/mfa 123456</code>")
            return
        profiles = self.config.profiles_for_chat(chat_id)
        pending = self.mfa.pending()
        profile_name: str | None = None
        if len(args) >= 2:
            first = args[0].strip().lower()
            for p in profiles:
                if p.name.lower() == first or p.slug == first:
                    profile_name = p.name
                    args = args[1:]
                    break
        code = "".join(args).strip()
        if profile_name is None:
            if len(pending) == 1:
                profile_name = pending[0]
            elif len(profiles) == 1:
                profile_name = profiles[0].name
            elif len(pending) > 1:
                names = ", ".join(messages.esc(n) for n in pending)
                await self._reply(
                    message,
                    f"Several logins are waiting ({names}). Send <code>/mfa &lt;name&gt; &lt;code&gt;</code>.",
                )
                return
        if not pending:
            await self._reply(
                message,
                "No Garmin login is waiting for a code right now. The next poll will ask again if needed.",
            )
            return
        if self.mfa.submit(profile_name, code):
            await self._reply(
                message,
                f"🔐 Code received for {messages.esc(profile_name or pending[0])}; finishing the Garmin login…",
            )
        else:
            names = ", ".join(messages.esc(n) for n in pending) or "none"
            await self._reply(
                message,
                f"No login is waiting for {messages.esc(profile_name or '?')}. Waiting: {names}.",
            )

    async def cmd_unknown(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = self._authorised_chat(update)
        message = self._message(update)
        if chat_id is None or message is None:
            return
        text = getattr(message, "text", None)
        words = text.split() if isinstance(text, str) else []
        first = words[0] if words else ""
        # In groups a command may be aimed at another bot (/cmd@OtherBot): stay quiet.
        if "@" in first:
            try:
                me = self.app.bot.username
            except Exception:  # noqa: BLE001 - bot not initialised yet
                me = None
            if me and not first.lower().endswith(f"@{str(me).lower()}"):
                return
        await self._reply(message, "I don't know that command. Send /help for the list.")

    # ------------------------------------------------------------- callbacks

    async def cb_felt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = getattr(update, "callback_query", None)
        if query is None:
            return
        chat_id = self._authorised_chat(update)
        if chat_id is None:
            try:
                await query.answer()
            except TelegramError:
                pass
            return
        m = FELT_CALLBACK_RE.match(str(getattr(query, "data", "") or ""))
        if not m:
            await query.answer()
            return
        episode_id, felt = int(m.group(1)), m.group(2) == "yes"
        if self.service is None:
            await query.answer("The monitor service is not running.")
            return
        try:
            result = await asyncio.to_thread(self.service.set_felt, None, episode_id, felt)
        except Exception:  # noqa: BLE001
            logger.exception("set_felt(%s, %s) failed", episode_id, felt)
            try:
                await query.answer("Sorry, that could not be recorded. Please try again.")
            except TelegramError:
                pass
            return
        toast = _plain(str(result or ("Recorded: felt." if felt else "Recorded: not noticed.")))
        try:
            await query.answer(toast[:MAX_CALLBACK_ANSWER_LEN])
        except TelegramError as exc:
            logger.warning("callback answer failed: %s", exc)
        suffix = "✅ recorded: felt" if felt else "recorded: not noticed"
        await self._strip_buttons(query, suffix)

    async def _strip_buttons(self, query: Any, suffix: str) -> None:
        """Remove the inline buttons from the alert and append ``suffix`` to its text/caption."""
        msg = getattr(query, "message", None)
        caption = getattr(msg, "caption", None)
        caption_html = getattr(msg, "caption_html", None)
        text_html = getattr(msg, "text_html", None)
        text = getattr(msg, "text", None)
        try:
            if isinstance(caption, str):
                base = caption_html if isinstance(caption_html, str) else messages.esc(caption)
                new = f"{base}\n\n{suffix}"
                if len(new) <= MAX_CAPTION_LEN:
                    await query.edit_message_caption(
                        caption=new, parse_mode=ParseMode.HTML, reply_markup=None
                    )
                    return
            elif isinstance(text, str) or isinstance(text_html, str):
                base = text_html if isinstance(text_html, str) else messages.esc(text)
                new = f"{base}\n\n{suffix}"
                if len(new) <= MAX_TEXT_LEN:
                    await query.edit_message_text(
                        text=new, parse_mode=ParseMode.HTML, reply_markup=None
                    )
                    return
            await query.edit_message_reply_markup(reply_markup=None)
        except BadRequest as exc:
            if "not modified" in str(exc).lower():
                return
            logger.warning("Could not edit alert message (%s); removing buttons only", exc)
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except TelegramError as exc2:
                logger.warning("Could not remove buttons: %s", exc2)
        except TelegramError as exc:
            logger.warning("Could not edit alert message: %s", exc)


__all__ = [
    "FELT_CALLBACK_RE",
    "HealthBot",
    "MfaBroker",
    "build_felt_keyboard",
    "chunk_text",
    "parse_symptom_args",
]
