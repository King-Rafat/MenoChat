

from __future__ import annotations

import asyncio
import io
import logging
import os
import subprocess
import tempfile
from typing import Optional

import httpx
import regex as re
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, RetryAfter
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from llm_utils import reset_session, run_chat_turn


load_dotenv("deploy.env")

BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    ,
)
TTS_BACKEND_URL = os.getenv("TTS_BACKEND_URL", "http://localhost:5431").rstrip("/")
EDIT_INTERVAL_S = float(os.getenv("TG_EDIT_INTERVAL_S", "1.2"))
MAX_MSG_CHARS = 4000  # below Telegram's 4096 cap, leave headroom
SHOW_TRANSCRIPTION = True

log = logging.getLogger("menochat_tg")


# ─────────────────────────────────────────────────────────────────────────────
# Shared module state (set by start_telegram_bot)
# ─────────────────────────────────────────────────────────────────────────────
_asr_pipe = None                       # callable: pipe(wav_path) -> {"text": ...}
_gpu_lock: Optional[asyncio.Lock] = None
_app: Optional[Application] = None
_started: bool = False
_start_lock = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Text + audio helpers
# ─────────────────────────────────────────────────────────────────────────────
_MD_PATTERNS = [
    re.compile(r"\[.*?\]"),
    re.compile(r"\*{1,3}(.*?)\*{1,3}"),
    re.compile(r"#{1,6}\s?"),
    re.compile(r"`{1,3}[^`]*`{1,3}"),
    re.compile(r"[-*+]\s"),
    re.compile(r"\d+\.\s"),
    re.compile(r">{1,}\s?"),
    re.compile(r"_"),
]


def clean_for_tts(text: str) -> str:
    for pat in _MD_PATTERNS:
        text = pat.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def session_id_for(chat_id: int) -> str:
    return f"tg_{chat_id}"


def _run_ffmpeg(args: list[str], stdin_bytes: bytes | None = None) -> bytes:
    p = subprocess.run(args, input=stdin_bytes, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed: {p.stderr[:300].decode(errors='ignore')}"
        )
    return p.stdout


def ffmpeg_to_wav16k(in_path: str, out_path: str) -> None:
    _run_ffmpeg([
        "ffmpeg", "-y", "-i", in_path,
        "-ar", "16000", "-ac", "1",
        "-f", "wav", out_path,
    ])


def ffmpeg_wav_to_ogg_opus(wav_bytes: bytes) -> bytes:
    return _run_ffmpeg(
        [
            "ffmpeg", "-y",
            "-i", "pipe:0",
            "-c:a", "libopus",
            "-b:a", "32k",
            "-ac", "1",
            "-f", "ogg",
            "pipe:1",
        ],
        stdin_bytes=wav_bytes,
    )


def _transcribe_sync(wav_path: str) -> str:
    if _asr_pipe is None:
        raise RuntimeError("ASR pipeline not provided to start_telegram_bot()")
    out = _asr_pipe(wav_path)
    if isinstance(out, dict):
        return out.get("text", "") or ""
    return str(out)


async def tts_one_shot(text: str) -> bytes | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{TTS_BACKEND_URL}/tts", json={"text": text})
            r.raise_for_status()
            return r.content
    except Exception as e:
        log.error("TTS request failed: %s", e)
        return None


async def _safe_reply(update: Update, text: str) -> None:
    MAX = 4096
    if len(text) <= MAX:
        await update.message.reply_text(text)
        return
    buf = text
    while buf:
        if len(buf) <= MAX:
            await update.message.reply_text(buf)
            return
        cut = buf.rfind("\n\n", 0, MAX)
        if cut == -1:
            cut = buf.rfind("\n", 0, MAX)
        if cut == -1:
            cut = MAX
        await update.message.reply_text(buf[:cut].strip())
        buf = buf[cut:].strip()


# Marker that llm_utils prepends before appending the sources list.
_SOURCES_MARKER = "📚 সূত্র"


def strip_sources(text: str) -> str:
    """Cut off the '📚 সূত্র: ...' block that llm_utils appends. Telegram only."""
    if not text:
        return text
    idx = text.find(_SOURCES_MARKER)
    if idx < 0:
        return text
    return text[:idx].rstrip()


def split_body_and_tail(streamed: str, full: str) -> tuple[str, str]:
    s = (streamed or "").strip()
    f = (full or "").strip()
    if not f:
        return s, ""
    if not s:
        return f, ""
    if f.startswith(s) and len(f) > len(s):
        return s, f[len(s):].strip()
    return f, ""


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    sid = session_id_for(update.effective_chat.id)
    reset_session(sid)
    log.info("New session: %s", sid)
    await update.message.reply_text(
        "আসসালামু আলাইকুম! আমি MenoChat 🌸\n\n"
        "বাংলা মহিলা স্বাস্থ্য সহকারী। পিরিয়ড, PCOS, পেরিমেনোপজ, মেনোপজ — "
        "টেক্সট বা ভয়েসে যেকোনো প্রশ্ন করতে পারেন।\n\n"
        "সেশন রিসেট: /reset"
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    sid = session_id_for(update.effective_chat.id)
    reset_session(sid)
    log.info("Session reset: %s", sid)
    await update.message.reply_text("সেশন রিসেট হয়েছে ✅")


# ─────────────────────────────────────────────────────────────────────────────
# Core turn handler: stream text, then send tail + voice
# ─────────────────────────────────────────────────────────────────────────────
async def _process_user_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_text: str,
) -> None:
    chat_id = update.effective_chat.id
    sid = session_id_for(chat_id)
    log.info("[%s] USER: %s", sid, user_text[:80])

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    placeholder = "✍️ ..."
    bubble = await update.message.reply_text(placeholder)

    state = {
        "text": "",
        "dirty": False,
        "done": False,
        "last_sent": placeholder,
    }

    async def stream_cb(token: str) -> None:
        state["text"] += token
        state["dirty"] = True

    async def edit_loop() -> None:
        while not state["done"]:
            await asyncio.sleep(EDIT_INTERVAL_S)
            if not state["dirty"]:
                continue
            new = state["text"][:MAX_MSG_CHARS].strip()
            if not new or new == state["last_sent"]:
                continue
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=bubble.message_id,
                    text=new,
                )
                state["last_sent"] = new
                state["dirty"] = False
            except RetryAfter as e:
                await asyncio.sleep(float(e.retry_after) + 0.5)
            except BadRequest as e:
                msg = str(e).lower()
                if "not modified" in msg:
                    state["dirty"] = False
                elif "message is too long" in msg:
                    return
                else:
                    log.warning("edit BadRequest: %s", e)
            except Exception as e:
                log.warning("edit error: %s", e)

    async def progress_cb(stage: str, data: dict):
        try:
            short = {
                k: (str(v)[:120] if not isinstance(v, (int, float, bool)) else v)
                for k, v in (data or {}).items()
            }
        except Exception:
            short = {}
        log.info("[%s] stage=%s %s", sid, stage, short)

    editor = asyncio.create_task(edit_loop())

    try:
        result = await run_chat_turn(
            sid,
            user_text,
            stream=True,
            stream_callback=stream_cb,
            progress_callback=progress_cb,
            debug=True,
        )
    except Exception as e:
        log.exception("run_chat_turn failed")
        state["done"] = True
        await editor
        try:
            await bubble.edit_text(f"❌ ত্রুটি হয়েছে: {e}")
        except Exception:
            pass
        return

    state["done"] = True
    await editor

    streamed_body = state["text"]
    full_answer = strip_sources((result.get("answer_text") or "").strip())
    body, tail = split_body_and_tail(streamed_body, full_answer)

    final_body = (body or full_answer)[:MAX_MSG_CHARS]
    if final_body and final_body != state["last_sent"]:
        try:
            await bubble.edit_text(final_body)
        except BadRequest as e:
            if "not modified" not in str(e).lower():
                log.warning("final body edit: %s", e)
        except Exception as e:
            log.warning("final body edit: %s", e)

    if tail:
        await _safe_reply(update, tail)

    # ── Single voice note at the end ─────────────────────────────────────────
    try:
        await context.bot.send_chat_action(
            chat_id=chat_id, action=ChatAction.RECORD_VOICE
        )
        cleaned = clean_for_tts(full_answer)
        if not cleaned:
            return

        wav_bytes = await tts_one_shot(cleaned)
        if not wav_bytes:
            return

        loop = asyncio.get_event_loop()
        ogg_bytes = await loop.run_in_executor(
            None, ffmpeg_wav_to_ogg_opus, wav_bytes
        )
        await context.bot.send_voice(
            chat_id=chat_id,
            voice=io.BytesIO(ogg_bytes),
        )
    except Exception as e:
        log.warning("voice send failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Message handlers
# ─────────────────────────────────────────────────────────────────────────────
async def _handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if not text:
        return
    await _process_user_text(update, context, text)


async def _handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    sid = session_id_for(chat_id)
    log.info("[%s] VOICE in", sid)

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)
    voice_file = await update.message.voice.get_file()

    with tempfile.TemporaryDirectory() as tmp:
        ogg_path = os.path.join(tmp, "in.ogg")
        wav_path = os.path.join(tmp, "in.wav")

        await voice_file.download_to_drive(ogg_path)

        try:
            ffmpeg_to_wav16k(ogg_path, wav_path)
        except Exception as e:
            log.error("audio convert failed: %s", e)
            await update.message.reply_text(
                "ভয়েস মেসেজ প্রক্রিয়া করতে সমস্যা হয়েছে। আবার পাঠান বা টেক্সটে লিখুন।"
            )
            return

        loop = asyncio.get_event_loop()
        try:
            if _gpu_lock is not None:
                async with _gpu_lock:
                    text = await loop.run_in_executor(None, _transcribe_sync, wav_path)
            else:
                text = await loop.run_in_executor(None, _transcribe_sync, wav_path)
        except Exception as e:
            log.error("ASR failed: %s", e)
            await update.message.reply_text(
                "ভয়েস থেকে লেখা বের করতে সমস্যা হয়েছে। টেক্সটে লিখে পাঠান।"
            )
            return

    text = (text or "").strip()
    if not text:
        await update.message.reply_text(
            "ভয়েস মেসেজ থেকে কিছু বুঝতে পারিনি। আবার বলুন বা টেক্সটে লিখুন।"
        )
        return

    log.info("[%s] ASR: %s", sid, text[:80])
    if SHOW_TRANSCRIPTION:
        await update.message.reply_text(
            f"🎙️ আপনি বলেছেন:\n_{text}_",
            parse_mode="Markdown",
        )

    await _process_user_text(update, context, text)


# ─────────────────────────────────────────────────────────────────────────────
# Public start / stop
# ─────────────────────────────────────────────────────────────────────────────
async def start_telegram_bot(
    *,
    asr,
    gpu_lock: Optional[asyncio.Lock] = None,
    bot_token: Optional[str] = None,
) -> Optional[Application]:
    """
    Launch the Telegram bot inside the current asyncio loop. Idempotent:
    calling this twice is a no-op after the first successful start.

    Pass:
      • asr        — the ASR pipeline that Chainlit already loaded.
      • gpu_lock   — chainlit's asyncio.Lock used to serialize GPU work.
      • bot_token  — overrides TELEGRAM_BOT_TOKEN env / hardcoded default.
    """
    global _asr_pipe, _gpu_lock, _app, _started

    async with _start_lock:
        if _started and _app is not None:
            return _app

        _asr_pipe = asr
        _gpu_lock = gpu_lock

        token = bot_token or BOT_TOKEN
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN missing")

        app = Application.builder().token(token).build()
        app.add_handler(CommandHandler("start", cmd_start))
        app.add_handler(CommandHandler("reset", cmd_reset))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_text))
        app.add_handler(MessageHandler(filters.VOICE, _handle_voice))

        log.info("Telegram bot booting inside host process...")
        await app.initialize()
        await app.start()
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        log.info("Telegram bot polling started.")
        _app = app
        _started = True
        return app


async def stop_telegram_bot() -> None:
    global _app, _started
    if _app is None:
        return
    try:
        await _app.updater.stop()
        await _app.stop()
        await _app.shutdown()
    except Exception as e:
        log.warning("telegram bot stop error: %s", e)
    finally:
        _app = None
        _started = False

