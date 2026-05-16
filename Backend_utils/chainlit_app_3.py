# system
import os
import httpx
import asyncio
import struct
import time
from typing import Optional
from dotenv import load_dotenv
from io import BytesIO
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import warnings
warnings.filterwarnings("ignore")
import uuid
os.makedirs("tmp_audio", exist_ok=True)
import chainlit as cl
from chainlit.types import ThreadDict

import numpy as np
import torch
import soundfile as sf
import io
import wave
import regex as re
from aiohttp import web
from contextlib import suppress

from jose import jwt, JWTError

#from load_models.load_tts import load_menstrual_bangla_tts
from load_models.load_asr import load_asr
from load_vector_db import load_vector_db, load_embedding_models

import gc
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from llm_utils import init_llm_pipeline, reset_session, run_chat_turn, llm_chat

load_dotenv("deploy.env")

API_URL      = os.getenv("API_URL")
FASTAPI_URL  = os.getenv("FASTAPI_URL")
LLM_URL      = os.getenv("LLM_URL").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "meno")

DB_DIR       = "vector_db"
EMB_MODEL_DIR   = "../models/embedder"
RERANK_MODEL_DIR = "../models/reranker"
TTS_MODEL_DIR   = "../models/spark_tts"
ASR_MODEL_DIR   = "../models/asr"

FAISS_INDEX_PATH = f"{DB_DIR}/faiss.index"
CHUNKS_PATH      = f"{DB_DIR}/CHUNKS.jsonl"
META_PATH        = f"{DB_DIR}/meta.jsonl"
UID_LIST_PATH    = f"{DB_DIR}/uid_list.txt"

SILENCE_THRESHOLD = 3500
SILENCE_TIMEOUT   = 1300.0

SECRET_KEY        = "C@odmUUt5H4$%UPob*zQDXcl=:q_fjXwFwlE-9cuXA?LjofbpbrHwsKA7SE5Fh6A"
ALGORITHM         = "HS256"

_EXECUTOR = ThreadPoolExecutor(max_workers=2)

gpu_lock = asyncio.Lock()

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

FIRST_CHUNK_WORDS = int(os.getenv("TTS_FIRST_CHUNK_WORDS", "10"))
FOLLOWUP_CHUNK_WORDS = int(os.getenv("TTS_FOLLOWUP_CHUNK_WORDS", "28"))
STREAM_HOST = os.getenv("TTS_STREAM_HOST", "127.0.0.1")
STREAM_PORT = int(os.getenv("TTS_STREAM_PORT", "8765"))
STREAM_PUBLIC_BASE = os.getenv("TTS_STREAM_PUBLIC_BASE", f"http://{STREAM_HOST}:{STREAM_PORT}")
TTS_TRACE = os.getenv("TTS_TRACE", "0") == "1"

# VITS HTTP backend (separate FastAPI service running tts_server.py)
TTS_SAMPLE_RATE = 22050  # VITS output sample rate
TTS_BACKEND_URL = os.getenv("TTS_BACKEND_URL", "").rstrip("/")

_active_streams: dict[str, asyncio.Queue] = {}
_stream_tasks: dict[str, asyncio.Task] = {}
_stream_runner: web.AppRunner | None = None
_stream_server_lock = asyncio.Lock()

# tts_model = load_menstrual_bangla_tts(TTS_MODEL_DIR)
faiss_index, db_chunks, db_meta = load_vector_db(
    FAISS_INDEX_PATH, CHUNKS_PATH, META_PATH, UID_LIST_PATH
)
embed_model, reranker = load_embedding_models(EMB_MODEL_DIR, RERANK_MODEL_DIR)
asr = load_asr()
asr.model.config.forced_decoder_ids = asr.tokenizer.get_decoder_prompt_ids(
    language="bn", task="transcribe"
)

with open(f"{DB_DIR}/uid_list.txt", encoding="utf-8") as f:
    db_uids = [line.strip() for line in f if line.strip()]

print(f"FAISS: {faiss_index.ntotal} vectors | Chunks: {len(db_chunks)}")
gc.collect()
torch.cuda.empty_cache()
import gc

def diagnose_vram():
    print("\n" + "=" * 80)
    print("VRAM DIAGNOSTIC")
    print("=" * 80)

    # Overall numbers
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Total VRAM: {total:.2f} GB")
    print(f"PyTorch allocated: {alloc:.2f} GB")
    print(f"PyTorch reserved:  {reserved:.2f} GB")
    print(f"(Non-PyTorch overhead = process_total - reserved)")

    # Check each known model
    print("\n--- Known model objects ---")
    candidates = {
        "embed_model.model": embed_model.model if hasattr(embed_model, "model") else None,
        "reranker.model": reranker.model if hasattr(reranker, "model") else None,
        "asr.model": asr.model if hasattr(asr, "model") else None,
    }
    for name, m in candidates.items():
        if m is None:
            print(f"  {name}: NOT FOUND")
            continue
        try:
            dev = next(m.parameters()).device
            n_params = sum(p.numel() for p in m.parameters())
            size_gb = sum(p.numel() * p.element_size() for p in m.parameters()) / 1e9
            print(f"  {name}: device={dev}, params={n_params/1e6:.1f}M, size={size_gb:.2f} GB")
        except Exception as e:
            print(f"  {name}: error reading: {e}")

    # Scan ALL tensors that ended up on CUDA, sorted by size
    print("\n--- All CUDA tensors in memory (top 20 by size) ---")
    cuda_tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                cuda_tensors.append((obj.numel() * obj.element_size(), obj.shape, obj.dtype))
        except Exception:
            pass
    cuda_tensors.sort(key=lambda x: -x[0])
    for size, shape, dtype in cuda_tensors[:20]:
        print(f"  {size/1e6:7.1f} MB  shape={tuple(shape)}  dtype={dtype}")
    print(f"  ... total CUDA tensors: {len(cuda_tensors)}")

    # PyTorch's own memory summary
    print("\n--- PyTorch memory_summary (abbreviated) ---")
    print(torch.cuda.memory_summary(abbreviated=True))
    print("=" * 80 + "\n")

diagnose_vram()



asyncio.run(
    init_llm_pipeline(
        openai_base_url=f"{LLM_URL}/v1",
        openai_api_key=OPENAI_API_KEY,
        model_name=LLM_MODEL_NAME,
        faiss_index=faiss_index,
        chunks=db_chunks,
        meta=db_meta,
        uids=db_uids,
        embed_model=embed_model,
        reranker=reranker,
        session_storage_dir="menochat_sessions",
        executor=_EXECUTOR,
    )
)

_MD_PATTERNS = [
    re.compile(r'\[.*?\]'),
    re.compile(r'\*{1,3}(.*?)\*{1,3}'),
    re.compile(r'#{1,6}\s?'),
    re.compile(r'`{1,3}[^`]*`{1,3}'),
    re.compile(r'[-*+]\s'),
    re.compile(r'\d+\.\s'),
    re.compile(r'>{1,}\s?'),
    re.compile(r'_'),
]

async def _stream_token_to_message(msg: cl.Message, token: str):
    await msg.stream_token(token)


# Human-friendly label for each planner route.
_ROUTE_LABELS = {
    "out-of-scope":         "Out of scope",
    "smalltalk":            "Small talk",
    "health_direct":        "Direct health question",
    "health_followup":      "Follow-up question",
    "health_education":     "Educational explainer",
    "sensitive_supportive": "Sensitive / supportive",
    "urgent_redflag":       "URGENT red flag",
}


async def run_chat(user_text: str, session_id: str, reset: bool = False):
    if reset:
        reset_session(session_id)

    msg = cl.Message(content="")

    state: dict = {
        "current_step": None,
        "current_stage": None,
        "t0": {},
        "started_answer": False,
        # Use 1 (UI transparency): live "Model thinking" step
        "thinking_step": None,
        "thinking_started": False,
    }

    async def on_thinking(token: str):
        # Lazily create the thinking step the first time a reasoning token arrives.
        if not state["thinking_started"]:
            step = cl.Step(name="🧠 ভাবছি (model thinking)", type="tool")
            await step.__aenter__()
            state["thinking_step"] = step
            state["thinking_started"] = True
        step = state["thinking_step"]
        if step is not None:
            await step.stream_token(token)

    async def _close_current(output_text: Optional[str] = None, success: bool = True):
        step = state["current_step"]
        if step is None:
            return
        if output_text:
            step.output = output_text
        try:
            await step.__aexit__(None, None, None)
        except Exception:
            pass
        state["current_step"] = None
        state["current_stage"] = None

    async def _open_step(stage: str, name: str):
        await _close_current()
        step = cl.Step(name=name, type="tool")
        await step.__aenter__()
        state["current_step"] = step
        state["current_stage"] = stage
        state["t0"][stage] = time.perf_counter()

    def _elapsed(stage: str) -> str:
        t0 = state["t0"].get(stage)
        if t0 is None:
            return ""
        return f" ({time.perf_counter() - t0:.1f}s)"

    async def on_progress(stage: str, data: dict):
        if stage == "planner_start":
            await _open_step("planner", "🧭 প্রশ্ন বুঝছি (planning)...")

        elif stage == "planner_done":
            route = data.get("route", "")
            route_label = _ROUTE_LABELS.get(route, route or "unknown")
            risk = data.get("risk_level", "none")
            resolved = (data.get("resolved_question") or "").strip()
            lines = [
                f"**Route:** {route_label}",
                f"**Risk level:** {risk}",
            ]
            if resolved:
                lines.append(f"**Resolved question:** {resolved}")
            lines.append(f"⏱️{_elapsed('planner')}")
            await _close_current("\n\n".join(lines))

        elif stage == "retrieval_start":
            if data.get("needs_retrieval"):
                await _open_step("retrieval", "🔎 তথ্য খুঁজছি (retrieving sources)...")
            else:
                step = cl.Step(name="🔎 তথ্য খোঁজা দরকার নেই", type="tool")
                await step.__aenter__()
                step.output = "No retrieval needed for this turn."
                await step.__aexit__(None, None, None)

        elif stage == "retrieval_done":
            count = data.get("count", 0)
            top = data.get("top_rerank")
            bits = [f"**Sources found:** {count}"]
            if isinstance(top, (int, float)):
                bits.append(f"**Top rerank score:** {top:.3f}")
            bits.append(f"⏱️{_elapsed('retrieval')}")
            await _close_current("\n\n".join(bits))

        elif stage == "answer_start":
            await _open_step("answer", "✍️ উত্তর লিখছি (writing answer)...")
            state["started_answer"] = True

        elif stage == "answer_done":
            length = data.get("length", 0)
            await _close_current(
                f"Answer generated ({length} chars)⏱️{_elapsed('answer')}"
            )

        elif stage == "comorbidity_start":
            await _open_step("comorbidity", "🥗 খাবার ও রোগ যাচাই করছি...")

        elif stage == "comorbidity_done":
            await _close_current(f"Checked⏱️{_elapsed('comorbidity')}")

        elif stage == "error":
            phase = data.get("phase", "unknown")
            err = data.get("error", "")
            await _close_current(f"❌ Error in {phase}: {err}", success=False)

        elif stage == "turn_done":
            await _close_current()

    try:
        result = await run_chat_turn(
            session_id,
            user_text,
            stream=True,
            stream_callback=lambda token: _stream_token_to_message(msg, token),
            thinking_callback=on_thinking,
            progress_callback=on_progress,
        )
    finally:
        await _close_current()
        # Close the thinking step if it was opened.
        if state["thinking_started"] and state["thinking_step"] is not None:
            try:
                await state["thinking_step"].__aexit__(None, None, None)
            except Exception:
                pass

    # Log each stage's thinking to console for debugging (Use 2).
    thinking = (result or {}).get("thinking", {}) if isinstance(result, dict) else {}
    for stage, txt in thinking.items():
        if txt:
            print(f"[thinking:{stage}] {txt[:200].replace(chr(10), ' ')}...")

    answer = result.get("answer_text", "")
    streamed = bool(msg.content)
    return answer, msg, streamed


async def warmup():
    llm_result = await llm_chat(
        "hello world",
        max_tokens=40,
        temperature=0.0,
        top_p=1.0,
    )
    print(f"LLM warmup result: {llm_result[:80]}")

    if TTS_BACKEND_URL:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    f"{TTS_BACKEND_URL}/tts",
                    json={"text": "হ্যালো"},
                )
                r.raise_for_status()
                print(f"TTS warmup ok ({len(r.content)} bytes)")
        except Exception as e:
            print(f"TTS warmup failed: {e}")
    else:
        print("TTS warmup skipped: TTS_BACKEND_URL not set")

def clean_text(text: str) -> str:
    for pattern in _MD_PATTERNS:
        text = pattern.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _split_15_words(sentence: str) -> list[str]:
    words = sentence.split()
    return [
        ' '.join(words[i:i+15])
        for i in range(0, len(words), 15)
    ]

def sentence_chunks(text: str) -> list[str]:
    cleaned = clean_text(text)
    cleaned = cleaned.replace('।', '.')

    raw_sentences = re.split(r'(?<=[.!?])\s+', cleaned)

    chunks = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        words = sentence.split()
        if len(words) <= 15:
            chunks.append(sentence)
        else:
            sub_chunks = _split_15_words(sentence)
            chunks.extend(sub_chunks)

    return chunks


def _chunk_by_word_budget(text: str, first_n: int, next_n: int) -> list[str]:
    words = [w for w in (text or "").split() if w]
    if not words:
        return []

    chunks = []
    i = 0
    first_n = max(4, first_n)
    next_n = max(first_n, next_n)

    first_chunk = " ".join(words[i : i + first_n]).strip()
    if first_chunk:
        chunks.append(first_chunk)
    i += first_n

    while i < len(words):
        chunk = " ".join(words[i : i + next_n]).strip()
        if chunk:
            chunks.append(chunk)
        i += next_n

    return chunks

def _wav_stream_header(
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    sample_width: int = SAMPLE_WIDTH,
) -> bytes:
    num_channels = channels
    bits_per_sample = sample_width * 8
    byte_rate = sample_rate * num_channels * sample_width
    block_align = num_channels * sample_width
    riff_size = 0xFFFFFFFF
    data_size = 0xFFFFFFFF

    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


def _arr_to_pcm(arr: np.ndarray) -> bytes:
    arr = np.asarray(arr).squeeze()
    if arr.size == 0:
        return b""
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype(np.int16)
    else:
        pcm = arr.astype(np.int16)
    return pcm.tobytes()


async def _tts_stream_handler(request: web.Request) -> web.StreamResponse:
    stream_id = request.match_info.get("stream_id", "")
    queue = _active_streams.get(stream_id)
    if queue is None:
        raise web.HTTPNotFound(text="stream not found")

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "audio/wav",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )
    await response.prepare(request)
    await response.write(_wav_stream_header(sample_rate=TTS_SAMPLE_RATE))

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            await response.write(item)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        _active_streams.pop(stream_id, None)
        task = _stream_tasks.pop(stream_id, None)
        if task and not task.done():
            task.cancel()
        with suppress(Exception):
            await response.write_eof()

    return response


async def _ensure_stream_server_started() -> None:
    global _stream_runner

    if _stream_runner is not None:
        return

    async with _stream_server_lock:
        if _stream_runner is not None:
            return

        app = web.Application()
        app.router.add_get("/tts/{stream_id}", _tts_stream_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, STREAM_HOST, STREAM_PORT)
        await site.start()
        _stream_runner = runner
        print(f"TTS stream server started at {STREAM_PUBLIC_BASE}")


async def _stream_tts_to_queue(stream_id: str, text: str):
    queue = _active_streams.get(stream_id)
    if queue is None:
        return

    if not TTS_BACKEND_URL:
        if TTS_TRACE:
            print(f"[TTS][{stream_id}] TTS_BACKEND_URL is empty; aborting")
        with suppress(asyncio.QueueFull):
            queue.put_nowait(None)
        return

    cleaned = clean_text(text)
    chunks = sentence_chunks(cleaned)

    if TTS_TRACE:
        print(f"[TTS][{stream_id}] start | chunks={len(chunks)} | chars={len(cleaned)}")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            for idx, chunk in enumerate(chunks):
                t0 = time.perf_counter()
                try:
                    r = await client.post(
                        f"{TTS_BACKEND_URL}/tts",
                        json={"text": chunk},
                    )
                    r.raise_for_status()
                except Exception as e:
                    if TTS_TRACE:
                        print(f"[TTS][{stream_id}] chunk={idx + 1} HTTP error: {e}")
                    continue

                try:
                    arr, _sr = sf.read(io.BytesIO(r.content), dtype="float32")
                except Exception as e:
                    if TTS_TRACE:
                        print(f"[TTS][{stream_id}] chunk={idx + 1} decode error: {e}")
                    continue

                if arr.size == 0:
                    continue

                pcm = _arr_to_pcm(arr)
                if not pcm:
                    continue

                try:
                    await asyncio.wait_for(queue.put(pcm), timeout=30.0)
                except asyncio.TimeoutError:
                    if TTS_TRACE:
                        print(f"[TTS][{stream_id}] queue.put timeout; aborting stream")
                    break

                if TTS_TRACE:
                    print(
                        f"[TTS][{stream_id}] chunk={idx + 1}/{len(chunks)} "
                        f"words={len(chunk.split())} dt={time.perf_counter() - t0:.3f}s"
                    )
    except asyncio.CancelledError:
        raise
    finally:
        with suppress(asyncio.QueueFull):
            queue.put_nowait(None)


async def stream_tts(text: str, tts_model, gpu_lock):
    chunks = sentence_chunks(text)
    if not chunks:
        return

    loop = asyncio.get_running_loop()
    os.makedirs("tmp_audio", exist_ok=True)

    _DONE = object()

    queue = asyncio.Queue(maxsize=2)

    async def _producer():
        try:
            for chunk in chunks:
                def _generate(c=chunk):
                    audio = tts_model.generate(text=c)
                    if torch.is_tensor(audio):
                        arr = audio.cpu().numpy().squeeze()
                    else:
                        arr = np.array(audio).squeeze()
                    if arr.size == 0:
                        return None, None

                    output_path = f"tmp_audio/tts_output_{uuid.uuid4().hex}.wav"
                    sf.write(output_path, arr, 16000)

                    buf = io.BytesIO()
                    sf.write(buf, arr, 16000, format='WAV')
                    buf.seek(0)
                    return buf.read(), output_path

                wav_bytes, file_path = await loop.run_in_executor(_EXECUTOR, _generate)
                await queue.put((wav_bytes, file_path))
        finally:
            await queue.put(_DONE)

    async with gpu_lock:
        producer_task = asyncio.create_task(_producer())

        chunk_idx = 0
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            wav_bytes, file_path = item
            if wav_bytes is not None:
                yield chunk_idx, wav_bytes, file_path
                chunk_idx += 1

        await producer_task


import os
import shutil
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
import httpx
import chainlit as cl


AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./saved_audio")).resolve()
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
print(f"[audio] saving uploaded audio into: {AUDIO_DIR}")


async def transcribe_audio(audio_file) -> str:
    print("[transcribe_audio] ENTER", flush=True)
    session_id = cl.user_session.get("id") or "unknown"

    counter = (cl.user_session.get("audio_counter") or 0) + 1
    cl.user_session.set("audio_counter", counter)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:6]
    audio_id  = f"{session_id}_{counter:04d}_{timestamp}_{short_uid}"

    loop = asyncio.get_running_loop()
    temp_path = str(AUDIO_DIR / f"input_audio_{audio_id}.wav")

    src_path = getattr(audio_file, "path", None)
    if src_path and os.path.exists(src_path):
        await loop.run_in_executor(_EXECUTOR, shutil.copyfile, src_path, temp_path)
    elif getattr(audio_file, "content", None):
        def _write_bytes():
            with open(temp_path, "wb") as f:
                f.write(audio_file.content)
        await loop.run_in_executor(_EXECUTOR, _write_bytes)
    elif getattr(audio_file, "url", None):
        async with httpx.AsyncClient() as client:
            r = await client.get(audio_file.url)
            r.raise_for_status()
            def _write_download():
                with open(temp_path, "wb") as f:
                    f.write(r.content)
        await loop.run_in_executor(_EXECUTOR, _write_download)
    else:
        raise Exception(
            f"Audio element has no path, content, or url. Got attrs: {dir(audio_file)}"
        )

    size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
    print(f"[audio] saved {temp_path} ({size} bytes)", flush=True)
    if size == 0:
        raise Exception(f"Audio save failed or file is empty at {temp_path}")

    cleaned_path = str(AUDIO_DIR / f"cleaned_{audio_id}.wav")

    async with gpu_lock:
        
        print(f"[transcribe_audio] preprocess_audio returned: {cleaned_path}", flush=True)
        asr_result = await loop.run_in_executor(_EXECUTOR, asr, temp_path)

    return asr_result["text"]

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="পিরিয়ডের সময় অসস্তি বা ব্যথা অনুভব করছো?",
            message="পিরিয়ডের সময় আমার অনেক পেটব্যথা হয়। এটা কীভাবে কমানো যায়?",
            icon="",
        ),
        cl.Starter(
            label="পিরিয়ড অনিয়মিত হলে কী করা উচিত?",
            message="আমার পিরিয়ড অনিয়মিত হয়েছে। এটা কি স্বাভাবিক, নাকি ডাক্তারের কাছে যাওয়া দরকার?",
            icon="",
        ),
        cl.Starter(
            label="প্যাড, কাপ, নাকি ট্যাম্পন — কোনটা তোমার জন্য ভালো?",
            message="প্যাড, মেনস্ট্রুয়াল কাপ আর ট্যাম্পনের মধ্যে কোনটা সবচেয়ে সুরক্ষিত ও আরামদায়ক?",
            icon="",
        ),
        cl.Starter(
            label="পিরিয়ড চলাকালীন খাবার বা ব্যায়াম নিয়ে দ্বিধায় আছো?",
            message="পিরিয়ডের সময় কী কী খাওয়া উচিত এবং ব্যায়াম করা ঠিক হবে কিনা জানতে চাই।",
            icon="",
        ),
        cl.Starter(
            label="পিরিয়ড নিয়ে মানসিক চাপ বা লজ্জা অনুভব করছো?",
            message="প্রতিবার পিরিয়ড এলে আমি দুশ্চিন্তা আর অস্বস্তি অনুভব করি। এটা কাটানোর উপায় কী?",
            icon="",
        ),
    ]

@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{FASTAPI_URL}/login",
            json={"username": username, "password": password},
            headers={"Content-Type": "application/json"},
        )
    if res.status_code != 200:
        return None
    data    = res.json()
    token   = data.get("access_token")
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    user_id = payload.get("sub")
    if user_id is None:
        return None
    return cl.User(identifier=user_id, metadata={"role": "admin", "provider": "credentials"})

@cl.on_chat_resume
async def on_chat_resume(thread):
    pass

@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_chunks", [])
    return True

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    audio_chunks = cl.user_session.get("audio_chunks")
    if audio_chunks is not None:
        audio_chunks.append(np.frombuffer(chunk.data, dtype=np.int16))

@cl.action_callback("generate_speech")
async def on_action(action: cl.Action):
    message_text = action.payload.get("text", "")
    if not message_text.strip():
        return

    try:
        await _ensure_stream_server_started()
        stream_id = uuid.uuid4().hex
        queue = asyncio.Queue(maxsize=4)
        _active_streams[stream_id] = queue

        task = asyncio.create_task(_stream_tts_to_queue(stream_id, message_text))
        _stream_tasks[stream_id] = task

        stream_url = f"{STREAM_PUBLIC_BASE}/tts/{stream_id}"
        await cl.Message(
            content="অডিও তৈরী হচ্ছে:",
            elements=[
                cl.Audio(
                    auto_play=True,
                    mime="audio/wav",
                    url=stream_url,
                    name="Audio will start as soon as the stream arrives.",
                )
            ],
        ).send()

    except Exception as e:
        import traceback
        print(f"Error generating speech: {e}")
        traceback.print_exc()

        await cl.Message(
            content="দুঃখিত, অডিও তৈরিতে সমস্যা হয়েছে। পরে আবার চেষ্টা করুন।"
        ).send()

async def process_audio(transcription, temp_path):
    with open(temp_path, "rb") as f:
        audio_buffer = f.read()

    input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")
    await cl.Message(
        author="You",
        type="user_message",
        content=transcription,
        elements=[input_audio_el],
    ).send()


@cl.on_audio_end
async def on_audio_end():
    print("[on_audio_end] ENTER", flush=True)
    audio_chunks = cl.user_session.get("audio_chunks") or []
    cl.user_session.set("audio_chunks", [])

    session_id = cl.user_session.get("id") or "unknown"
    loop = asyncio.get_running_loop()

    if not audio_chunks:
        await cl.Message(content="⚠️ No audio recorded").send()
        return

    counter = (cl.user_session.get("audio_counter") or 0) + 1
    cl.user_session.set("audio_counter", counter)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:6]
    audio_id  = f"{session_id}_{counter:04d}_{timestamp}_{short_uid}"

    temp_path = str(AUDIO_DIR / f"recorded_audio_{audio_id}.wav")
    cleaned_path = str(AUDIO_DIR / f"cleaned_recorded_{audio_id}.wav")

    try:
        concatenated = np.concatenate(audio_chunks)
        audio = AudioSegment(
            concatenated.tobytes(), frame_rate=48000, sample_width=2, channels=1
        ).set_frame_rate(16000)
        await loop.run_in_executor(_EXECUTOR, lambda: audio.export(temp_path, format="wav"))

        size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
        print(f"[audio] saved {temp_path} ({size} bytes)", flush=True)
        if size == 0:
            raise Exception(f"Recorded audio save failed at {temp_path}")

        async with cl.Step(name="🎤 Transcribing audio...") as step:
            async with gpu_lock:
                
                asr_result = await loop.run_in_executor(_EXECUTOR, asr, temp_path)
            transcription = asr_result["text"]
            step.output   = f"📝 {transcription}"
            await process_audio(transcription, temp_path)

        answer, msg, streamed = await run_chat(transcription, session_id=session_id, reset=False)
        msg.content = answer

        msg.actions = [
            cl.Action(
                name="generate_speech",
                payload={"text": answer},
                label="🔊 শুনুন (Listen)",
                description="এই উত্তরটি অডিওতে শুনুন",
            )
        ]

        if streamed:
            await msg.update()
        else:
            await msg.send()

    except Exception as e:
        import traceback
        print(f"Error processing audio: {e}")
        traceback.print_exc()

        await cl.Message(
            content="দুঃখিত, অডিও তৈরিতে সমস্যা হয়েছে। পরে আবার চেষ্টা করুন।"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    session_id    = cl.user_session.get("id")
    question_text = message.content
    loop          = asyncio.get_running_loop()

    try:
        if hasattr(message, "elements") and message.elements:
            for element in message.elements:
                if hasattr(element, "mime") and element.mime and (
                    "audio" in element.mime or "video" in element.mime
                ):
                    await cl.Message(content="🎤 অডিও ট্রান্সক্রাইব করা হচ্ছে...").send()
                    try:
                        question_text = await transcribe_audio(element)
                        await cl.Message(content=f"**আপনার প্রশ্ন:**\n{question_text}").send()
                    except Exception as e:
                        import traceback
                        await cl.Message(
                            content=f"ASR Error: {e}\n\n```\n{traceback.format_exc()}\n```"
                        ).send()
                        return
                    break

        if not question_text or not question_text.strip():
            return

        if question_text.strip().lower() == "/warmup":
            await cl.Message(content="Model warmup শুরু হচ্ছে...").send()
            await warmup()
            await cl.Message(content="Warmup complete.").send()
            return

        answer, msg, streamed = await run_chat(question_text, session_id=session_id, reset=False)
        msg.content = answer

        msg.actions = [
            cl.Action(
                name="generate_speech",
                payload={"text": answer},
                label="🔊 শুনুন (Listen)",
                description="এই উত্তরটি অডিওতে শুনুন",
            )
        ]

        if streamed:
            await msg.update()
        else:
            await msg.send()

    except Exception as e:
        import traceback
        await cl.Message(
            content=f"❌ Error: {e}\n\nDetails:\n```\n{traceback.format_exc()}\n```"
        ).send()
