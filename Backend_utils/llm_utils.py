import asyncio
import inspect
import json
import os
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

import numpy as np
import regex as re
from openai import AsyncOpenAI

import comorbidity


# FIX: bumped from 300 to 500. Multi-intent JSON (cramps + diabetes etc.)
# was getting truncated mid-output at 300, which broke JSON parsing and
# triggered the out_of_scope fallback.
PLANNER_MAX_NEW_TOKENS = 500
PLANNER_TEMPERATURE = 0.0
PLANNER_TOP_P = 0.9

# FIX: safety net for when the planner over-zealously marks a comorbidity
# turn as out_of_scope. If the user message clearly contains a menstrual
# or menopausal keyword, we override the route to health_direct.
_MENSTRUAL_TRIGGERS = re.compile(
    r"period|menstr|cramp|pms|pcos|endometri|fibroid|menopaus|perimenopaus|"
    r"ovari|hormone|bleeding|spotting|"
    r"মাসিক|পিরিয়ড|ক্র্যাম্প|পেটে\s*ব্যথা|পেট\s*ব্যথা|হরমোন|রক্তস্রাব|"
    r"রজঃনিবৃত্তি|মেনোপজ|পেরিমেনোপজ|পিসিওএস|এন্ডোমেট্রিওসিস|ফাইব্রয়েড|তলপেট",
    re.IGNORECASE,
)

MULTIQUERY_MAX_NEW_TOKENS = 180
MULTIQUERY_TEMPERATURE = 0.0
MULTIQUERY_TOP_P = 0.9

RESPONDER_MAX_NEW_TOKENS = 500
RESPONDER_TEMPERATURE = 0.2
RESPONDER_TOP_P = 0.9


RETRIEVE_TOP_K = 20
ANSWER_BLOCK_MAX_CHARS = 1400
ANSWER_BLOCK_MAX_ITEMS = 5
RERANK_SCORE_FLOOR = 0.15
# Secondary intents (e.g. a comorbidity the user just mentions in passing) are
# scored against their own question but live in a corpus that is biased toward
# the primary domain (menstrual health). A slightly lower floor keeps them from
# being filtered out entirely when the user's main ask is something else.
RERANK_SCORE_FLOOR_SECONDARY = 0.12

# Intent decomposition: most turns are 1 intent, occasionally 2, rarely 3.
# Hard cap at 3 to keep latency bounded and prevent planner from over-spawning.
MAX_INTENTS = 3
PRIMARY_INTENT_BLOCK_QUOTA = 3
SECONDARY_INTENT_BLOCK_QUOTA = 2

# Single-intent turns keep the tighter token budget so we don't pay extra
# latency on the common case. Multi-intent turns get more room since 2-3
# conditions cannot be safely addressed in 3-5 compact lines.
RESPONDER_MAX_NEW_TOKENS_MULTI = 400

ALLOWED_RISK_LEVELS = {"none", "routine", "elevated", "urgent"}
ALLOWED_PLANNER_ROUTES = {
	"out_of_scope",
	"smalltalk",
	"health_direct",
	"health_followup",
	"health_education",
	"sensitive_supportive",
	"urgent_redflag",
}
RETRIEVAL_REQUIRED_ROUTES = {
	"health_direct",
	"health_followup",
	"health_education",
	"sensitive_supportive",
	"urgent_redflag",
}

BAD_INLINE_PATTERNS = [
	r"email\s+preview",
	r"privacy\s+practices?",
	r"thank you for subscribing",
	r"subscribe!?",
	r"newsletter",
	r"cookie",
	r"clinical trials",
	r"error",
	r"retry",
	r"click here",
]

DEFAULT_THREAD_STATE = {
	"active_topic": "",
	"symptom_cluster": [],
	"risk_level": "none",
	"emotion_flag": False,
	"last_resolved_question": "",
	"last_retrieval_query": "",
	"last_route": "",
	# Comorbidities/conditions the user has reported, accumulated across turns.
	# Was previously a module-level list (DISEASEs_USER), which leaked across
	# sessions in the same process. Now session-scoped.
	"known_user_conditions": [],
}

SESSIONS: dict[str, dict[str, Any]] = {}

# Cache planner decisions for the fixed Chainlit starter messages so the
# first turn is essentially free for those exact strings.
_STARTER_PLANNER_CACHE: dict[str, dict] = {}

# Cheap regex pre-filter for food2disease. If the answer text has no food
# signal, we skip the LLM call entirely. Saves ~1-2s on most turns where
# the user is asking about cramps/PCOS/etc with no food mentioned.
_BANGLA_FOOD_HINTS = re.compile(
	r"খাবার|খাও|খা[বয়]|খেতে|খেলে|পান\s|পানীয়|চা\b|দুধ|ভাত|মাছ|মাংস|ডিম|"
	r"ফল\b|শাক|সবজি|বাদাম|ডাল|তেল|মসলা|মিষ্টি|রুটি|আলু|বার্লি|ওটস|"
	r"food|drink|eat|meal|diet|fruit|vegetable|protein|carb"
)

MAX_HISTORY_TURNS = 3

_llm_client: AsyncOpenAI | None = None
_llm_model_name: str = "meno"

_faiss_index = None
_db_chunks = None
_db_meta = None
_db_uids = None
_embed_model = None
_reranker = None

# Tokenizer thread-safety locks.
#
# BGE-M3 and BGE-reranker both use HuggingFace's Rust `tokenizers` library
# under the hood. The Rust side holds mutable state via PyO3, so calling
# `_embed_model.encode(...)` (or `_reranker.compute_score(...)`) from two
# threads at once raises `RuntimeError('Already borrowed')`.
#
# Per-intent retrieval fans out N FAISS embed calls and M rerank calls
# concurrently through the asyncio thread executor, which trips this. The
# fix is to serialize ONLY the tokenizer-touching call inside each function;
# everything else (FAISS index search, numpy ops, the per-intent rerank
# running concurrently with another intent's FAISS) stays parallel.
#
# Two separate locks: embed and rerank can still overlap with each other,
# they just can't overlap with another instance of themselves.
_embed_lock = threading.Lock()
_rerank_lock = threading.Lock()

_session_storage_dir = Path("menochat_sessions")
_executor = None

# Where eval JSONL files live. One file per session day, so you can mine the
# model's reasoning later without grepping through thousands of tiny files.
_evals_storage_dir = Path("menochat_evals")


def _save_turn_eval(
	session_id: str,
	user_message: str,
	answer: str,
	plan: dict,
	thinking: dict,
	retrieval_debug: dict,
) -> None:
	"""Append one full turn (input + reasoning + output) to a JSONL file.

	Everything stays local. You can later mine these to spot reasoning
	failures, retrieval gaps, or weird planner choices, and add them to a
	fine-tune dataset.
	"""
	try:
		_evals_storage_dir.mkdir(parents=True, exist_ok=True)
		day = datetime.utcnow().strftime("%Y%m%d")
		safe_sid = "".join(c for c in str(session_id) if c.isalnum() or c in "-_") or "unnamed"
		path = _evals_storage_dir / f"{safe_sid}_{day}.jsonl"
		record = {
			"ts": _utc_now_iso(),
			"session_id": session_id,
			"user_message": user_message,
			"answer": answer,
			"plan": {
				"route": plan.get("route"),
				"risk_level": plan.get("risk_level"),
				"needs_retrieval": plan.get("needs_retrieval"),
				"resolved_question": plan.get("resolved_question"),
				"retrieval_query": plan.get("retrieval_query"),
				"answer_goal": plan.get("answer_goal"),
			},
			"thinking": thinking,
			"retrieval_debug": {
				"reason": retrieval_debug.get("reason"),
				"merged_count": retrieval_debug.get("merged_count"),
				"final_context_count": retrieval_debug.get("final_context_count"),
				"top_rerank_score": retrieval_debug.get("top_rerank_score"),
				"query_variants": retrieval_debug.get("query_variants"),
			},
		}
		with open(path, "a", encoding="utf-8") as f:
			f.write(json.dumps(record, ensure_ascii=False) + "\n")
	except Exception as e:
		print(f">>> EVAL SAVE FAILED: {repr(e)}")


def _log_weird_planner_or_retrieval(
	plan: dict,
	thinking: dict,
	retrieval_debug: dict,
) -> None:
	"""Print a noisy debug block when something looks off. Use 2 (debugging)."""
	route = plan.get("route", "")
	top_score = retrieval_debug.get("top_rerank_score")
	final_n = retrieval_debug.get("final_context_count", 0)
	needs_ret = plan.get("needs_retrieval", False)

	weird = []
	if route == "out_of_scope":
		weird.append("route=out_of_scope")
	if needs_ret and final_n == 0:
		weird.append("retrieval_returned_nothing")
	if isinstance(top_score, (int, float)) and top_score < 0.20 and final_n > 0:
		weird.append(f"low_top_rerank={top_score:.3f}")
	if plan.get("_parse_failed"):
		weird.append("planner_json_parse_failed")

	if not weird:
		return

	print("!" * 80)
	print(f"[debug] suspicious turn: {', '.join(weird)}")
	print(f"  route={route}  risk={plan.get('risk_level')}  needs_retrieval={needs_ret}")
	print(f"  resolved_question: {plan.get('resolved_question', '')[:200]}")
	planner_think = thinking.get("planner", "")
	if planner_think:
		print(f"  planner THINKING (first 500): {planner_think[:500]}")
	mq_think = thinking.get("multiquery", "")
	if mq_think:
		print(f"  multiquery THINKING (first 300): {mq_think[:300]}")
	print("!" * 80)


def _utc_now_iso() -> str:
	return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_strip(x: Any) -> str:
	if x is None:
		return ""
	return str(x).strip()


def _normalize_ws(text: str) -> str:
	return " ".join(_safe_strip(text).split())


def _norm_ws(text: str) -> str:
	return re.sub(r"\s+", " ", _safe_strip(text)).strip()


def _ensure_q(text: str) -> str:
	t = _normalize_ws(text)
	if not t:
		return t
	t = t.rstrip("।.!?।,; ")
	return t + "?"


def _extract_first_json_object(text: str) -> dict | None:
	if not text:
		return None
	start = text.find("{")
	if start == -1:
		return None
	depth = 0
	for i in range(start, len(text)):
		ch = text[i]
		if ch == "{":
			depth += 1
		elif ch == "}":
			depth -= 1
			if depth == 0:
				try:
					return json.loads(text[start : i + 1])
				except Exception:
					return None
	return None


def _session_file_path(session_id: str) -> Path:
	safe_id = "".join(c for c in str(session_id) if c.isalnum() or c in "-_")
	if not safe_id:
		safe_id = "unnamed"
	return _session_storage_dir / f"{safe_id}.json"


def save_session_to_disk(session_id: str) -> bool:
	if session_id not in SESSIONS:
		return False
	path = _session_file_path(session_id)
	try:
		path.parent.mkdir(parents=True, exist_ok=True)
		with open(path, "w", encoding="utf-8") as f:
			json.dump(SESSIONS[session_id], f, ensure_ascii=False, indent=2)
		return True
	except Exception as e:
		print(f">>> SESSION SAVE FAILED for {session_id}: {repr(e)}")
		return False


def load_session_from_disk(session_id: str) -> bool:
	path = _session_file_path(session_id)
	if not path.exists():
		return False
	try:
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
		if not isinstance(data, dict):
			return False
		data.setdefault("session_id", session_id)
		data.setdefault("created_at", _utc_now_iso())
		data.setdefault("history", [])
		data.setdefault("thread_state", deepcopy(DEFAULT_THREAD_STATE))
		data.setdefault("controller_last_decision", {})
		for k, v in DEFAULT_THREAD_STATE.items():
			data["thread_state"].setdefault(k, deepcopy(v))
		SESSIONS[session_id] = data
		return True
	except Exception as e:
		print(f">>> SESSION LOAD FAILED for {session_id}: {repr(e)}")
		return False


def list_saved_sessions() -> list[str]:
	if not _session_storage_dir.exists():
		return []
	return sorted(f.stem for f in _session_storage_dir.glob("*.json"))


def delete_session_file(session_id: str) -> bool:
	path = _session_file_path(session_id)
	if path.exists():
		try:
			path.unlink()
			return True
		except Exception as e:
			print(f">>> SESSION DELETE FAILED for {session_id}: {repr(e)}")
	return False


def get_session(session_id: str) -> dict:
	if session_id not in SESSIONS:
		if not load_session_from_disk(session_id):
			SESSIONS[session_id] = {
				"session_id": session_id,
				"created_at": _utc_now_iso(),
				"history": [],
				"thread_state": deepcopy(DEFAULT_THREAD_STATE),
				"controller_last_decision": {},
			}
	return SESSIONS[session_id]


def reset_session(session_id: str) -> dict:
	if session_id in SESSIONS:
		del SESSIONS[session_id]
	delete_session_file(session_id)
	return get_session(session_id)


def _trim_history(session_id: str) -> None:
	hist = get_session(session_id)["history"]
	max_entries = MAX_HISTORY_TURNS * 2
	if len(hist) > max_entries:
		del hist[: len(hist) - max_entries]


def add_user_message(session_id: str, text: str) -> None:
	get_session(session_id)["history"].append(
		{"role": "user", "content": str(text).strip(), "ts": _utc_now_iso()}
	)
	_trim_history(session_id)


def add_assistant_message(session_id: str, text: str) -> None:
	get_session(session_id)["history"].append(
		{"role": "assistant", "content": str(text).strip(), "ts": _utc_now_iso()}
	)
	_trim_history(session_id)


def delete_latest_user_message(session_id: str) -> bool:
	history = get_session(session_id)["history"]
	for i in range(len(history) - 1, -1, -1):
		if history[i].get("role") == "user":
			del history[i]
			return True
	return False


def _sanitize_thread_state_patch(patch: dict) -> dict:
	if not isinstance(patch, dict):
		return {}
	out = {}
	if "active_topic" in patch:
		v = patch.get("active_topic")
		out["active_topic"] = str(v).strip() if v is not None else ""
	if "symptom_cluster" in patch:
		v = patch.get("symptom_cluster")
		if isinstance(v, list):
			out["symptom_cluster"] = [str(x).strip() for x in v if _safe_strip(x)]
		else:
			out["symptom_cluster"] = []
	if "risk_level" in patch:
		v = str(patch.get("risk_level", "")).strip().lower()
		out["risk_level"] = v if v in ALLOWED_RISK_LEVELS else "none"
	if "emotion_flag" in patch:
		out["emotion_flag"] = bool(patch.get("emotion_flag"))
	if "last_resolved_question" in patch:
		v = patch.get("last_resolved_question")
		out["last_resolved_question"] = str(v).strip() if v is not None else ""
	if "last_retrieval_query" in patch:
		v = patch.get("last_retrieval_query")
		out["last_retrieval_query"] = str(v).strip() if v is not None else ""
	if "last_route" in patch:
		v = patch.get("last_route")
		out["last_route"] = str(v).strip() if v is not None else ""
	return out


def apply_thread_state_patch(session_id: str, patch: dict) -> None:
	sess = get_session(session_id)
	sess["thread_state"].update(_sanitize_thread_state_patch(patch))


def get_known_user_conditions(session_id: str) -> list[str]:
	"""Session-scoped list of comorbidities the user has reported.

	Reads from thread_state. Used by the responder so advice can account for
	conditions like diabetes, hypertension, anemia, etc., across turns —
	without leaking those across sessions (the old module-level DISEASEs_USER
	leaked).
	"""
	sess = get_session(session_id)
	conds = sess.get("thread_state", {}).get("known_user_conditions", []) or []
	# Defensive: thread_state may have been loaded from disk with a non-list.
	if not isinstance(conds, list):
		return []
	return [str(c).strip() for c in conds if _safe_strip(c)]


def add_known_user_conditions(session_id: str, new_conds: list[str]) -> list[str]:
	"""Merge new conditions into the session list (dedup, preserve order)."""
	if not new_conds:
		return get_known_user_conditions(session_id)
	sess = get_session(session_id)
	current = get_known_user_conditions(session_id)
	for c in new_conds:
		c = _safe_strip(c)
		if c and c not in current:
			current.append(c)
	sess["thread_state"]["known_user_conditions"] = current
	return current


def format_recent_history(session_id: str, max_turns: int = MAX_HISTORY_TURNS, assistant_max_chars: int = 200) -> str:
	sess = get_session(session_id)
	history = sess.get("history", [])
	recent = history[-(max_turns * 2) :]
	lines = []
	for entry in recent:
		role_raw = entry.get("role", "")
		content = str(entry.get("content", "")).strip()
		role = "User" if role_raw == "user" else "Assistant"
		if role_raw == "assistant" and len(content) > assistant_max_chars:
			clipped = content[:assistant_max_chars]
			cut = max(
				clipped.rfind("। "),
				clipped.rfind(". "),
				clipped.rfind("! "),
				clipped.rfind("? "),
			)
			if cut >= int(assistant_max_chars * 0.5):
				content = clipped[:cut + 1].strip() + " […]"
			else:
				content = clipped.strip() + " […]"
		lines.append(f"{role}: {content}")
	return "\n".join(lines) if lines else "(no history yet)"


def _require_init() -> None:
	if _llm_client is None:
		raise RuntimeError("LLM pipeline not initialized. Call init_llm_pipeline first.")
	if any(x is None for x in [_faiss_index, _db_chunks, _db_meta, _db_uids, _embed_model, _reranker]):
		raise RuntimeError("Vector resources are not initialized. Call init_llm_pipeline first.")


async def init_llm_pipeline(
	*,
	openai_base_url: str,
	openai_api_key: str,
	model_name: str,
	faiss_index,
	chunks,
	meta,
	uids,
	embed_model,
	reranker,
	session_storage_dir: str = "menochat_sessions",
	executor=None,
) -> None:
	global _llm_client, _llm_model_name
	global _faiss_index, _db_chunks, _db_meta, _db_uids, _embed_model, _reranker
	global _session_storage_dir, _executor

	base_url = _safe_strip(openai_base_url).rstrip("/")
	if not base_url:
		raise ValueError("openai_base_url is required")

	_llm_client = AsyncOpenAI(
		base_url=base_url,
		api_key=_safe_strip(openai_api_key) or "EMPTY",
	)
	_llm_model_name = _safe_strip(model_name) or "meno"

	_faiss_index = faiss_index
	_db_chunks = chunks
	_db_meta = meta
	_db_uids = uids
	_embed_model = embed_model
	_reranker = reranker
	_executor = executor

	_session_storage_dir = Path(session_storage_dir)
	_session_storage_dir.mkdir(parents=True, exist_ok=True)
	os.makedirs("menochat_diseases", exist_ok=True)


def llama_cpp_extra_body(repeat_penalty: float = 1.1, enable_thinking: bool = True) -> dict:
    """llama.cpp server-compatible generation parameters."""
    body = {"repeat_penalty": repeat_penalty}
    if not enable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


# Holds the most recent thinking/reasoning text per tag so callers (e.g. the
# Chainlit app) can pull it out and show it as a step, log it for debugging,
# or analyze decision quality later.
LAST_THINKING: dict[str, str] = {}


def _merge_content_and_reasoning(msg) -> tuple[str, str]:
	"""
	llama.cpp routes 'thinking' tokens into msg.reasoning_content and the
	final answer into msg.content. If generation gets cut off mid-thought,
	the whole thing can end up in reasoning_content with content empty.

	Returns (final_text_for_pipeline, reasoning_text).
	final_text is what JSON parsers / responder see.
	"""
	content = (getattr(msg, "content", "") or "").strip()
	reasoning = (getattr(msg, "reasoning_content", "") or "").strip()
	# If content has the answer, prefer it. Otherwise fall back to whatever
	# is in reasoning_content, so JSON buried inside thinking still parses.
	final = content if content else reasoning
	return final, reasoning


async def llm_chat(
	prompt: str,
	*,
	max_tokens: int,
	temperature: float,
	top_p: float,
	extra_body: dict | None = None,
	tag: str = "llm",
) -> str:
	"""LLM call with timing prints. Tag identifies which stage called us.

	Handles llama.cpp's reasoning_content / content split for thinking models.
	The latest reasoning text is stashed in LAST_THINKING[tag] so callers
	can show it in the UI or log it.
	"""
	_require_init()
	t0 = time.perf_counter()
	resp = await _llm_client.chat.completions.create(
		model=_llm_model_name,
		messages=[{"role": "user", "content": prompt}],
		max_tokens=max_tokens,
		temperature=temperature,
		top_p=top_p,
		extra_body=extra_body or {},

	)
	msg = resp.choices[0].message
	out, reasoning = _merge_content_and_reasoning(msg)
	LAST_THINKING[tag] = reasoning
	dt = time.perf_counter() - t0
	in_tok = resp.usage.prompt_tokens if resp.usage else 0
	out_tok = resp.usage.completion_tokens if resp.usage else 0
	tps = out_tok / max(dt, 0.01)
	has_think = "Y" if reasoning else "N"
	print(f"[llm_chat:{tag}] {dt:.2f}s | in={in_tok} out={out_tok} | think={has_think} | {tps:.1f} tok/s")
	return out


# =====================================================================
# PLANNER (separate call)
# =====================================================================

def build_planner_prompt(session_id: str, user_message: str) -> str:
	sess = get_session(session_id)
	history_text = format_recent_history(session_id, max_turns=6)
	thread_state = sess.get("thread_state", {})
	return f"""You are the router for a Bengali women's-health assistant. Output ONLY JSON.

SCOPE: menstrual + menopausal health (period, cramp, PCOS, endometriosis, fibroid, PMS, perimenopause, menopause).

HARD RULE ABOUT COMORBIDITIES (READ THIS BEFORE PICKING A ROUTE):
If the user mentions ANY menstrual or menopausal symptom in the message,
the route is NEVER out_of_scope, even if they also mention diabetes,
high blood pressure, thyroid, anemia, kidney issues, asthma, heart
problems, pregnancy, or any other condition. The other condition becomes
a SECONDARY intent. It modulates safe advice but does not kick the
message out.

Positive examples (these are IN-SCOPE):
- "I have cramps and diabetes"          -> health_direct, 2 intents (cramps PRIMARY, diabetes SECONDARY)
- "PCOS with high blood pressure"        -> health_direct, 2 intents (PCOS PRIMARY, BP SECONDARY)
- "মাসিক ব্যথা আর ডায়াবেটিস"             -> health_direct, 2 intents
- "পিরিয়ডে অনেক রক্ত যাচ্ছে আর আমি anemic" -> urgent_redflag or health_direct, 2 intents

Negative examples (these ARE out_of_scope, no menstrual word):
- "I have a fever"                       -> out_of_scope
- "diabetes management tips"             -> out_of_scope
- "back pain"                            -> out_of_scope

ROUTES:
- out_of_scope: not menstrual/menopausal
- smalltalk: greeting, thanks
- health_direct: clear in-scope question
- health_followup: in-scope follow-up needing prior context
- health_education: explain a concept
- sensitive_supportive: ONLY when user shows explicit fear/shame/distress
- urgent_redflag: heavy bleeding, fainting, severe weakness, chest pain, severe pain + fever

RULES:
- Plain pain or worry = health_direct, NOT sensitive_supportive
- When unsure between health_direct and sensitive_supportive, pick health_direct
- answer_goal = concrete instruction, neutral for yes/no
- needs_retrieval = true for all in-scope routes except smalltalk

INTENT DECOMPOSITION (intents array):
- If the user mentions ONE condition or asks ONE question, return ONE intent.
- If the user mentions multiple distinct conditions in the same turn
  (e.g. "cramps AND diabetes", "PCOS and high blood pressure"), return ONE
  intent PER condition, max {MAX_INTENTS}.
- Each intent must be self-contained — reading just that intent should be
  enough to retrieve for it (no pronouns referring to other intents).
- The FIRST intent must be the PRIMARY one (what the user is most directly
  asking about). Comorbidities the user just mentions in passing
  ("I also have X") are SECONDARY intents — they still belong in the list
  because they affect safe advice.
- resolved_question: ONE clean Bangla question for this intent, max 200 chars.
- retrieval_query: best Bangla/English query for vector search.

OUTPUT (JSON only, no other text):
{{
  "route": "out_of_scope|smalltalk|health_direct|health_followup|health_education|sensitive_supportive|urgent_redflag",
  "risk_level": "none|routine|elevated|urgent",
  "depends_on_history": true|false,
  "needs_retrieval": true|false,
  "intents": [
    {{
      "topic": "short tag e.g. 'cramps', 'diabetes'",
      "resolved_question": "...",
      "retrieval_query": "...",
      "is_primary": true|false
    }}
  ],
  "answer_goal": "...",
  "thread_state_patch": {{"active_topic": "", "risk_level": "", "last_resolved_question": "", "last_retrieval_query": "", "last_route": ""}}
}}

Recent conversation:
{history_text}

User: {user_message}
""".strip()


async def run_planner(session_id: str, user_message: str) -> dict:
	user_message = _normalize_ws(user_message)
	sess = get_session(session_id)
	prompt = build_planner_prompt(session_id, user_message)

	gen_text = await llm_chat(
		prompt,
		max_tokens=PLANNER_MAX_NEW_TOKENS,
		temperature=PLANNER_TEMPERATURE,
		top_p=PLANNER_TOP_P,
		extra_body=llama_cpp_extra_body(1.1, enable_thinking = False),
		tag="planner",
	)

	parsed = _extract_first_json_object(gen_text)
	print("%" * 100)
	print(parsed)
	if not isinstance(parsed, dict):
		print("RAW PLANNER OUTPUT:")
		print(gen_text)
		parsed = {
			"route": "out_of_scope",
			"risk_level": "none",
			"depends_on_history": False,
			"needs_retrieval": False,
			"resolved_question": user_message[:300],
			"retrieval_query": "",
			"answer_goal": "ask the user to restate the question clearly",
			"thread_state_patch": {},
			"_parse_failed": True,
			"_raw_output": gen_text,
		}

	route = _normalize_ws(parsed.get("route", "out_of_scope"))
	if route not in ALLOWED_PLANNER_ROUTES:
		route = "out_of_scope"

	# FIX: safety override. If the planner panicked and said out_of_scope
	# but the user clearly mentioned a menstrual/menopausal symptom, force
	# the route back to health_direct. Catches the small-model failure
	# mode where naming any comorbidity (diabetes, etc.) kicks the whole
	# turn out of scope.
	if route == "out_of_scope" and _MENSTRUAL_TRIGGERS.search(user_message or ""):
		print(
			f"[planner] OVERRIDE out_of_scope -> health_direct "
			f"(menstrual trigger found in: {user_message[:80]!r})"
		)
		route = "health_direct"

	risk_level = _normalize_ws(parsed.get("risk_level", "none")).lower()
	if risk_level not in ALLOWED_RISK_LEVELS:
		risk_level = "none"

	depends_on_history = bool(parsed.get("depends_on_history", False))
	needs_retrieval = bool(parsed.get("needs_retrieval", False))

	# --- Intent decomposition ---
	# New schema: parsed["intents"] is a list. Fall back to legacy single-intent
	# fields if the planner didn't emit it (older model, parse failure, etc.)
	# so behavior on single-intent turns is identical to before.
	raw_intents = parsed.get("intents") or []
	intents: list[dict] = []
	if isinstance(raw_intents, list):
		for it in raw_intents[:MAX_INTENTS]:
			if not isinstance(it, dict):
				continue
			rq = _normalize_ws(it.get("resolved_question", ""))[:300]
			rtq = _normalize_ws(it.get("retrieval_query", ""))[:300] or rq
			topic = _normalize_ws(it.get("topic", ""))[:60]
			if not rq:
				continue
			intents.append({
				"topic": topic or rq[:30],
				"resolved_question": rq,
				"retrieval_query": rtq,
				"is_primary": bool(it.get("is_primary", len(intents) == 0)),
			})
	# Backward-compat fallback.
	if not intents:
		legacy_rq = _normalize_ws(parsed.get("resolved_question", user_message))[:300]
		legacy_rtq = _normalize_ws(parsed.get("retrieval_query", ""))[:300] or legacy_rq
		intents = [{
			"topic": "main",
			"resolved_question": legacy_rq,
			"retrieval_query": legacy_rtq,
			"is_primary": True,
		}]
	# Guarantee exactly one primary intent (first one wins if multiple flagged).
	primary_found = False
	for i in intents:
		if i["is_primary"] and not primary_found:
			primary_found = True
		else:
			i["is_primary"] = False
	if not primary_found:
		intents[0]["is_primary"] = True

	primary = next(i for i in intents if i["is_primary"])
	resolved_question = primary["resolved_question"]
	retrieval_query = primary["retrieval_query"]
	answer_goal = _normalize_ws(parsed.get("answer_goal", "answer the user's question directly"))

	if route in RETRIEVAL_REQUIRED_ROUTES:
		needs_retrieval = True
	if route in {"smalltalk", "out_of_scope"}:
		needs_retrieval = False
		retrieval_query = ""

	llm_patch = parsed.get("thread_state_patch", {}) or {}
	# Surface any non-primary intent topics into known_user_conditions so future
	# turns remember them (e.g. user mentions diabetes once, future cramps
	# advice still factors it in).
	secondary_topics = [i["topic"] for i in intents if not i["is_primary"]]
	if secondary_topics:
		add_known_user_conditions(sess["session_id"], secondary_topics)

	thread_state_patch = _sanitize_thread_state_patch(
		{
			"active_topic": llm_patch.get("active_topic", "") or resolved_question[:100],
			"risk_level": risk_level,
			"last_resolved_question": resolved_question,
			"last_retrieval_query": retrieval_query,
			"last_route": route,
		}
	)

	out = {
		**parsed,
		"route": route,
		"risk_level": risk_level,
		"depends_on_history": depends_on_history,
		"needs_retrieval": needs_retrieval,
		"intents": intents,
		"resolved_question": resolved_question,  # primary, for back-compat
		"retrieval_query": retrieval_query,       # primary, for back-compat
		"answer_goal": answer_goal,
		"thread_state_patch": thread_state_patch,
	}

	# FIX: one-line debug print so terminal shows exactly what the planner
	# decided for each turn. Helpful for diagnosing routing failures like
	# "cramps + diabetes" getting kicked to out_of_scope.
	print(
		f"[planner] route={route} | risk={risk_level} | "
		f"needs_retrieval={needs_retrieval} | n_intents={len(intents)} | "
		f"topics={[i.get('topic', '') for i in intents]}"
	)

	sess["controller_last_decision"] = out
	return out


# =====================================================================
# MULTI-QUERY (separate call, kept on its own as you wanted)
# =====================================================================

def build_multiquery_prompt(session_id: str, user_message: str, intents: list[dict] | None = None) -> str:
	"""
	Multi-query prompt now generates queries PER INTENT. This lets us cover
	each condition the user mentioned (e.g. cramps AND diabetes) instead of
	collapsing into one topic.

	Depends only on user message + history + the planner's intent list, so
	this LLM call can still run IN PARALLEL with the planner — we just pass
	a single-intent fallback when intents aren't available yet.
	"""
	recent_history = format_recent_history(session_id, max_turns=6)
	intents = intents or [{"topic": "main", "resolved_question": user_message}]
	intents_block = json.dumps(
		[{"topic": i.get("topic", ""), "question": i.get("resolved_question", "")} for i in intents],
		ensure_ascii=False,
	)
	return f"""
You generate retrieval queries for a Bengali women's-health assistant.

Your ONLY job is to produce search queries for a vector database, ONE GROUP
PER INTENT. The intents are pre-decomposed for you. Do not remove USER HEALTH
INFORMATION from the paraphrased queries.

RULES:
- For EACH intent below, produce 1-2 queries focused on THAT intent's topic.
- Total queries across all intents: max 4.
- Within an intent, query 1 should be the best standalone Bangla version,
  query 2 (optional) can be an English/paraphrased version for retrieval.
- Keep every query self-contained — no pronouns referring to other intents.
- Do NOT merge intents into a single query — each query targets ONE intent.
- Do NOT answer the question.
- Do NOT explain anything.
- If the latest message is clearly out-of-scope or pure smalltalk, return
  an empty list for that intent.

Intents to cover:
{intents_block}

Return ONLY valid JSON:
{{
  "queries_by_intent": [
    {{"topic": "...", "queries": ["q1", "q2"]}}
  ]
}}

Recent conversation:
{recent_history}

Latest user message:
{user_message}
""".strip()


async def generate_multi_queries(
	session_id: str,
	user_message: str,
	max_queries: int = 4,
	intents: list[dict] | None = None,
) -> tuple[dict[str, list[str]], dict | None]:
	"""
	Generate retrieval queries per intent.

	Returns ({topic -> [queries]}, raw_parsed_json).
	On parse failure or older planner output, falls back to a single-intent
	dict keyed by the first intent's topic (or "main"), so callers that only
	want a flat list can still flatten the values.
	"""
	prompt = build_multiquery_prompt(session_id, user_message, intents=intents)
	gen_text = await llm_chat(
		prompt,
		max_tokens=MULTIQUERY_MAX_NEW_TOKENS,
		temperature=MULTIQUERY_TEMPERATURE,
		top_p=MULTIQUERY_TOP_P,
		extra_body=llama_cpp_extra_body(1.1, enable_thinking=False),
		tag="multiquery",
	)
	parsed = _extract_first_json_object(gen_text)

	queries_by_intent: dict[str, list[str]] = {}
	total = 0

	if isinstance(parsed, dict):
		raw_groups = parsed.get("queries_by_intent", []) or []
		# Be lenient: also accept legacy {"queries": [...]} shape.
		if not raw_groups and isinstance(parsed.get("queries"), list):
			legacy_topic = (intents[0].get("topic") if intents else "main") or "main"
			raw_groups = [{"topic": legacy_topic, "queries": parsed["queries"]}]
		for grp in raw_groups:
			if not isinstance(grp, dict):
				continue
			topic = _normalize_ws(grp.get("topic", "")) or "main"
			qs = []
			for q in grp.get("queries", []) or []:
				q = _normalize_ws(q)
				if q and q not in qs:
					qs.append(q)
				if total + len(qs) >= max_queries:
					break
			if qs:
				queries_by_intent.setdefault(topic, []).extend(qs)
				total += len(qs)
			if total >= max_queries:
				break

	# Always seed each intent with its own resolved_question/retrieval_query
	# so retrieval can't return empty just because the multi-query LLM failed.
	if intents:
		for it in intents:
			topic = it.get("topic", "main") or "main"
			seed_q = _normalize_ws(it.get("retrieval_query") or it.get("resolved_question") or "")
			if seed_q and seed_q not in queries_by_intent.get(topic, []):
				queries_by_intent.setdefault(topic, []).append(seed_q)

	return queries_by_intent, parsed if isinstance(parsed, dict) else None


def tell_this_is_beyond_chatbot_capacity() -> str:
	return (
		"দুঃখিত, এই প্রশ্নের উত্তর আমি দিতে পারছি না। "
		"আমি একজন মাসিক ও মেনোপজ স্বাস্থ্য সহকারী — শুধুমাত্র পিরিয়ড, "
		"মাসিক স্বাস্থ্য, হরমোনের পরিবর্তন, পেরিমেনোপজ, মেনোপজ, "
		"এবং এগুলোর সাথে সম্পর্কিত শারীরিক ও মানসিক বিষয় নিয়ে কথা বলতে পারি।\n\n"
		"আপনার যদি এই বিষয়গুলো নিয়ে কোনো প্রশ্ন থাকে, আমাকে নির্দ্বিধায় "
		"জিজ্ঞেস করুন — আমি আপনাকে সাহায্য করার জন্য এখানে আছি। 🌸"
	)


def _get_chunk_text(chunk_obj) -> str:
	if isinstance(chunk_obj, dict):
		return chunk_obj.get("text", "")
	if isinstance(chunk_obj, str):
		return chunk_obj
	return str(chunk_obj)


def _get_meta_dict(meta_obj) -> dict:
	if isinstance(meta_obj, dict):
		md = meta_obj.get("metadata", {})
		return md if isinstance(md, dict) else {}
	return {}


def _get_url(meta_obj) -> str:
	return _safe_strip(_get_meta_dict(meta_obj).get("website_url", ""))


def _get_domain(meta_obj) -> str:
	url = _get_url(meta_obj)
	try:
		return urlparse(url).netloc.lower()
	except Exception:
		return ""


def _remove_inline_junk(text: str) -> str:
	t = _norm_ws(text)
	for pat in BAD_INLINE_PATTERNS:
		t = re.sub(pat, " ", t, flags=re.IGNORECASE)
	return _norm_ws(t)


def _looks_too_junky(text: str) -> bool:
	t = _norm_ws(text).lower()
	if len(t) < 180:
		return True
	if t.count("|") >= 4:
		return True
	if any(re.search(p, t) for p in BAD_INLINE_PATTERNS):
		return True
	return False


def _trim_to_sentence_boundary(text: str, max_chars: int = ANSWER_BLOCK_MAX_CHARS) -> str:
	t = _norm_ws(text)
	if len(t) <= max_chars:
		return t
	clipped = t[:max_chars]
	cut = max(clipped.rfind("। "), clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
	if cut >= int(max_chars * 0.55):
		clipped = clipped[: cut + 1]
	return _norm_ws(clipped)


def _normalize_for_dedup(text: str) -> str:
	t = _norm_ws(text).lower()
	t = re.sub(r"[^\w\s]", " ", t)
	t = re.sub(r"\s+", " ", t).strip()
	return t


# =====================================================================
# RETRIEVAL: FAISS-only (cheap, parallelizable) + single rerank pass
# =====================================================================

def _faiss_only_search(query: str, top_k: int = RETRIEVE_TOP_K) -> list[dict]:
	"""FAISS only. No rerank. Cheap so we can run several in parallel.

	The `_embed_model.encode(...)` call is serialized via `_embed_lock`
	because BGE-M3's HF tokenizer is not thread-safe (PyO3 'Already borrowed'
	error on concurrent access). Everything else stays parallel-safe.
	"""
	with _embed_lock:
		q_emb = _embed_model.encode(
			[query],
			batch_size=1,
			max_length=512,
			return_dense=True,
			return_sparse=False,
			return_colbert_vecs=False,
		)["dense_vecs"]
	q = np.array(q_emb, dtype=np.float32)
	if q.ndim == 1:
		q = q.reshape(1, -1)
	# L2 normalize so cosine/IP search matches how the FAISS index was built
	norms = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
	q = (q / norms).astype("float32")
	scores, indices = _faiss_index.search(q, top_k)
	out = []
	for faiss_score, idx in zip(scores[0], indices[0]):
		if idx < 0:
			continue
		chunk_obj = _db_chunks[idx]
		meta_obj = _db_meta[idx]
		uid_obj = _db_uids[idx]
		out.append(
			{
				"idx": int(idx),
				"uid": uid_obj,
				"faiss_score": float(faiss_score),
				"text": _get_chunk_text(chunk_obj),
				"chunk": chunk_obj,
				"meta": meta_obj,
				"source_query": query,
			}
		)
	return out


async def faiss_only_search_async(query: str, top_k: int = RETRIEVE_TOP_K) -> list[dict]:
	loop = asyncio.get_running_loop()
	return await loop.run_in_executor(_executor, _faiss_only_search, query, top_k)


def _rerank_once(query: str, items: list[dict]) -> list[dict]:
	"""One rerank pass over the deduped union of all FAISS results.

	The `_reranker.compute_score(...)` call is serialized via `_rerank_lock`
	because the reranker's HF tokenizer is not thread-safe (PyO3 'Already
	borrowed' error on concurrent access). Sorting/scoring around it stays
	parallel-safe.
	"""
	if not items:
		return []
	t0 = time.perf_counter()
	pairs = [(query, it["text"]) for it in items]
	with _rerank_lock:
		scores = _reranker.compute_score(pairs, normalize=True)
	if isinstance(scores, (float, int)):
		scores = [float(scores)]
	else:
		scores = [float(x) for x in scores]
	for it, s in zip(items, scores):
		it["rerank_score"] = s
	sorted_items = sorted(items, key=lambda x: x["rerank_score"], reverse=True)
	dt = time.perf_counter() - t0
	_top5 = [round(float(it["rerank_score"]), 3) for it in sorted_items[:5]]
	print(f"[rerank] {dt:.2f}s | pairs={len(pairs)} | top5={_top5}")
	return sorted_items


async def rerank_once_async(query: str, items: list[dict]) -> list[dict]:
	loop = asyncio.get_running_loop()
	return await loop.run_in_executor(_executor, _rerank_once, query, items)


def build_larger_clean_context_blocks(
	results: list[dict],
	max_items: int = ANSWER_BLOCK_MAX_ITEMS,
	max_chars: int = ANSWER_BLOCK_MAX_CHARS,
	floor: float = RERANK_SCORE_FLOOR,
) -> list[dict]:
	blocks = []
	seen = set()
	for item in results:
		if float(item.get("rerank_score", -999)) < floor:
			continue
		text = item.get("text", "") or _get_chunk_text(item.get("chunk", ""))
		text = _remove_inline_junk(text)
		text = _trim_to_sentence_boundary(text, max_chars=max_chars)
		if _looks_too_junky(text):
			continue
		norm = _normalize_for_dedup(text)
		if norm in seen:
			continue
		seen.add(norm)
		meta_obj = item.get("meta", {})
		blocks.append(
			{
				"text": text,
				"url": _get_url(meta_obj),
				"domain": _get_domain(meta_obj),
				"rerank_score": float(item.get("rerank_score", -999)),
				"dataset": _get_meta_dict(meta_obj).get("dataset", "N/A"),
				"chunk_id": _get_meta_dict(meta_obj).get("chunk_id", "N/A"),
			}
		)
		if len(blocks) >= max_items:
			break
	return blocks


def format_larger_context_blocks_for_prompt(blocks: list[dict]) -> str:
	if not blocks:
		return "No reliable retrieval context available."
	parts = []
	for i, b in enumerate(blocks, 1):
		parts.append(f"[Source {i}]\n{b['text']}")
	return "\n\n" + ("\n\n" + "-" * 100 + "\n\n").join(parts)


# Friendly names for known health-info domains. Extend as you encounter more.
_DOMAIN_PRETTY_NAMES = {
	"unfpa.org": "UNFPA",
	"who.int": "WHO",
	"healthline.com": "Healthline",
	"mayoclinic.org": "Mayo Clinic",
	"webmd.com": "WebMD",
	"nih.gov": "NIH",
	"ncbi.nlm.nih.gov": "NIH",
	"nimh.nih.gov": "NIMH",
	"cdc.gov": "CDC",
	"nhs.uk": "NHS",
	"medlineplus.gov": "MedlinePlus",
	"acog.org": "ACOG",
	"plannedparenthood.org": "Planned Parenthood",
	"clevelandclinic.org": "Cleveland Clinic",
	"hopkinsmedicine.org": "Johns Hopkins",
	"verywellhealth.com": "Verywell Health",
	"medicalnewstoday.com": "Medical News Today",
	"news.llu.edu": "Loma Linda Health News",
	"my.clevelandclinic.org": "Cleveland Clinic",
	"emedicine.medscape.com": "Medscape",
	"medscape.com": "Medscape",
}


def _strip_url_fragment(url: str) -> str:
	"""Drop the #:~:text=... and any trailing junk so the link is short and clean."""
	if not url:
		return ""
	# Cut off the fragment entirely
	clean = url.split("#", 1)[0]
	# Strip stray quote/space junk often pasted along with URLs
	return clean.rstrip().rstrip('"').rstrip("'").rstrip(")")


def _pretty_source_label(url: str, fallback: str = "") -> str:
	"""Turn a URL into a human-readable label like 'UNFPA: Menstrual Health'."""
	if not url:
		return fallback or "source"
	try:
		# Always parse the fragment-stripped URL so :~:text= junk does not leak in
		clean_url = _strip_url_fragment(url)
		parsed = urlparse(clean_url)
		domain = (parsed.netloc or "").lower().replace("www.", "")
		pretty_domain = _DOMAIN_PRETTY_NAMES.get(domain, domain)
		# Title-case the last meaningful path segment
		path_parts = [p for p in (parsed.path or "").split("/") if p]
		topic = ""
		for seg in reversed(path_parts):
			if seg.isdigit() or len(seg) <= 2:
				continue
			topic = seg.replace("-", " ").replace("_", " ").strip()
			for ext in (".html", ".htm", ".php", ".aspx"):
				if topic.lower().endswith(ext):
					topic = topic[: -len(ext)]
			topic = topic.title()
			break
		if pretty_domain and topic:
			return f"{pretty_domain}: {topic}"
		return pretty_domain or topic or (fallback or "source")
	except Exception:
		return fallback or "source"


async def _retrieve_for_intent(
	intent: dict,
	mq_queries: list[str],
	prefetch_results: list[dict] | None,
	use_prefetch: bool,
) -> tuple[dict, list[dict]]:
	"""FAISS for this intent's queries (in parallel), merge with prefetch if
	primary, then rerank against THIS intent's resolved_question.

	Reranking per intent is the key fix: previously one global rerank against
	the primary question filtered out everything off-topic (e.g. diabetes
	chunks scored against "cramps"). Per-intent reranking lets each topic
	bring its own best chunks into the responder context.
	"""
	queries = [
		intent.get("retrieval_query", ""),
		intent.get("resolved_question", ""),
	]
	for q in mq_queries or []:
		if q and q not in queries:
			queries.append(q)
	queries = [q for q in dict.fromkeys(filter(None, queries))][:3]
	if not queries:
		return intent, []

	# 1) FAISS in parallel across this intent's queries
	faiss_lists = await asyncio.gather(*[faiss_only_search_async(q) for q in queries])

	# 2) Merge with prefetch (only the primary intent gets prefetch, since
	#    prefetch was a FAISS search on the raw user message and may be biased
	#    toward the primary topic). Dedupe by chunk idx.
	merged: list[dict] = []
	seen_idx: set = set()
	if use_prefetch and prefetch_results:
		for r in prefetch_results:
			if r["idx"] not in seen_idx:
				seen_idx.add(r["idx"])
				merged.append(r)
	for results in faiss_lists:
		for r in results:
			if r["idx"] not in seen_idx:
				seen_idx.add(r["idx"])
				merged.append(r)

	if not merged:
		return intent, []

	# 3) Rerank against THIS intent's resolved_question
	rerank_query = intent.get("resolved_question") or queries[0]
	reranked = await rerank_once_async(rerank_query, merged)
	return intent, reranked


async def maybe_run_retrieval_from_plan(
	plan: dict,
	user_message: str | None = None,
	session_id: str | None = None,
	prefetch_results: list[dict] | None = None,
	queries_by_intent: dict[str, list[str]] | None = None,
	mq_parsed: dict | None = None,
) -> list[dict] | str:
	"""
	Per-intent retrieval pipeline.

	Steps:
	  1) For EACH intent, run FAISS in parallel over its queries.
	  2) Merge with prefetch (only on the primary intent — prefetch was a
	     FAISS search on the raw user message and is naturally biased toward
	     the primary topic).
	  3) Rerank EACH intent's pool against THAT intent's resolved_question.
	  4) Quota-merge: primary intent gets PRIMARY_INTENT_BLOCK_QUOTA blocks,
	     each secondary gets SECONDARY_INTENT_BLOCK_QUOTA. Secondaries use a
	     slightly lower rerank floor.

	Returns a flat list of context blocks (for callers that consume the old
	contract) AND populates plan["_context_blocks_by_intent"] for the
	responder to render grouped context.
	"""
	route = plan.get("route", "")
	needs_retrieval = bool(plan.get("needs_retrieval", False))

	if route == "out_of_scope":
		plan["_retrieval_debug"] = {"reason": "planner_marked_out_of_scope"}
		return tell_this_is_beyond_chatbot_capacity()

	if not needs_retrieval:
		plan["_retrieval_debug"] = {"reason": "planner_marked_no_retrieval"}
		plan["_context_blocks_by_intent"] = []
		return []

	intents = plan.get("intents") or []
	if not intents:
		# Should never happen with the new planner, but be defensive.
		intents = [{
			"topic": "main",
			"resolved_question": plan.get("resolved_question", "") or (user_message or ""),
			"retrieval_query": plan.get("retrieval_query", "") or (user_message or ""),
			"is_primary": True,
		}]

	queries_by_intent = queries_by_intent or {}

	t0 = time.perf_counter()

	# Per-intent retrieval in parallel (each intent itself does parallel FAISS).
	intent_tasks = []
	for it in intents:
		mq_queries = queries_by_intent.get(it.get("topic", ""), [])
		intent_tasks.append(
			_retrieve_for_intent(
				it,
				mq_queries,
				prefetch_results,
				use_prefetch=bool(it.get("is_primary")),
			)
		)
	intent_results = await asyncio.gather(*intent_tasks)

	faiss_dt = time.perf_counter() - t0
	print(
		f"[retrieval] per-intent: {faiss_dt:.2f}s | "
		f"intents={len(intents)} | "
		f"pools={[len(rr) for _, rr in intent_results]}"
	)

	# Build blocks per intent with quotas and floors.
	context_blocks_by_intent: list[dict] = []
	flat_blocks: list[dict] = []
	flat_seen = set()
	for it, reranked in intent_results:
		is_primary = bool(it.get("is_primary"))
		quota = PRIMARY_INTENT_BLOCK_QUOTA if is_primary else SECONDARY_INTENT_BLOCK_QUOTA
		floor = RERANK_SCORE_FLOOR if is_primary else RERANK_SCORE_FLOOR_SECONDARY

		blocks = build_larger_clean_context_blocks(
			reranked,
			max_items=quota,
			max_chars=ANSWER_BLOCK_MAX_CHARS,
			floor=floor,
		)
		context_blocks_by_intent.append({"intent": it, "blocks": blocks})

		# Flatten with cross-intent dedup so the legacy flat list doesn't
		# repeat the same chunk twice.
		for b in blocks:
			key = _normalize_for_dedup(b.get("text", ""))
			if key and key not in flat_seen:
				flat_seen.add(key)
				flat_blocks.append(b)

	plan["_context_blocks_by_intent"] = context_blocks_by_intent
	plan["_retrieval_debug"] = {
		"reason": "retrieval_ran",
		"llm_multiquery_output": mq_parsed if isinstance(mq_parsed, dict) else {},
		"intents": [
			{
				"topic": grp["intent"].get("topic", ""),
				"is_primary": grp["intent"].get("is_primary", False),
				"n_blocks": len(grp["blocks"]),
				"top_rerank": grp["blocks"][0]["rerank_score"] if grp["blocks"] else None,
			}
			for grp in context_blocks_by_intent
		],
		"merged_count": sum(len(rr) for _, rr in intent_results),
		"final_context_count": len(flat_blocks),
		"top_rerank_score": (
			max((grp["blocks"][0]["rerank_score"] for grp in context_blocks_by_intent if grp["blocks"]),
				default=None)
		),
		"query_variants": [q for qs in queries_by_intent.values() for q in qs],
	}
	# Cap the flat list at ANSWER_BLOCK_MAX_ITEMS for callers that still rely
	# on the old size envelope.
	return flat_blocks[:ANSWER_BLOCK_MAX_ITEMS + SECONDARY_INTENT_BLOCK_QUOTA * (MAX_INTENTS - 1)]


def format_grouped_context_for_prompt(groups: list[dict]) -> str:
	"""Render retrieved context grouped by intent so the responder can
	clearly see which chunks belong to which user-mentioned condition.

	Falls back to a flat 'No reliable retrieval context available.' when
	nothing came back."""
	if not groups:
		return "No reliable retrieval context available."
	any_blocks = any(g.get("blocks") for g in groups)
	if not any_blocks:
		return "No reliable retrieval context available."

	parts = []
	for g in groups:
		intent = g.get("intent", {})
		topic = intent.get("topic", "main")
		rq = intent.get("resolved_question", "")
		is_primary = intent.get("is_primary", False)
		tag = "PRIMARY" if is_primary else "SECONDARY"
		blocks = g.get("blocks") or []
		header = f"[{tag} | Topic: {topic} | Question: {rq}]"
		if not blocks:
			parts.append(f"{header}\n(no reliable context found for this intent — say so honestly)")
			continue
		block_text = "\n\n".join(
			f"-- Source {i + 1} --\n{b['text']}" for i, b in enumerate(blocks)
		)
		parts.append(f"{header}\n{block_text}")
	separator = "\n\n" + ("=" * 60) + "\n\n"
	return "\n\n" + separator.join(parts)


def build_responder_prompt(
	session_id: str,
	user_message: str,
	plan: dict,
	context_blocks: list[dict],
) -> str:
	sess = get_session(session_id)
	recent_history = format_recent_history(session_id, max_turns=6)
	thread_state = sess.get("thread_state", {})
	# Prefer grouped (per-intent) context if available; fall back to flat.
	groups = plan.get("_context_blocks_by_intent") or []
	if groups:
		retrieval_context = format_grouped_context_for_prompt(groups)
	else:
		retrieval_context = format_larger_context_blocks_for_prompt(context_blocks)

	route = plan.get("route", "health_direct")
	risk_level = plan.get("risk_level", "routine")
	intents = plan.get("intents") or []
	multi_intent = len(intents) > 1

	route_instructions = ""
	if route == "urgent_redflag" or risk_level == "urgent":
		route_instructions = """
জরুরি পরিস্থিতি।
- প্রথম লাইনেই স্পষ্টভাবে বলবে যে এটি জরুরি।
- এখনই হাসপাতালে বা জরুরি বিভাগে যেতে বলবে।
- ঘরোয়া উপায়, দীর্ঘ ব্যাখ্যা, বা দেরি করার পরামর্শ দেবে না।
- খুব সংক্ষিপ্তভাবে বলবে কেন এটি জরুরি।
"""
	elif route == "sensitive_supportive":
		route_instructions = """
সংবেদনশীল/আবেগপূর্ণ পরিস্থিতি।
- প্রথমে উষ্ণভাবে validation দাও।
- তারপর শান্ত, সহজ, তথ্যভিত্তিক ব্যাখ্যা দাও।
- reassurance দেবে, কিন্তু বানানো আশ্বাস দেবে না।
"""

	# Conditional length cap: 3-5 lines is fine for single-intent turns, but
	# two or three conditions can't be addressed safely in that space.
	if multi_intent:
		length_rule = (
			"- প্রতিটি intent-কে আলাদা ছোট অনুচ্ছেদে সম্বোধন করো, মোট ৬-৯ লাইনের মধ্যে রাখো।"
		)
		multi_intent_block = """
একাধিক সমস্যার ক্ষেত্রে (MULTI-INTENT):
- ব্যবহারকারী একই টার্নে একাধিক সমস্যা উল্লেখ করেছেন (যেমন ক্র্যাম্প + ডায়াবেটিস)।
  প্রতিটি সমস্যাকে আলাদাভাবে স্বীকার ও সম্বোধন করতে হবে — কোনোটিকে উপেক্ষা করবে না।
- প্রাথমিক (PRIMARY) সমস্যাকে আগে বিস্তারিতভাবে উত্তর দাও।
- তারপর সহ-অবস্থা (SECONDARY) কীভাবে প্রাথমিক পরামর্শকে প্রভাবিত করছে সেটি সংক্ষেপে
  বলো — যেমন ডায়াবেটিস থাকলে কোন ঘরোয়া উপায় নিরাপদ নয়, কী এড়িয়ে চলতে হবে।
- কোনো intent-এর জন্য নির্ভরযোগ্য context না থাকলে সৎভাবে বলো যে সেই বিষয়ে
  নির্দিষ্ট তথ্য পাওয়া যায়নি, ডাক্তারের পরামর্শ নিতে বলো।
"""
	else:
		length_rule = "- make your answer generation maximum three to five compact lines."
		multi_intent_block = ""

	intents_summary = (
		json.dumps(
			[{"topic": i.get("topic", ""),
			  "resolved_question": i.get("resolved_question", ""),
			  "is_primary": i.get("is_primary", False)} for i in intents],
			ensure_ascii=False,
		)
		if intents else "[]"
	)
	known_conditions = get_known_user_conditions(session_id)

	return f"""
তুমি একজন সহানুভূতিশীল, তথ্যনির্ভর বাংলা স্বাস্থ্য-সহকারী। Make the answers concise

{route_instructions}
{multi_intent_block}

তোমার উত্তর:
- উত্তর ১০০% বাংলায় হবে। কোনো ইংরেজি শব্দ, transliteration, বা latin script ব্যবহার করবে না।
{length_rule}
- কোনো ওষুধের নাম (brand বা generic) কখনো উল্লেখ করবে না। ওষুধের প্রয়োজন হলে শুধু "ডাক্তারের পরামর্শ নিন" বলবে।
- সহজ, paragraph-based, সম্মানজনক, বাংলাদেশি ব্যবহারকারীর জন্য উপযোগী
- প্রথমে সরাসরি উত্তর, তারপর সংক্ষিপ্ত ব্যাখ্যা
- make the asnwers concise
- follow-up হলে আগের context ব্যবহার করবে, পুরোনো উত্তর repeat করবে না
- retrieved context-এর তথ্য নিজের ভাষায় ব্যবহার করবে, কপি করবে না
- thin, vague, repetitive উত্তর দেবে না
- metadata, id, score, source label, chunk dump কিছুই output করবে না
- "PRIMARY", "SECONDARY", "Topic:" — এসব internal label কখনো উত্তরে লিখবে না

Grounding (hallucination prevention):
শুধু (ক) ব্যবহারকারী এই conversation-এ যা বলেছেন, বা (খ) retrieved context-এ যা স্পষ্টভাবে আছে — সেগুলোই ব্যবহার করবে। নিজের অন্য medical knowledge যোগ করবে না।

"কি করব / এখন কী করা উচিত" জিজ্ঞেস করলে আগে action-first practical answer, তারপর ব্যাখ্যা।

Context-এ নির্দিষ্ট তথ্য না থাকলে সৎভাবে বলো যে এই বিষয়ে নির্ভরযোগ্য তথ্য নেই, ডাক্তারের পরামর্শের কথা বলো।

Warning sign থাকলে পরিষ্কারভাবে বলো, অযথা ভয় ধরাবে না, কিন্তু জরুরি হলে স্পষ্ট সতর্কতা দেবে।

Latest user message:
{user_message}

Route: {route} | Risk: {risk_level}
Intents (for your reference, do not echo these labels):
{intents_summary}
Answer goal: {plan.get("answer_goal", "")}
Known user conditions across session: {known_conditions}
Thread state: {json.dumps(thread_state, ensure_ascii=False)}

Recent conversation:
{recent_history}

Retrieved context (grouped by intent):
{retrieval_context}
""".strip()


async def _stream_answer(
	prompt: str,
	stream_callback: Optional[Callable[[str], Awaitable[None] | None]] = None,
	thinking_callback: Optional[Callable[[str], Awaitable[None] | None]] = None,
	max_tokens: int = RESPONDER_MAX_NEW_TOKENS,
) -> str:
	"""Stream the responder output, splitting thinking and final-answer tokens.

	- stream_callback receives the user-facing answer tokens (delta.content).
	- thinking_callback (optional) receives the model's reasoning tokens
	  (delta.reasoning_content). Use this in Chainlit to show a "thinking"
	  step that updates live while the model reasons.

	If llama-server is run with --reasoning-format none, everything will come
	through as delta.content and thinking_callback simply never fires. So this
	wrapper is safe either way.

	The full reasoning string is also stashed in LAST_THINKING['responder'].
	"""
	_require_init()
	t0 = time.perf_counter()
	full_response = ""
	full_thinking = ""
	first_token_t = None
	stream = await _llm_client.chat.completions.create(
		model=_llm_model_name,
		messages=[{"role": "user", "content": prompt}],
		stream=True,
		max_tokens=max_tokens,
		temperature=RESPONDER_TEMPERATURE,
		top_p=RESPONDER_TOP_P,
		extra_body=llama_cpp_extra_body(1.1, enable_thinking = False),
	)
	async for chunk in stream:
		if not chunk.choices:
			continue
		delta_obj = chunk.choices[0].delta
		text_delta = getattr(delta_obj, "content", None)
		think_delta = getattr(delta_obj, "reasoning_content", None)

		if think_delta:
			if first_token_t is None:
				first_token_t = time.perf_counter() - t0
			full_thinking += think_delta
			if thinking_callback:
				maybe = thinking_callback(think_delta)
				if inspect.isawaitable(maybe):
					await maybe

		if text_delta:
			if first_token_t is None:
				first_token_t = time.perf_counter() - t0
			full_response += text_delta
			if stream_callback:
				maybe = stream_callback(text_delta)
				if inspect.isawaitable(maybe):
					await maybe

	LAST_THINKING["responder"] = full_thinking.strip()

	# If content never arrived but reasoning did, the model never escaped
	# its thinking phase. Surface the thinking as the answer so the pipeline
	# does not silently die.
	if not full_response.strip() and full_thinking.strip():
		full_response = full_thinking
		if stream_callback:
			# Flush the recovered text so the UI shows something.
			maybe = stream_callback(full_thinking)
			if inspect.isawaitable(maybe):
				await maybe

	dt = time.perf_counter() - t0
	approx_tok = (len(full_response) + len(full_thinking)) // 3
	tps = approx_tok / max(dt, 0.01)
	ttft = first_token_t if first_token_t is not None else 0.0
	has_think = "Y" if full_thinking else "N"
	print(f"[llm_chat:responder_stream] total={dt:.2f}s | TTFT={ttft:.2f}s | think={has_think} | ~{approx_tok} tok | ~{tps:.1f} tok/s")
	return full_response.strip()


async def generate_answer(
	session_id: str,
	user_message: str,
	plan: dict,
	context_blocks: list[dict],
	*,
	stream: bool = False,
	stream_callback: Optional[Callable[[str], Awaitable[None] | None]] = None,
	thinking_callback: Optional[Callable[[str], Awaitable[None] | None]] = None,
) -> str:
	prompt = build_responder_prompt(session_id, user_message, plan, context_blocks)
	# Multi-intent turns need more room; single-intent keep the tighter budget
	# so we don't pay extra latency on the common case.
	n_intents = len(plan.get("intents") or [])
	max_tokens = RESPONDER_MAX_NEW_TOKENS_MULTI if n_intents > 1 else RESPONDER_MAX_NEW_TOKENS
	if stream:
		return await _stream_answer(
			prompt,
			stream_callback=stream_callback,
			thinking_callback=thinking_callback,
			max_tokens=max_tokens,
		)
	return await llm_chat(
		prompt,
		max_tokens=max_tokens,
		temperature=RESPONDER_TEMPERATURE,
		top_p=RESPONDER_TOP_P,
		extra_body=llama_cpp_extra_body(1.1, enable_thinking = False),
		tag="responder",
	)


async def food2disease_como(user_message: str, session_id: str | None = None) -> str:
	# Cheap pre-filter: if the answer mentions no food at all, skip the LLM
	# call entirely. Most turns about cramps/PCOS/etc never mention food.
	if not _BANGLA_FOOD_HINTS.search(user_message or ""):
		print(f"[food2disease] skipped (no food signal)")
		return ""

	prompt = f"""
You are a food entity extractor specialized in Bangla cuisine and ingredients.

TASK:
1. Identify every specific FOOD, DRINK, or INGREDIENT mentioned in the User Message.
   - DO NOT include: medical procedures (গরম সেঁক, হিটিং প্যাড), exercises (ব্যায়াম),
     supplements/nutrients (ভিটামিন ডি, ম্যাগনেসিয়াম, ক্যালসিয়াম), cooking methods (সেদ্ধ alone).
   - Only real edible items.

2. For each identified item, generate 2-5 variations that help match a database.
   IMPORTANT: If the user says a GENERIC term, include SPECIFIC sub-types in variations.
   - "বাদাম" -> ["বাদাম", "কাঠবাদাম", "কাজু বাদাম", "পেস্তা বাদাম", "আখরোট"]
   - "মাছ" -> ["মাছ", "রুই", "ইলিশ", "পাঙ্গাস"]
   - "ডাল" -> ["ডাল", "মসুর ডাল", "মুগ ডাল", "ছোলার ডাল"]
   - "ভাত" -> ["ভাত", "সাদা ভাত", "লাল চাল ভাত", "ব্রাউন রাইস"]
   - "শাকসবজি" -> ["শাকসবজি", "পালং শাক", "লাউ", "টমেটো", "গাজর"]
   - "ফল" -> ["ফল", "আপেল", "কলা", "পেয়ারা", "পেঁপে"]
   - "পুদিনা চা" -> ["চা", "দুধ চা", "গ্রিন টি", "লিকার চা"]
   - "দুধ" -> ["দুধ", "গরুর দুধ", "বাদাম দুধ", "সয়া দুধ"]

3. If the SAME generic term is mentioned, keep it in the category form.
   The FIRST variation should always be the original term so category lookup works.

4. Also detect any disease the user is discussing (diabetes, high pressure, heavy bleeding,
   PCOS, period pain, menopause, anemia, etc.). Use Bangla names.

GUIDELINES:
- Bangla script only. No transliteration.
- Do NOT answer the question.
- Do NOT explain anything.
- If a term is not food, do NOT include it in detected_items.

Latest user message:
{user_message}

Return ONLY valid JSON:
{{
  "contains_food": "YES" | "NO",
  "detected_items": [
    {{
      "original": "original word as user said",
      "variations": ["original", "specific1", "specific2", "specific3"]
    }}
  ],
  "detected_disease": ["disease1", "disease2"]
}}
""".strip()

	raw = await llm_chat(
		prompt,
		max_tokens=1500,
		temperature=RESPONDER_TEMPERATURE,
		top_p=RESPONDER_TOP_P,
        extra_body=llama_cpp_extra_body(1.1, enable_thinking = False),
		tag="food2disease",
	)

	answer = _extract_first_json_object(raw)
	print(answer)
	if not isinstance(answer, dict):
		answer = {
			"contains_food": "NO",
			"detected_items": [],
			"detected_disease": [],
		}

	if str(answer.get("contains_food", "NO")).casefold() == "no":
		return ""
	else:
		print("Do not")
	print(answer)

	results = []
	for item in answer.get("detected_items", []):
		original = item.get("original", "")
		variations = item.get("variations", [])

		matched = original
		warnings = comorbidity.get_warnings(original)
		is_category = False

		if not warnings:
			for v in variations:
				w = comorbidity.get_warnings(v)
				if w:
					warnings, matched = w, v
					break

		if not warnings:
			w = comorbidity.get_warnings_by_category(original)
			if w:
				warnings, matched, is_category = w, original, True

		if not warnings:
			for v in variations:
				w = comorbidity.get_warnings_by_category(v)
				if w:
					warnings, matched, is_category = w, v, True
					break

		if warnings:
			diseases = ", ".join(warnings[:6])
			if is_category:
				label = f"{original} (সার্বিক)"
			elif matched != original:
				label = f"{original} ({matched})"
			else:
				label = original
			results.append(f"⚠️ {label} — {diseases} থাকলে এড়িয়ে চলুন")

	print(results)

	# Session-scoped condition tracking (replaces the old module-level
	# DISEASEs_USER list which leaked across sessions in the same process).
	detected_diseases = [
		_safe_strip(d) for d in (answer.get("detected_disease") or []) if _safe_strip(d)
	]
	if session_id and detected_diseases:
		add_known_user_conditions(session_id, detected_diseases)
		# Also persist to disk for backward compatibility with anything that
		# reads the file (e.g. external scripts).
		try:
			os.makedirs("menochat_diseases", exist_ok=True)
			safe_sid = "".join(c for c in str(session_id) if c.isalnum() or c in "-_") or "unnamed"
			file_path = os.path.join("menochat_diseases", f"diseases_{safe_sid}.txt")
			conds = get_known_user_conditions(session_id)
			with open(file_path, "w", encoding="utf-8") as f:
				for d in conds:
					f.write(d + "\n")
		except Exception as e:
			print(f">>> diseases file write failed: {repr(e)}")

	return "\nগুরুত্বপূর্ণ তথ্য:\n" + "\n".join(results) if results else ""


# =====================================================================
# MAIN TURN: planner + parallel FAISS prefetch, then retrieval, then answer
# =====================================================================

async def run_chat_turn(
	session_id: str,
	user_message: str,
	*,
	debug: bool = False,
	stream: bool = False,
	stream_callback: Optional[Callable[[str], Awaitable[None] | None]] = None,
	thinking_callback: Optional[Callable[[str], Awaitable[None] | None]] = None,
	progress_callback: Optional[Callable[[str, dict], Awaitable[None] | None]] = None,
) -> dict[str, Any]:
	_require_init()

	turn_t0 = time.perf_counter()

	async def _fire(stage: str, data: dict | None = None) -> None:
		if progress_callback is None:
			return
		try:
			ret = progress_callback(stage, data or {})
			if inspect.isawaitable(ret):
				await ret
		except Exception as e:
			if debug:
				print(f"progress_callback raised at stage={stage}: {repr(e)}")

	user_message = _normalize_ws(user_message)
	if not user_message:
		return {
			"answer_text": "দুঃখিত, কিছু লেখা পাওয়া যায়নি। আবার বলুন।",
			"plan": {},
			"retrieval_debug": {},
			"context_blocks": [],
		}

	add_user_message(session_id, user_message)

	# --- Three things start in parallel: planner LLM, multi-query LLM, FAISS prefetch ---
	#
	# Wall-clock = max(planner_time, multiquery_seed_time, prefetch_time).
	#
	# Note: the multi-query call below is fired WITHOUT intents (planner is
	# still running). After the planner finishes, if we got multiple intents,
	# we fire a second, intent-aware multi-query call to fan out per-intent
	# queries. For single-intent turns (the common case), the first call's
	# output is used directly — no extra latency.
	await _fire("planner_start", {"user_message": user_message})

	prefetch_task: Optional[asyncio.Task] = asyncio.create_task(
		faiss_only_search_async(user_message)
	)
	planner_task = asyncio.create_task(run_planner(session_id, user_message))
	multiquery_task = asyncio.create_task(
		generate_multi_queries(session_id, user_message, max_queries=4, intents=None)
	)

	try:
		plan = await planner_task
	except Exception as e:
		# Cancel siblings to free the GPU/event loop quickly.
		for t in (prefetch_task, multiquery_task):
			if t and not t.done():
				t.cancel()
		delete_latest_user_message(session_id)
		await _fire("error", {"phase": "planner", "error": repr(e)})
		if debug:
			print("PLANNER FAILED:")
			print(repr(e))
		return {
			"answer_text": "দুঃখিত, এই টার্নটি পরিকল্পনা করতে পারিনি। আবার বলুন।",
			"plan": {"_planner_error": repr(e)},
			"retrieval_debug": {},
			"context_blocks": [],
		}

	await _fire("planner_done", {
		"route": plan.get("route", ""),
		"risk_level": plan.get("risk_level", ""),
		"resolved_question": plan.get("resolved_question", ""),
		"needs_retrieval": bool(plan.get("needs_retrieval", False)),
		"n_intents": len(plan.get("intents") or []),
	})

	# Collect multi-query and prefetch results IF we still need retrieval.
	if plan.get("needs_retrieval") and plan.get("route") != "out_of_scope":
		try:
			prefetch_results = await prefetch_task
		except Exception:
			prefetch_results = []
		try:
			queries_by_intent, mq_parsed = await multiquery_task
		except Exception:
			queries_by_intent, mq_parsed = {}, None

		# If we have multiple intents but the first multi-query call had no
		# intent awareness, fire a second pass so each intent gets dedicated
		# queries. Single-intent turns skip this — no extra latency.
		intents = plan.get("intents") or []
		if len(intents) > 1:
			try:
				queries_by_intent_v2, mq_parsed_v2 = await generate_multi_queries(
					session_id, user_message, max_queries=4, intents=intents,
				)
				# Merge v1 into v2 if v2 missed any intent (defensive).
				for topic, qs in (queries_by_intent or {}).items():
					queries_by_intent_v2.setdefault(topic, [])
					for q in qs:
						if q not in queries_by_intent_v2[topic]:
							queries_by_intent_v2[topic].append(q)
				queries_by_intent, mq_parsed = queries_by_intent_v2, mq_parsed_v2
			except Exception:
				# Stick with the v1 (non-intent-aware) queries.
				pass
	else:
		# Cancel both siblings since we won't use them.
		for t in (prefetch_task, multiquery_task):
			if t and not t.done():
				t.cancel()
		prefetch_results = []
		queries_by_intent, mq_parsed = {}, None

	# --- Retrieval (per-intent FAISS + per-intent rerank, all in parallel) ---
	await _fire("retrieval_start", {
		"needs_retrieval": bool(plan.get("needs_retrieval", False)),
		"route": plan.get("route", ""),
		"n_intents": len(plan.get("intents") or []),
	})
	try:
		context_blocks = await maybe_run_retrieval_from_plan(
			plan,
			user_message,
			session_id,
			prefetch_results=prefetch_results,
			queries_by_intent=queries_by_intent,
			mq_parsed=mq_parsed,
		)
	except Exception as e:
		delete_latest_user_message(session_id)
		await _fire("error", {"phase": "retrieval", "error": repr(e)})
		if debug:
			print("RETRIEVAL FAILED:")
			print(repr(e))
		return {
			"answer_text": "দুঃখিত, দরকারি তথ্য খুঁজতে গিয়ে সমস্যা হয়েছে। আবার বলুন।",
			"plan": plan,
			"retrieval_debug": {"reason": "retrieval_exception", "error": repr(e)},
			"context_blocks": [],
		}

	if isinstance(context_blocks, str):
		delete_latest_user_message(session_id)
		await _fire("retrieval_done", {"count": 0, "note": "out-of-scope or no retrieval"})
		if debug:
			print("RETRIEVAL RETURNED STRING INSTEAD OF CONTEXT BLOCKS:")
			print(context_blocks)
		return {
			"answer_text": context_blocks.strip(),
			"plan": plan,
			"retrieval_debug": plan.get("_retrieval_debug", {}),
			"context_blocks": [],
		}

	if not isinstance(context_blocks, list):
		delete_latest_user_message(session_id)
		await _fire("error", {"phase": "retrieval", "error": "invalid_type"})
		if debug:
			print("RETRIEVAL RETURNED INVALID TYPE:")
			print(type(context_blocks))
		return {
			"answer_text": "দুঃখিত, এই টার্নটি ঠিকভাবে প্রক্রিয়া করতে পারিনি। আবার বলুন।",
			"plan": plan,
			"retrieval_debug": plan.get("_retrieval_debug", {}),
			"context_blocks": [],
		}

	await _fire("retrieval_done", {
		"count": len(context_blocks),
		"top_rerank": (plan.get("_retrieval_debug") or {}).get("top_rerank_score"),
	})

	# --- Thread state patch (BEFORE answer generation) ---
	try:
		apply_thread_state_patch(session_id, plan.get("thread_state_patch", {}))
	except Exception as e:
		if debug:
			print("THREAD STATE PATCH FAILED:")
			print(repr(e))

	# --- Answer generation ---
	await _fire("answer_start", {
		"route": plan.get("route", ""),
		"context_count": len(context_blocks),
	})
	try:
		answer = await generate_answer(
			session_id,
			user_message,
			plan,
			context_blocks,
			stream=stream,
			stream_callback=stream_callback,
			thinking_callback=thinking_callback,
		)
	except Exception as e:
		delete_latest_user_message(session_id)
		await _fire("error", {"phase": "answer", "error": repr(e)})
		if debug:
			print("ANSWER GENERATION FAILED:")
			print(repr(e))
		return {
			"answer_text": "দুঃখিত, উত্তর তৈরি করতে সমস্যা হয়েছে। আবার বলুন।",
			"plan": plan,
			"retrieval_debug": plan.get("_retrieval_debug", {}),
			"context_blocks": context_blocks,
		}
	await _fire("answer_done", {"length": len(answer or "")})

	# --- food2disease comorbidity (still runs after answer is streamed; user is reading) ---
	await _fire("comorbidity_start", {})
	try:
		coms = await food2disease_como(answer, session_id=session_id)
		answer = answer + coms
	except Exception as e:
		if debug:
			print("FOOD2DISEASE_COMO FAILED (non-fatal):")
			print(repr(e))
	await _fire("comorbidity_done", {})

	if not isinstance(answer, str):
		delete_latest_user_message(session_id)
		if debug:
			print("ANSWER IS NOT A STRING:")
			print(type(answer))
		return {
			"answer_text": "দুঃখিত, উত্তরটি ঠিকভাবে তৈরি হয়নি। আবার বলুন।",
			"plan": plan,
			"retrieval_debug": plan.get("_retrieval_debug", {}),
			"context_blocks": context_blocks,
		}

	answer = answer.strip()
	if not answer:
		delete_latest_user_message(session_id)
		if debug:
			print("ANSWER IS EMPTY AFTER STRIP.")
		return {
			"answer_text": "দুঃখিত, উত্তরটি ফাঁকা এসেছে। আবার বলুন।",
			"plan": plan,
			"retrieval_debug": plan.get("_retrieval_debug", {}),
			"context_blocks": context_blocks,
		}

	# Append a 📚 সূত্র (Sources) section with pretty labels and clean URLs
	if isinstance(context_blocks, list) and context_blocks:
		seen_urls = set()
		src_lines = []
		for b in context_blocks:
			raw_url = (b.get("url") or "").strip()
			clean_url = _strip_url_fragment(raw_url)
			if not clean_url or clean_url in seen_urls:
				continue
			seen_urls.add(clean_url)
			label = _pretty_source_label(raw_url, fallback=b.get("dataset") or "")
			src_lines.append(f"{len(src_lines) + 1}. [{label}]({clean_url})")
			if len(src_lines) >= ANSWER_BLOCK_MAX_ITEMS:
				break
		if src_lines:
			answer = answer.rstrip() + "\n\n📚 সূত্র:\n" + "\n".join(src_lines)

	add_assistant_message(session_id, answer)

	try:
		save_session_to_disk(session_id)
	except Exception as e:
		if debug:
			print("SESSION SAVE FAILED (non-fatal):")
			print(repr(e))

	turn_dt = time.perf_counter() - turn_t0
	print(f"[turn] total wall-clock: {turn_dt:.2f}s")

	# Use 2 (debugging): noisy log on suspicious turns
	thinking_snapshot = {
		"planner": LAST_THINKING.get("planner", ""),
		"multiquery": LAST_THINKING.get("multiquery", ""),
		"food2disease": LAST_THINKING.get("food2disease", ""),
		"responder": LAST_THINKING.get("responder", ""),
	}
	_log_weird_planner_or_retrieval(
		plan,
		thinking_snapshot,
		plan.get("_retrieval_debug", {}) or {},
	)

	# Use 3 (eval data): append this turn to a per-session JSONL file
	_save_turn_eval(
		session_id=session_id,
		user_message=user_message,
		answer=answer,
		plan=plan,
		thinking=thinking_snapshot,
		retrieval_debug=plan.get("_retrieval_debug", {}) or {},
	)

	await _fire("turn_done", {"answer_length": len(answer or "")})

	if debug:
		print("=" * 100)
		print("USER MESSAGE:")
		print(user_message)

		print("\nPLAN:")
		try:
			print(json.dumps(plan, ensure_ascii=False, indent=2))
		except Exception:
			print(plan)

		print("\nRETRIEVAL DEBUG:")
		try:
			print(json.dumps(plan.get("_retrieval_debug", {}), ensure_ascii=False, indent=2))
		except Exception:
			print(plan.get("_retrieval_debug", {}))

		print(f"\nCONTEXT BLOCKS: {len(context_blocks)}")
		for i, b in enumerate(context_blocks, 1):
			print("-" * 100)
			if isinstance(b, dict):
				rerank_score = b.get("rerank_score", "")
				rerank_text = f"{rerank_score:.3f}" if isinstance(rerank_score, (int, float)) else str(rerank_score)
				print(f"[Block {i}] domain={b.get('domain', '')} | rerank={rerank_text}")
				print(f"url: {b.get('url', '')}")
				print(f"text: {str(b.get('text', ''))[:600]}")
			else:
				print(f"[Block {i}] NON-DICT BLOCK")
				print(str(b)[:600])

		print("\nANSWER:")
		print(answer)
		print("=" * 100)

	return {
		"answer_text": answer,
		"plan": plan,
		"retrieval_debug": plan.get("_retrieval_debug", {}),
		"context_blocks": context_blocks,
		"thinking": {
			"planner": LAST_THINKING.get("planner", ""),
			"multiquery": LAST_THINKING.get("multiquery", ""),
			"food2disease": LAST_THINKING.get("food2disease", ""),
			"responder": LAST_THINKING.get("responder", ""),
		},
	}
