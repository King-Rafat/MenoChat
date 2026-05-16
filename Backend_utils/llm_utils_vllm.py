import asyncio
import inspect
import json
import os
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


PLANNER_MAX_NEW_TOKENS = 400
PLANNER_TEMPERATURE = 0.0
PLANNER_TOP_P = 0.9

MULTIQUERY_MAX_NEW_TOKENS = 180
MULTIQUERY_TEMPERATURE = 0.0
MULTIQUERY_TOP_P = 0.9

RESPONDER_MAX_NEW_TOKENS = 1000
RESPONDER_TEMPERATURE = 0.2
RESPONDER_TOP_P = 0.9

# Smaller top_k means fewer pairs to rerank later. 20 is plenty when we
# also use a parallel prefetch on the raw user message.
RETRIEVE_TOP_K = 20
ANSWER_BLOCK_MAX_CHARS = 1400
ANSWER_BLOCK_MAX_ITEMS = 5
RERANK_SCORE_FLOOR = 0.15

ALLOWED_RISK_LEVELS = {"none", "routine", "elevated", "urgent"}
ALLOWED_PLANNER_ROUTES = {
	"out-of-scope",
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
}

SESSIONS: dict[str, dict[str, Any]] = {}
DISEASEs_USER: list[str] = []

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

_session_storage_dir = Path("menochat_sessions")
_executor = None


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


async def llm_chat(
	prompt: str,
	*,
	max_tokens: int,
	temperature: float,
	top_p: float,
	extra_body: dict | None = None,
	tag: str = "llm",
) -> str:
	"""LLM call with timing prints. Tag identifies which stage called us."""
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
	out = (resp.choices[0].message.content or "").strip()
	dt = time.perf_counter() - t0
	in_tok = resp.usage.prompt_tokens if resp.usage else 0
	out_tok = resp.usage.completion_tokens if resp.usage else 0
	tps = out_tok / max(dt, 0.01)
	print(f"[llm_chat:{tag}] {dt:.2f}s | in={in_tok} out={out_tok} | {tps:.1f} tok/s")
	return out


# =====================================================================
# PLANNER (separate call)
# =====================================================================

def build_planner_prompt(session_id: str, user_message: str) -> str:
	sess = get_session(session_id)
	history_text = format_recent_history(session_id, max_turns=6)
	thread_state = sess.get("thread_state", {})
	return f"""
You are the planner for a Bengali women's-health assistant. Decide how to
route each user turn and rewrite vague turns into clean standalone questions.

═══════════════════════════════════════════════════════════════════════════
SCOPE
═══════════════════════════════════════════════════════════════════════════
IN SCOPE: menstrual and menopausal health. Periods, cramps, PCOS, endometriosis,
fibroids, perimenopause, menopause, PMS/PMDD

OUT OF SCOPE: general headache without menstrual context, general fever,
weather, politics, general knowledge,anything unrelated to menstrual/menopausal health.

When out of scope: route = "out_of_scope", needs_retrieval = false.

═══════════════════════════════════════════════════════════════════════════
ROUTING
═══════════════════════════════════════════════════════════════════════════
    
- smalltalk            → greetings, thanks
- health_direct        → in-scope standalone health questions
- health_followup      → in-scope context-dependent follow-ups
- health_education     → in-scope explanatory / conceptual questions
- sensitive_supportive → emotional or fearful turns needing warmth first
- urgent_redflag       → heavy bleeding, fainting, severe weakness, chest pain,
                          severe symptoms requiring immediate medical attention

CALIBRATION RULES — read before routing:
- A woman asking "পিরিয়ডে ব্যথা হয়" is health_direct, NOT sensitive_supportive.
  sensitive_supportive = only when the user expresses EXPLICIT fear, shame, or distress
  ("ভয় পাচ্ছি", "লজ্জা লাগছে", "কাঁদছি", "একা feel করছি")
- urgent_redflag = ONLY for the physical danger signals listed below.
  A worried tone alone does NOT make something urgent.
- When in doubt between health_direct and sensitive_supportive, choose health_direct.
URGENCY RULES
Route = "urgent_redflag" and risk_level = "urgent" if the user describes:
- Heavy bleeding (soaking pads quickly, large clots, bleeding that will not stop)
- Dizziness, lightheadedness, fainting, seeing darkness
- Severe weakness, inability to stand
- Chest pain, shortness of breath
- Severe abdominal pain with fever

When in doubt, escalate. A false positive is mild. A false negative is catastrophic.

"elevated" = concerning but not immediately dangerous (irregular periods for months, new pain patterns).

RESOLVED_QUESTION RULES
resolved_question is a SINGLE clean retrieval-friendly question in the user's language.
It is NOT the answer. It is NOT multi-sentence. It does NOT contain acknowledgments.
Max 200 characters.

BAD: "আমি বুঝতে পারছি আপনার অস্বস্তি হচ্ছে। আপনি যদি বিস্তারিত বলতে পারেন..."
GOOD: "গত দুই মাস ধরে পিরিয়ড অনিয়মিত হওয়ার সম্ভাব্য কারণ কী?"


ANSWER_GOAL RULES
Prefer concrete goals. Avoid vague goals like "summarize", "understand", "support".
For yes/no questions, the answer_goal must be NEUTRAL (frame the question, don't assume the answer).

GOOD: "give action-first practical guidance on managing menstrual cramps"
GOOD: "explain whether running during menstruation is advisable based on evidence"
BAD:  "explain why hot milk worsens menstrual pain" (assumes the answer)
BAD:  "provide emotional support" (too vague)


OUTPUT SCHEMA
Return ONLY valid JSON with this exact schema:
{{
  "route":  "smalltalk" | "health_direct" | "health_followup" | "health_education" | "sensitive_supportive" | "urgent_redflag",
  "risk_level": "none" | "routine" | "elevated" | "urgent",
  "depends_on_history": true or false,
  "needs_retrieval": true or false,
  "resolved_question": "single clean question, max 200 chars",
  "retrieval_query": "retrieval-friendly query string; empty if retrieval not needed",
  "answer_goal": "concrete goal for the responder",
  "thread_state_patch": {{
    "active_topic": "",
    "risk_level": "",
    "last_resolved_question": "",
    "last_retrieval_query": "",
    "last_route": ""
  }}
}}

CURRENT TURN
Recent conversation:
{history_text}

Latest user message:
{user_message}

Return JSON only, no other text.
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
		extra_body={"repetition_penalty": 1.1},
		tag="planner",
	)

	parsed = _extract_first_json_object(gen_text)
	print("%" * 100)
	print(parsed)
	if not isinstance(parsed, dict):
		parsed = {
			"route": "out-of-scope",
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

	route = _normalize_ws(parsed.get("route", "out-of-scope"))
	if route == "out_of_scope":
		route = "out-of-scope"
	if route not in ALLOWED_PLANNER_ROUTES:
		route = "out-of-scope"

	risk_level = _normalize_ws(parsed.get("risk_level", "none")).lower()
	if risk_level not in ALLOWED_RISK_LEVELS:
		risk_level = "none"

	depends_on_history = bool(parsed.get("depends_on_history", False))
	needs_retrieval = bool(parsed.get("needs_retrieval", False))
	resolved_question = _normalize_ws(parsed.get("resolved_question", user_message))[:300]
	retrieval_query = _normalize_ws(parsed.get("retrieval_query", ""))[:300]
	answer_goal = _normalize_ws(parsed.get("answer_goal", "answer the user's question directly"))

	if route in RETRIEVAL_REQUIRED_ROUTES:
		needs_retrieval = True
	if route in {"smalltalk", "out-of-scope"}:
		needs_retrieval = False
		retrieval_query = ""
	if needs_retrieval and not retrieval_query:
		retrieval_query = resolved_question

	llm_patch = parsed.get("thread_state_patch", {}) or {}
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
		"resolved_question": resolved_question,
		"retrieval_query": retrieval_query,
		"answer_goal": answer_goal,
		"thread_state_patch": thread_state_patch,
	}

	sess["controller_last_decision"] = out
	return out


# =====================================================================
# MULTI-QUERY (separate call, kept on its own as you wanted)
# =====================================================================

def build_multiquery_prompt(session_id: str, user_message: str) -> str:
	"""
	Multi-query prompt now depends ONLY on the user's message + history.
	No planner output is required, so this LLM call can run IN PARALLEL with
	the planner instead of after it.
	"""
	recent_history = format_recent_history(session_id, max_turns=6)
	return f"""
You generate retrieval queries for a Bengali women's-health assistant.

Your ONLY job is to produce 2-4 good search queries for a vector database
based on the user's latest message and recent conversation.

RULES:
- Query 1 must be the best standalone version of the user's actual question (Bangla).
- Queries 2-4 should be paraphrases, slightly broader formulations, or English
  retrieval-friendly versions of the same question.
- All queries must stay focused on the same topic.
- Mix Bengali and English if that improves retrieval.
- If the latest message is a vague follow-up (e.g. "তাহলে কি করব?"), use the
  recent conversation to infer the actual underlying topic and write
  standalone queries about that topic.
- Do NOT answer the question.
- Do NOT explain anything.
- Do NOT add queries about unrelated topics.
- If the latest message is clearly out-of-scope (not menstrual/menopausal
  health) or pure smalltalk, return an empty list.

Return ONLY valid JSON:
{{
  "queries": ["q1", "q2", "q3"]
}}

Recent conversation:
{recent_history}

Latest user message:
{user_message}
""".strip()


async def generate_multi_queries(
	session_id: str, user_message: str, max_queries: int = 4
) -> tuple[list[str], dict | None]:
	"""
	Generate retrieval queries from user_message + history alone.
	No planner dependency, so this can be fired in parallel with the planner.
	"""
	base_queries = [_normalize_ws(user_message)] if user_message else []

	prompt = build_multiquery_prompt(session_id, user_message)
	gen_text = await llm_chat(
		prompt,
		max_tokens=MULTIQUERY_MAX_NEW_TOKENS,
		temperature=MULTIQUERY_TEMPERATURE,
		top_p=MULTIQUERY_TOP_P,
		extra_body={"repetition_penalty": 1.1},
		tag="multiquery",
	)
	parsed = _extract_first_json_object(gen_text)

	llm_queries = []
	if isinstance(parsed, dict):
		for q in parsed.get("queries", []):
			q = _normalize_ws(q)
			if q and q not in llm_queries:
				llm_queries.append(q)

	final_queries = []
	for q in llm_queries + base_queries:
		if q and q not in final_queries:
			final_queries.append(q)

	return final_queries[:max_queries], parsed if isinstance(parsed, dict) else None


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
	"""FAISS only. No rerank. Cheap so we can run several in parallel."""
	q_emb = _embed_model.encode([query], batch_size=1, max_length=512)["dense_vecs"]
	q = np.array(q_emb, dtype=np.float32)
	if q.ndim == 1:
		q = q.reshape(1, -1)
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
	"""One rerank pass over the deduped union of all FAISS results."""
	if not items:
		return []
	t0 = time.perf_counter()
	pairs = [(query, it["text"]) for it in items]
	scores = _reranker.compute_score(pairs)
	if isinstance(scores, (float, int)):
		scores = [float(scores)]
	else:
		scores = [float(x) for x in scores]
	for it, s in zip(items, scores):
		it["rerank_score"] = s
	sorted_items = sorted(items, key=lambda x: x["rerank_score"], reverse=True)
	dt = time.perf_counter() - t0
	print(f"[rerank] {dt:.2f}s | pairs={len(pairs)}")
	return sorted_items


async def rerank_once_async(query: str, items: list[dict]) -> list[dict]:
	loop = asyncio.get_running_loop()
	return await loop.run_in_executor(_executor, _rerank_once, query, items)


def build_larger_clean_context_blocks(
	results: list[dict],
	max_items: int = ANSWER_BLOCK_MAX_ITEMS,
	max_chars: int = ANSWER_BLOCK_MAX_CHARS,
) -> list[dict]:
	blocks = []
	seen = set()
	for item in results:
		if float(item.get("rerank_score", -999)) < RERANK_SCORE_FLOOR:
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
		parts.append(
			f"[Source {i}]\n"
			f"dataset: {b['dataset']}\n"
			f"chunk_id: {b['chunk_id']}\n"
			f"url: {b['url']}\n"
			f"rerank_score: {b['rerank_score']:.4f}\n"
			f"text: {b['text']}"
		)
	return "\n\n" + ("\n\n" + "-" * 100 + "\n\n").join(parts)


async def maybe_run_retrieval_from_plan(
	plan: dict,
	user_message: str | None = None,
	session_id: str | None = None,
	prefetch_results: list[dict] | None = None,
	query_variants: list[str] | None = None,
	mq_parsed: dict | None = None,
) -> list[dict] | str:
	"""
	Retrieval pipeline. Multi-query is now expected to be pre-computed and
	passed in via `query_variants` (because it ran in parallel with the
	planner upstream in run_chat_turn).

	Steps:
	  1) FAISS search for each query variant in PARALLEL.
	  2) Merge with prefetch_results (FAISS on raw user message).
	  3) Dedupe by chunk idx.
	  4) ONE rerank pass over the deduped union, scored against the resolved
	     question. Way cheaper than reranking each query's results separately.
	"""
	route = plan.get("route", "")
	needs_retrieval = bool(plan.get("needs_retrieval", False))

	if route == "out-of-scope":
		plan["_retrieval_debug"] = {"reason": "planner_marked_out_of_scope"}
		return tell_this_is_beyond_chatbot_capacity()

	if not needs_retrieval:
		plan["_retrieval_debug"] = {"reason": "planner_marked_no_retrieval"}
		return []

	# Use pre-computed queries from the parallel multi-query call.
	queries = [q for q in (query_variants or []) if q]
	# Always seed at least the planner's resolved_question or user message.
	for fallback in [plan.get("resolved_question"), user_message]:
		f = _normalize_ws(fallback or "")
		if f and f not in queries:
			queries.append(f)
	if not queries:
		queries = [_normalize_ws(user_message or "")]
	queries = [q for q in queries if q][:4]

	t0 = time.perf_counter()

	# 1) FAISS in parallel
	faiss_tasks = [faiss_only_search_async(q) for q in queries]
	faiss_lists = await asyncio.gather(*faiss_tasks) if faiss_tasks else []

	# 2 + 3) Merge with prefetch and dedupe
	merged: list[dict] = []
	seen_idx: set = set()
	if prefetch_results:
		for r in prefetch_results:
			if r["idx"] not in seen_idx:
				seen_idx.add(r["idx"])
				merged.append(r)
	for results in faiss_lists:
		for r in results:
			if r["idx"] not in seen_idx:
				seen_idx.add(r["idx"])
				merged.append(r)

	faiss_dt = time.perf_counter() - t0
	print(f"[retrieval] faiss merge: {faiss_dt:.2f}s | queries={len(queries)} | merged={len(merged)}")

	# 4) Single rerank pass over deduped union
	rerank_query = plan.get("resolved_question") or (queries[0] if queries else "")
	merged = await rerank_once_async(rerank_query, merged)

	# Clean context blocks
	context_blocks = build_larger_clean_context_blocks(
		merged,
		max_items=ANSWER_BLOCK_MAX_ITEMS,
		max_chars=ANSWER_BLOCK_MAX_CHARS,
	)

	plan["_retrieval_debug"] = {
		"reason": "retrieval_ran",
		"llm_multiquery_output": mq_parsed if isinstance(mq_parsed, dict) else {},
		"query_variants": queries,
		"merged_count": len(merged),
		"final_context_count": len(context_blocks),
		"top_rerank_score": float(merged[0]["rerank_score"]) if merged else None,
	}
	return context_blocks


def build_responder_prompt(
	session_id: str,
	user_message: str,
	plan: dict,
	context_blocks: list[dict],
) -> str:
	sess = get_session(session_id)
	recent_history = format_recent_history(session_id, max_turns=6)
	thread_state = sess.get("thread_state", {})
	retrieval_context = format_larger_context_blocks_for_prompt(context_blocks)
	route = plan.get("route", "health_direct")
	risk_level = plan.get("risk_level", "routine")

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

	return f"""
তুমি একজন সহানুভূতিশীল, তথ্যনির্ভর বাংলা স্বাস্থ্য-সহকারী।

{route_instructions}

তোমার উত্তর:
- সহজ, স্বাভাবিক, পরিষ্কার, paragraph-based এবং সম্মানজনক বাংলায় হবে
- বাংলাদেশি ব্যবহারকারীর জন্য উপযোগী হবে
- simple social reply না হলে উত্তর অযথা ছোট কোরো না।
- সর্বপ্রথম ব্যবহারকারীর সর্বশেষ প্রশ্নের সরাসরি উত্তর দেবে
- follow-up হলে আগের relevant context ব্যবহার করবে, কিন্তু পুরোনো উত্তর repeat করবে না
- retrieved context থেকে useful তথ্য নিজের ভাষায় ব্যবহার করবে, কপি করবে না
- thin, vague, repetitive, বা surface-level উত্তর দেবে না
- metadata, ids, source labels, score, raw dict text, chunk dump — কিছুই output করবে না

Grounding rule (খুব গুরুত্বপূর্ণ hallucination prevention):
উত্তরে শুধুমাত্র সেই facts, symptoms, warning signs, conditions, medications, বা claims ব্যবহার করবে যা
(ক) ব্যবহারকারী এই conversation-এ বলেছেন, অথবা
(খ) retrieved context-এ স্পষ্টভাবে আছে।
Context-এ না থাকলে নিজের মনে থাকা অন্য কোনো medical knowledge যোগ করবে না।

যদি user "এখন কী করা উচিত", "কি করব", "প্রথমে কোনটা করব" জিজ্ঞেস করে:
- আগে action-first practical answer দাও
- তারপর explanation দাও paragraph-based এবং সম্মানজনক বাংলায় হবে

যদি retrieved context-এ কোনো নির্দিষ্ট খাবার, পানীয়, কাজ, বা বিষয়ে সরাসরি তথ্য না থাকে:
- specific claim করবে না
- সৎভাবে বলবে যে এই নির্দিষ্ট বিষয়ে পর্যাপ্ত নির্ভরযোগ্য তথ্য নেই
- তারপর general safe guidance দিতে পারো, যদি প্রাসঙ্গিক হয়

যদি warning sign থাকে:
- পরিষ্কারভাবে বলবে
- অযথা ভয় ধরাবে না
- কিন্তু জরুরি হলে পরিষ্কার জরুরি সতর্কতা দেবে

Latest user message:
{user_message}

Route: {route}
Risk level: {risk_level}

Resolved question:
{plan.get("resolved_question", "")}

Answer goal:
{plan.get("answer_goal", "")}

User Disease:
{DISEASEs_USER}

Current thread state:
{json.dumps(thread_state, ensure_ascii=False)}

Recent conversation:
{recent_history}

LARGER CLEANED RETRIEVAL CONTEXT:
{retrieval_context}

উত্তরের কাঠামো:
- প্রথমে সরাসরি উত্তর
- তারপর দরকারি ব্যাখ্যা
""".strip()


async def _stream_answer(
	prompt: str,
	stream_callback: Optional[Callable[[str], Awaitable[None] | None]] = None,
) -> str:
	_require_init()
	t0 = time.perf_counter()
	full_response = ""
	first_token_t = None
	stream = await _llm_client.chat.completions.create(
		model=_llm_model_name,
		messages=[{"role": "user", "content": prompt}],
		stream=True,
		max_tokens=RESPONDER_MAX_NEW_TOKENS,
		temperature=RESPONDER_TEMPERATURE,
		top_p=RESPONDER_TOP_P,
		extra_body={"repetition_penalty": 1.15, "no_repeat_ngram_size": 15},
	)
	async for chunk in stream:
		delta = chunk.choices[0].delta.content
		if delta:
			if first_token_t is None:
				first_token_t = time.perf_counter() - t0
			full_response += delta
			if stream_callback:
				maybe_awaitable = stream_callback(delta)
				if inspect.isawaitable(maybe_awaitable):
					await maybe_awaitable
	dt = time.perf_counter() - t0
	approx_tok = len(full_response) // 3
	tps = approx_tok / max(dt, 0.01)
	ttft = first_token_t if first_token_t is not None else 0.0
	print(f"[llm_chat:responder_stream] total={dt:.2f}s | TTFT={ttft:.2f}s | ~{approx_tok} tok | ~{tps:.1f} tok/s")
	return full_response.strip()


async def generate_answer(
	session_id: str,
	user_message: str,
	plan: dict,
	context_blocks: list[dict],
	*,
	stream: bool = False,
	stream_callback: Optional[Callable[[str], Awaitable[None] | None]] = None,
) -> str:
	prompt = build_responder_prompt(session_id, user_message, plan, context_blocks)
	if stream:
		return await _stream_answer(prompt, stream_callback=stream_callback)
	return await llm_chat(
		prompt,
		max_tokens=RESPONDER_MAX_NEW_TOKENS,
		temperature=RESPONDER_TEMPERATURE,
		top_p=RESPONDER_TOP_P,
		extra_body={"repetition_penalty": 1.15, "no_repeat_ngram_size": 15},
		tag="responder",
	)


async def food2disease_como(user_message: str) -> str:
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
		max_tokens=RESPONDER_MAX_NEW_TOKENS,
		temperature=RESPONDER_TEMPERATURE,
		top_p=RESPONDER_TOP_P,
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

	DISEASEs_USER.extend(answer.get("detected_disease", []))
	unique_diseases = list(set(DISEASEs_USER))
	DISEASEs_USER.clear()
	DISEASEs_USER.extend(unique_diseases)

	file_path = os.path.join("menochat_diseases", "diseases_USER.txt")
	with open(file_path, "w") as f:
		for d in DISEASEs_USER:
			f.write(d + "\n")

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
	# Multi-query no longer depends on planner output (it only uses
	# user_message + recent_history), so it can run AT THE SAME TIME as the
	# planner. Add a FAISS search on the raw user message for free recall.
	#
	# Wall-clock = max(planner_time, multiquery_time, prefetch_time)
	# instead of planner_time + multiquery_time + prefetch_time.
	await _fire("planner_start", {"user_message": user_message})

	prefetch_task: Optional[asyncio.Task] = asyncio.create_task(
		faiss_only_search_async(user_message)
	)
	planner_task = asyncio.create_task(run_planner(session_id, user_message))
	multiquery_task = asyncio.create_task(
		generate_multi_queries(session_id, user_message, max_queries=4)
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
	})

	# Collect multi-query and prefetch results IF we still need retrieval.
	if plan.get("needs_retrieval") and plan.get("route") != "out-of-scope":
		try:
			prefetch_results = await prefetch_task
		except Exception:
			prefetch_results = []
		try:
			query_variants, mq_parsed = await multiquery_task
		except Exception:
			query_variants, mq_parsed = [], None
	else:
		# Cancel both siblings since we won't use them.
		for t in (prefetch_task, multiquery_task):
			if t and not t.done():
				t.cancel()
		prefetch_results = []
		query_variants, mq_parsed = [], None

	# --- Retrieval (uses prefetch + parallel multi-query results, single rerank) ---
	await _fire("retrieval_start", {
		"needs_retrieval": bool(plan.get("needs_retrieval", False)),
		"route": plan.get("route", ""),
	})
	try:
		context_blocks = await maybe_run_retrieval_from_plan(
			plan,
			user_message,
			session_id,
			prefetch_results=prefetch_results,
			query_variants=query_variants,
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
		coms = await food2disease_como(answer)
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

	add_assistant_message(session_id, answer)

	try:
		save_session_to_disk(session_id)
	except Exception as e:
		if debug:
			print("SESSION SAVE FAILED (non-fatal):")
			print(repr(e))

	turn_dt = time.perf_counter() - turn_t0
	print(f"[turn] total wall-clock: {turn_dt:.2f}s")

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
	}
