# MenoChat x Gemma 4

> Bangla voice-first women's health assistant. Gemma 4 E2B brain, custom Whisper-small ASR, fine-tuned Bangladeshi VITS TTS. Delivered as a Telegram bot. Looking to host on Twilio.
> Access via Telegram: https://t.me/MenoChatBot


## About

MenoChat is a voice-first Bangla health assistant for menstrual and menopausal concerns. She speaks in Bangla. MenoChat listens, thinks in Bangla, and answers in Bangla, out loud, in a voice she can understand. No typing, no English, no judgement. Gemma 4 E2B is the brain. We chose it for three properties that made the rest of the system possible: open weights, so we could fine-tune freely on a domain no closed model would let us touch; a 2B footprint, so four structured calls still run in four to five seconds on one laptop GPU; the four calls allow more control and better denoising; and native multilingual training, which already understood Bangla before we taught it medical Bangla.

## Gemma 4 (the brain)

Gemma 4 E2B with LoRA fine-tune on 5,187 Bangla women's health conversations.
The model is wrapped in a 4-stage pipeline (see `./Backend_utils/llm_utils.py`):
Retrieval is from a verified database consisting of websites and papers verified by doctors: https://drive.google.com/drive/folders/1txqITO1NRjx8-6yzozMYEs9Ulq10tvak?usp=sharing 
**Stage 1 (Planner) + Stage 2 (Multi-query) — fired in parallel**

| Planner output | Allowed values |
| --- | --- |
| `route` | `out_of_scope`, `smalltalk`, `health_direct`, `health_followup`, `health_education`, `sensitive_supportive`, `urgent_redflag` |
| `risk_level` | `none`, `routine`, `elevated`, `urgent` |
| `needs_retrieval` | bool, forced true for in-scope health routes |
| `resolved_question` | one clean Bangla question, ≤200 chars |
| `thread_state_patch` | updates `active_topic`, `symptom_cluster`, `risk_level`, `emotion_flag`, `last_resolved_question`, `last_retrieval_query`, `last_route` |

Multi-query asks Gemma for 2 to 4 retrieval queries (mix of Bangla + English paraphrases). This solves noisy ASR queries and also enables more and better retrieval.

**Stage 3 (Retrieval) — cheap, parallel, single rerank**

1. FAISS search in parallel for each query (top-20 each)
2. Merge with a prefetch FAISS hit on the raw user message
3. ONE rerank pass over the deduped union (BGE reranker, score floor 0.15)
4. Build up to 5 clean context blocks, each ≤1400 chars, sentence-trimmed
5. Junk filter strips `newsletter`, `cookie`, `subscribe`, `click here`, etc.
6. Sources get human labels (`UNFPA: Menstrual Health`, `WHO: ...`) from a doctor verified website map

**Stage 4 (Responder) — grounded Bangla reply**

- Bangla only
- No drug names, no dosages
- Cite sources by friendly label
- Refer to a doctor when risk is `elevated` or `urgent`

**Stage 5 (Safety pass)**

Comorbidity + food-disease check from `comorbidity.py`. A Bangla food-word regex pre-filter skips this call when the answer contains no food signal, saving 1 to 2 seconds on most turns.

**Session memory.** Each session is a JSON file in `menochat_sessions/`.
History capped at 3 turns (6 entries), assistant replies clipped to 200 chars
on recall. Thread state survives across turns so follow-ups work.

**GGUF_models_E2B for inference:** [`Gemma_E2B_GGUF`](https://huggingface.co/afifaimran/afi-gemma4-e2b-merged-gguf)
**End-to-end latency:** ~4 to 5 seconds per turn on a single GPU 5070 laptop

## ASR

Fine-tuned **Whisper-small (240M)** for Bangla women's health speech.

- Model Checkpoint: https://huggingface.co/Apurba-NSU-RnD-Lab/MenoChat_Whisper_Small
- Medical Adapter: https://huggingface.co/RafatK/Whisper_good_adapt
- **Full Model Training** -> Noise augmentation (fans, traffic, TV chatter, white noise etc) and SpecAugment for robustness (MUSAN + Audiomentations + ESC 50) on 3200 hrs of Audio (Common Voice, OpenSLR, MADASR, KathBath, Shrutilipi)
- **Adapter Training** -> Noise augmentation + Bangla medical vocab in fine-tune of adapter data
- Flash Attention for fast inference on one GPU
- Audio path: Telegram OGG/Opus → ffmpeg 16k mono WAV → denoise → Whisper

**Benchmark on Common Voice Test-Set**
| Model | WER |
| --- | ---: |
| Whisper large v3 | 40.35 |
| Gemma 4 transcription | 30.74 |
| Whisper-Small-Finetuned (Ours) | **16.54** |

**User scores (n=12, Likert 1–5):** catches medical terms **4.42**, handles noisy speech **4.00**, fast **4.50**. Users tested on real semi-noisy scenarios



**User scores (n=12, Likert 1–5):**

| Dimension | Mean |
| --- | ---: |
| Understood what I said | 4.19 |
| Medically accurate | 4.14 |
| Comorbidity-aware | 3.65 |
| Empathetic + culturally appropriate | 4.08 |
| Advice felt safe | **4.47** |
| Trust the health info | 4.33 |
| Would recommend | **4.83** |

## TTS

Custom **Bangladeshi VITS**, fine-tuned on Bangladeshi voice data. One-shot (non-autoregressive) for low-latency synthesis. Output is re-encoded to OGG/Opus so Telegram plays it as a real voice note.

TTS system has not been evaluated
TTS Checkpoint: https://huggingface.co/Apurba-NSU-RnD-Lab/MenoChat_ViTs_Bangla_TTS

## Training Script

[`Gemma4_E2B_Menochat.ipynb`](./Gemma4_E2B_Menochat.ipynb), built on Unsloth's Gemma 4 E2B template, adapted to the Menochat JSONL dataset.

| Setting | Value |
| --- | --- |
| Dataset | 5,187 Bangla women's health conversations (JSONL, HF `messages` format) |
| Epochs | 2 |
| Effective batch | 8 (per-device 1 × grad-accum 8) |
| LR / schedule | 1e-4 / cosine, warmup 5% |
| Precision | bf16, adamw_8bit, gradient checkpointing |
| Loss masking | `train_on_responses_only` (model loss only) |
| Best-checkpoint | `load_best_model_at_end=True` on `eval_loss` |
| Train and Validation Dataset: | https://huggingface.co/datasets/RafatK/Menochat_train_val |

Outputs: LoRA adapter saved locally, GGUF `Q8_0` export for `llama.cpp`.

## User Evaluation (n = 12)

Structured evaluation through 10 clinicians and 2 normal users. Each user completed **6 task scenarios**, rated each on 7 dimensions, plus SUS, trust, ASR, and TTS questions.

**Cohort:** 12 women, urban Bangladesh. Age 6–25 (3), 26–35 (3), 36–45 (3), 46–55 (3). Education: Master+ (8), Bachelor (4). Daily phone users.

**Per-task means (Likert 1–5):**

| # | Task | Understood | Accurate | Comorbid | Empathy | Voice | Safe | Fast | Avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Common period Q&A | **4.83** | 4.17 | 3.67 | 3.67 | 3.00 | 4.50 | 3.83 | 3.95 |
| 2 | Sensitive menstruation | 3.75 | 4.17 | 3.58 | **4.58** | 2.92 | **4.58** | 3.75 | 3.90 |
| 3 | Menopause + diabetes | 4.42 | **4.33** | **3.83** | 4.00 | 3.08 | 4.25 | 4.08 | **4.00** |
| 4 | Red-team / safety | 4.08 | 4.08 | 3.58 | 4.08 | 2.92 | 4.50 | 3.92 | 3.88 |
| 5 | Management + emotional | 3.92 | 4.00 | 3.58 | 4.08 | 3.17 | 4.42 | 4.08 | 3.89 |
| 6 | Out-of-scope | 4.17 | 4.08 | 3.67 | 4.08 | **3.67** | **4.58** | **4.25** | **4.07** |

**Per-user averages (across all 6 tasks) + SUS:**

| # | Respondent | Understood | Accurate | Comorbid | Empathy | Voice | Safe | Fast | SUS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 |  | 5.00 | 5.00 | 5.00 | 5.00 | 4.67 | 5.00 | 5.00 | 97.5 |
| 2 | - | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 55.0 |
| 3 | - | 3.67 | 4.17 | 3.17 | 3.50 | 1.67 | 4.33 | 3.33 | 40.0 |
| 4 | - | 3.83 | 4.00 | 3.00 | 3.67 | 2.00 | 4.67 | 1.83 | 37.5 |
| 5 | - | 4.17 | 4.17 | 3.67 | 4.67 | 2.33 | 4.50 | 4.50 | 87.5 |
| 6 | - | 4.50 | 3.83 | 3.50 | 4.50 | 3.33 | 4.50 | 4.67 | 90.0 |
| 7 | - | 4.33 | 3.67 | 3.67 | 4.17 | 3.50 | 4.00 | 4.17 | 55.0 |
| 8 | - | 4.67 | 4.83 | 4.50 | 4.67 | 4.50 | 4.83 | 4.67 | 95.0 |
| 9 | - | 4.50 | 3.00 | 3.17 | 3.67 | 4.67 | 3.50 | 4.17 | 67.5 |
| 10 | - | 2.50 | 3.00 | 3.00 | 3.50 | 1.00 | 3.67 | 2.67 | 67.5 |
| 11 | - | 4.00 | 4.33 | 2.83 | 3.00 | 3.33 | 4.83 | 4.33 | 55.0 |
| 12 | - | 4.17 | 4.67 | 3.33 | 3.67 | 1.50 | 4.83 | 3.50 | 40.0 |
| | **Group mean** | **4.19** | **4.14** | **3.65** | **4.08** | **3.12** | **4.47** | **3.99** | **65.6** |

The 9.8 point gap between the standard and positive-keyed SUS suggests reverse-item miskeying, a known artifact when SUS is administered to participants less familiar with bidirectional survey scales (Sauro and Lewis, 2011). We interpret 75.4 as the more reliable usability estimate for this population.
**Headline:** safe **4.47**, trust **4.33**, recommend **4.83**, SUS **65.6**. Positive SUS: 75.4. ASR is the strongest leg, TTS is the weakest. Voice clarity scores stay low across all 6 tasks regardless of task type, confirming the bottleneck is synthesis, not reasoning.

## Quick Start

```bash

```

## Layout

| File | What |
| --- | --- |
| `Gemma4_E4B_Menochat.ipynb` | LoRA training |
| `chainlit_app_cpu.py` | runs chainlit web app |
| `telegram_inside_chainlit.py` | Telegram front-end (text + voice) along with chainlit |
| `llm_utils.py` | Planner → RAG → responder → safety |
| `CPP_gemma.sh` | Run Gemma on GGUF |
| `Opening page` | This is the starting page for the webapp |
| `tts_server.py` | Runs the tts model on 5432 host |
| `tts_server.py` | Runs the tts model on 5432 host using deploy_tts.sh|


## Acknowledgements
 Unsloth (training template), Google DeepMind (Gemma 4). All of ASR+TTS+GEMMA+Chainlit have their own pyproject.toml to use the dependancies.

## License

Apache 2.0. Gemma 4 base weights under Google's Gemma license.
