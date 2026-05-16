# MenoChat x Gemma 4

> Bangla voice-first women's health assistant. Gemma 4 E4B brain, custom Whisper-small ASR, fine-tuned Bangladeshi VITS TTS. Delivered as a Telegram bot.


## About

MenoChat is a private, Bangla-language voice assistant for menstrual and menopausal health. You speak in Bangla, it answers in Bangla, out loud. Built to lower both the language barrier and the social shame that keep many of Bangladesh's 170M+ Bangla speakers from getting clear answers about women's health.

## ASR

Fine-tuned **Whisper-small (240M)** for Bangla women's health speech.

- Flash Attention for fast inference on one GPU
- Noise augmentation (fans, traffic, TV chatter) and SpecAugment for robustness
- Bangla medical vocab in fine-tune data
- Audio path: Telegram OGG/Opus → ffmpeg 16k mono WAV → denoise → Whisper

**User scores (n=12, Likert 1–5):** catches medical terms **4.42**, handles noisy speech **4.00**, fast / realtime **4.50**.

## Gemma 4 (the brain)

**Gemma 4 E4B** with LoRA fine-tune on 5,187 Bangla women's health conversations. Wrapped in a 4-call pipeline:

1. **Planner.** Gemma native function calling picks one of: `route_smalltalk`, `route_out_of_scope`, `answer_health_question`, `escalate_red_flag`, `provide_emotional_support`.
2. **Retrieval.** Multi-query rewrite → FAISS + BGE-M3 embeddings + BGE reranker.
3. **Responder.** Grounded Bangla reply, no drug names or dosages.
4. **Safety pass.** Comorbidity and food-disease check against the user's known conditions.

Adapter: `Apurba-NSU-RnD-Lab/MenoChat_gemma3_4b_finetuned`. Merged target: `RafatK/menochat-gemma3_4b-merged`. Served on vLLM with LoRA hot-loading.

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

**User scores (n=12, Likert 1–5):** speaks words accurately **2.83**, empathetic **2.83**, fast / realtime **3.25**. Current bottleneck of the system, top fix on the roadmap.

## Training Script

[`Gemma4_E4B_Menochat.ipynb`](./Gemma4_E4B_Menochat.ipynb), built on Unsloth's Gemma 4 E4B template, adapted to the Menochat JSONL dataset.

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-4-E2B-it",
    max_seq_length = 2048,
    load_in_4bit   = True,        # QLoRA
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers   = False,    # text decoder only
    finetune_language_layers = True,
    r = 16, lora_alpha = 32, lora_dropout = 0.05,
)
```

| Setting | Value |
| --- | --- |
| Dataset | 5,187 Bangla women's health conversations (JSONL, HF `messages` format) |
| Epochs | 2 |
| Effective batch | 8 (per-device 1 × grad-accum 8) |
| LR / schedule | 1e-4 / cosine, warmup 5% |
| Precision | bf16, adamw_8bit, gradient checkpointing |
| Loss masking | `train_on_responses_only` (model loss only) |
| Best-checkpoint | `load_best_model_at_end=True` on `eval_loss` |

Outputs: LoRA adapter saved locally, optional merged push to HuggingFace, optional GGUF `Q8_0` export for `llama.cpp`.

## User Evaluation (n = 12)

Structured evaluation through the NSU-APURBA-icddr,b form. Each user completed **6 task scenarios**, rated each on 7 dimensions, plus SUS, trust, ASR, and TTS questions.

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

**Headline:** safe **4.47**, trust **4.33**, recommend **4.83**, SUS **65.6**. ASR is the strongest leg, TTS is the weakest. Voice clarity scores stay low across all 6 tasks regardless of task type, confirming the bottleneck is synthesis, not reasoning.

## Quick Start

```bash
vllm serve ./gemma_4b --enable_lora --lora_modules meno="./adapter_latest"
export TELEGRAM_BOT_TOKEN="..."
python telegram_bot.py
```

## Layout

| File | What |
| --- | --- |
| `Gemma4_E4B_Menochat.ipynb` | LoRA training |
| `menochat_pipeline.py` | Planner → RAG → responder → safety |
| `telegram_bot.py` | Telegram front-end (text + voice) |
| `audio_denoise.py` | Pre-ASR denoising |
| `MenoChat_Kaggle_Writeup.md` | Hackathon write-up |

## Acknowledgements

Apurba-NSU-RnD-Lab (dataset + baseline adapter), icddr,b (evaluation cohort), Unsloth (training template), Google DeepMind (Gemma 4).

## License

Code + LoRA: Apache 2.0. Gemma 4 base weights under Google's Gemma license.
