"""
VITS Bangla TTS HTTP server.

This service is intentionally kept tiny and independent of the Chainlit app.
It must run in its OWN virtual environment because the `TTS` (Coqui) package
pins very specific versions of numpy / torch / transformers that conflict
with the Chainlit app's dependency tree.

Run with:
    uvicorn tts_server:app --host 0.0.0.0 --port 5431
"""

import io
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from TTS.utils.synthesizer import Synthesizer

REPO_ID = "Apurba-NSU-RnD-Lab/MenoChat_ViTs_Bangla_TTS"

print("Downloading VITS model files...")
model_path = hf_hub_download(REPO_ID, "pytorch_model.pth")
config_path = hf_hub_download(REPO_ID, "config.json")

print("Loading VITS model...")
synth = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    use_cuda=False,
)
print("Model ready.")

app = FastAPI()


class Req(BaseModel):
    text: str


@app.post("/tts")
def tts(req: Req):
    text = (req.text or "").strip()
    if not text:
        return Response(status_code=400, content="empty text")
    wav = np.array(synth.tts(text), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, 22050, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.get("/healthz")
def healthz():
    return {"ok": True}