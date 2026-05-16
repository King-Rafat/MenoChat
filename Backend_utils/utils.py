import chainlit as cl

async def text_to_speech(text: str):
    """Convert text to speech using VoxCPM"""
    cleaned = re.sub(r"\[.*?\]", "", text)
    
    prompt_list = []
    for chunk in cleaned.split("\n"):
        parts = re.split(r"।", chunk)
        for p in parts:
            p = p.strip()
            if p:
                prompt_list.append(p + "।")
    print(prompt_list)
    audio_arr_tot = []

    for prompt in prompt_list:
        audio = tts_model.generate(
            text=prompt,
            prompt_wav_path=None,
            prompt_text=None,
            cfg_value=4.0,
            inference_timesteps=15,
            normalize=False,
            denoise=False,
            retry_badcase=False,
        )
        if torch.is_tensor(audio):
            audio_arr = audio.cpu().numpy().squeeze()
        else:
            audio_arr = np.array(audio).squeeze()
        audio_arr_tot.append(audio_arr)

    combined = np.concatenate(audio_arr_tot, axis=0)
    
    output_path = "tts_output.wav"
    sampling_rate = 16000
    sf.write(output_path, combined, sampling_rate)
    
    return output_path

async def transcribe_audio(audio_file: cl.File):
    """Transcribe audio file using ASR model"""
    try:
        if hasattr(audio_file, 'path') and audio_file.path:
            temp_path = audio_file.path
        elif hasattr(audio_file, 'content') and audio_file.content:
            audio_bytes = audio_file.content
            temp_path = "temp_input_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_bytes)
        else:
            raise Exception("Audio file has no content or path")
        
        print(f"DEBUG: Transcribing audio from: {temp_path}")
        
        asr_result = asr(temp_path)
        transcription = asr_result["text"]
        
        return transcription
        
    except Exception as e:
        import traceback
        print(f"DEBUG: ASR Error traceback:\n{traceback.format_exc()}")
        raise Exception(f"ASR transcription failed: {e}")

def decode_only_new_tokens(outputs, inputs):
    input_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[0][input_len:]
    return ft_tok.decode(gen_ids, skip_special_tokens=True).strip()