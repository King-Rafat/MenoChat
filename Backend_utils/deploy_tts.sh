sudo apt update -y && sudo apt-get install espeak-ng -y && source TTSe/bin/activate && uvicorn tts_server:app --host 0.0.0.0 --port 5431
