First run `uv sync`. if you dont have uv, use `pip install uv`.

install dependencies using:
`uv venv --python 3.12 --seed --managed-python --clear && source .venv/bin/activate && uv pip install vllm --torch-backend=auto`

download the models using:
`pip download_models.py`

spin up the server using:
`source .venv/bin/activate && vllm serve ./gemma_4b --enable_lora --lora_modules meno="./adapter_latest" --max_model_len=4096`

Note for windows:

`uv venv --python 3.12 --seed --managed-python --clear && .venv\Scripts\activate && uv pip install vllm --torch-backend=auto`
`.venv\Scripts\activate && vllm serve ./gemma_4b --enable_lora --lora_modules meno="./adapter_latest" --max_model_len=4096`