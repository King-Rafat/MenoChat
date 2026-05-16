from unsloth import FastLanguageModel

def load_llm(base_model_dir, adapter_dir):
    print("Loading LLM model...")
    ft_model, ft_tok = FastLanguageModel.from_pretrained(
        model_name=base_model_dir,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    ft_model = FastLanguageModel.get_peft_model(ft_model, r=16)
    ft_model.load_adapter(adapter_dir, adapter_name="apu_adapter")
    ft_model.set_adapter("apu_adapter")
    FastLanguageModel.for_inference(ft_model)
    print("LLM model loaded successfully!")
    return ft_model, ft_tok