from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./lora-tutor", 
    max_seq_length = 2048,
    load_in_4bit = True,
)

model.save_pretrained_gguf("tutor_model", tokenizer, quantization_method = "q4_k_m")