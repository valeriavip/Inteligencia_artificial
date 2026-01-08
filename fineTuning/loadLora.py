from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model = PeftModel.from_pretrained(base_model, "./lora-tutor")

messages = [{"role": "user", "content": "Explícame qué es una función en Python."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(
    inputs, 
    max_new_tokens=500, 
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))