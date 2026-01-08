import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
import os
import certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

dataset = load_dataset("json", data_files="datasetTutor.jsonl")

model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  
    device_map="auto"
)

# Configurar LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

def format_instruction(example):
    texto = f" Instrucción: {example['prompt']}\n Respuesta: {example['response']}"
    return tokenizer(texto, truncation=True, padding="max_length", max_length=512)
tokenized = dataset.map(format_instruction)

# Entrenamiento
training_args = TrainingArguments(
    output_dir="./lora-tutor",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=50,
    num_train_epochs=3,
    fp16=True,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# Adaptadores LoRA
model.save_pretrained("./lora-tutor")
print("Entrenamiento completado")