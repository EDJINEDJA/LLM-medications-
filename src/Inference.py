from peft import LoraConfig, get_peft_model,PeftConfig, PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

def inference(text, config):
    device = config["models"]["device"]
    
    config = PeftConfig.from_pretrained(config["models"]["hp_repo"])
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict = True , Load_in_8bit=True, device_map="auto")

    model = PeftModel(model, config["models"]["hp_repo"])
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
