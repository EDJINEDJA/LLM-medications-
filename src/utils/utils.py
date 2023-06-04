from json import load
from os import sep
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

class Utils():
    def __init__(self, config) -> None:
        self.config = config

    def load_datasets(self):
        data = load_dataset('text' , data_files = self.config["data"]["dataset_url"])
        return data

    def tokenization(self, tokenizer):
        data = self.load_datasets()
        data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

        return data
    
    def print_trainable_parameters(self,model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def load_models(self):
        model_id = self.config["models"]["model_id"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
        

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        
        config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            target_modules=["query_key_value"], 
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, config)
        self.print_trainable_parameters(model)

        return model , tokenizer