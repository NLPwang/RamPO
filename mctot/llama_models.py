
import torch
import gradio as gr
from typing import List
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings
from accelerate import Accelerator
import numpy as np


class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_token,prompt):
        self.tokenizer = tokenizer
        self.stop_token = stop_token
        self.stop_token_ids = self.tokenizer.encode(stop_token, add_special_tokens=False)
        self.prompt = prompt
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.prompt in self.tokenizer.decode(input_ids[0], skip_special_tokens=True):
            generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).replace(self.prompt, "")
        else:
            generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).split(self.prompt[-5:])[-1]

        return self.stop_token in generated_text


class GPT:
    def __init__(self,args,temperature = 0.4,max_tokens = 256):
        if args.base_model == 'Llama3b':
            base_model = 'llama-3.2-3b-instruct'
        elif args.base_model == 'Llama8b':
            base_model = 'llama-3.1-8b-instruct'
        else:
            base_model = args.base_model
        tokenizer = AutoTokenizer.from_pretrained(base_model,padding_side = "left")
        
        if args.base_model == 'Llama3b':
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
            )
        elif args.base_model == 'Llama8b':
            # quantization_config = BitsAndBytesConfig(
            #     load_in_8bit=True,
            #     bnb_8bit_compute_dtype=torch.bfloat16
            # )
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                # quantization_config = quantization_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
            )

        model.config.pad_token_id = tokenizer.pad_token_id = 2  # unk
        tokenizer.pad_token = tokenizer.eos_token
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.config.pad_token_id = model.config.eos_token_id
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.model = self.accelerator.prepare_model(model)
        self.model.eval()
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens


    def generate(self,prompt,n = 1, stop_token_list = [], do_sample = True):
        inputs = self.tokenizer(prompt, return_tensors="pt") 
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        stopping_criteria = StoppingCriteriaList()
        for stop_token in stop_token_list:
            stopping_criteria.append(StopAtSpecificTokenCriteria(self.tokenizer, stop_token,prompt))

        s = []
        for i in range(n):
            generation_output = self.model.generate(input_ids=input_ids, pad_token_id = 2, max_new_tokens=self.max_tokens, attention_mask=attention_mask, do_sample=do_sample, temperature=self.temperature, num_return_sequences=1, stopping_criteria=stopping_criteria, repetition_penalty=1.1)
            s.append(self.tokenizer.batch_decode(generation_output, skip_special_tokens = True)[0])
        return s, input_ids



