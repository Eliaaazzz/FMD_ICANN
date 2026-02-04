#!/usr/bin/env python3
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    free = torch.cuda.mem_get_info()[0]/1024**3
    total = torch.cuda.get_device_properties(0).total_memory/1024**3
    print("VRAM: %.1f/%.1f GB free" % (free, total))

print("\nTrying to load model...")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("/home/ufb/models/FMDLlama3")
print("Tokenizer OK")

print("Configuring 4-bit quantization...")
cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "/home/ufb/models/FMDLlama3",
    quantization_config=cfg,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
print("Model loaded!")
print("VRAM used: %.2f GB" % (torch.cuda.memory_allocated()/1024**3))

# Test inference
print("\nTesting inference...")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

test_prompt = "Human:\nTest prompt\n\nAssistant:\n"
inputs = tok(test_prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tok.pad_token_id)
result = tok.decode(out[0], skip_special_tokens=True)
print("Inference OK:", result[:100])
print("\nDiagnosis complete - model works!")
