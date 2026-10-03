import torch
import sys
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path('/data/workspace/MiniGPT-3D').resolve()))

from pointllm.eval.eval_objaverse import setup_model

cfg = OmegaConf.load('./eval_configs/benchmark_evaluation_paper.yaml')
model, tokenizer, _ = setup_model(cfg, gpu_id=0)

llm_embed = model.llm_model.model.embed_tokens.weight.data[0].cpu()

official_phi = AutoModelForCausalLM.from_pretrained('microsoft/phi-2', trust_remote_code=True)
official_embed = official_phi.model.embed_tokens.weight.data[0].cpu()

cos = torch.nn.functional.cosine_similarity(llm_embed.float(), official_embed.float(), dim=0)
print(f'Cosine similarity: {cos.item():.6f}')