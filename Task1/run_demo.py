import sys
sys.path.append("..")

from transformers_100.src.transformers.models.qwen2 import Qwen2Config, Qwen2Model
import torch

def run_qwen2():
    qwen2config = Qwen2Config(vocab_size=151936,
                              hidden_size=4096//2,
                              intermediate_size=22016//2,
                              num_hidden_layers=32//2,
                              num_attention_heads=32,
                              max_position_embeddings=2048//2)
    
    qwen2config = Qwen2Model(config=qwen2config)

    inputids = torch.randint(0, qwen2config.config.vocab_size, (4, 30))

    res = qwen2config(inputids)

    print(type(res))

if __name__ == "__main__":
    run_qwen2()