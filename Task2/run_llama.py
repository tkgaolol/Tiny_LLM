import sys
sys.path.append("..")

from transformers_100.src.transformers.models.llama import LlamaConfig, LlamaModel
import torch

def run_llama():
    llama_config = LlamaConfig(vocab_size=151936,
                              hidden_size=4096//2,
                              intermediate_size=22016//2,
                              num_hidden_layers=32//2,
                              num_attention_heads=32,
                              max_position_embeddings=2048//2)
    
    llama_config = LlamaModel(config=llama_config)

    inputids = torch.randint(0, llama_config.vocab_size, (4, 30))

    res = llama_config(inputids)

    print(type(res))

if __name__ == "__main__":
    run_llama()