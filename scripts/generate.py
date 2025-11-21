import torch
import argparse
from transformers import AutoTokenizer
from src.model.transformer import TransformerLM
from src.utils.config import ModelConfig
import torch.nn.functional as F

def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Autoregressive generation loop.
    Note: This is a simplified version not using KV-caching for brevity of the script file,
    but the Model supports it.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -1024:] # Crop to context window
        
        # Forward
        with torch.no_grad():
            logits = model(idx_cond)["logits"]
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
    return idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Hello world")
    args = parser.parse_args()

    # Load dummy config for model init (in real life, load from checkpoint)
    config = ModelConfig(d_model=768, n_layers=12, n_heads=12, d_ff=3072, vocab_size=50257)
    model = TransformerLM(config)
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(args.prompt, return_tensors="pt")
    
    out = generate(model, tokens, max_new_tokens=50)
    print(tokenizer.decode(out[0]))

if __name__ == "__main__":
    main()