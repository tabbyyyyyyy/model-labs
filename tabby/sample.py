"""Generate text from a trained TabbyGPT checkpoint.

Usage:
    python sample.py --prompt "The cat " --length 300
"""
import argparse
import os

import torch

from model import TabbyConfig, TabbyGPT
from tokenizer import CharTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--length", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join(args.ckpt, "tabby.pt"), map_location=device)
    tok = CharTokenizer.load(os.path.join(args.ckpt, "vocab.json"))

    model = TabbyGPT(TabbyConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = tok.encode(args.prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, args.length, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
