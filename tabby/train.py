"""Train TabbyGPT on a text file.

Usage:
    python train.py                         # trains on data/input.txt
    python train.py --data my.txt --steps 3000
"""
import argparse
import os
import time

import torch

from model import TabbyConfig, TabbyGPT
from tokenizer import CharTokenizer


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters=20):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    print(f"chars: {len(data):,}  vocab: {tok.vocab_size}")

    cfg = TabbyConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = TabbyGPT(cfg).to(device)
    print(f"params: {model.num_params() / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(splits["train"], args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss(model, splits, args.block_size, args.batch_size, device)
            print(f"step {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} | {time.time() - t0:.1f}s")

    os.makedirs(args.out, exist_ok=True)
    torch.save({"config": cfg.__dict__, "model": model.state_dict()}, os.path.join(args.out, "tabby.pt"))
    tok.save(os.path.join(args.out, "vocab.json"))
    print(f"saved to {args.out}/")


if __name__ == "__main__":
    main()
