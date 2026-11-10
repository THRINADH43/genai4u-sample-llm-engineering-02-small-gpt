"""Character-level training with separate train/validation sets and checkpoints.

python train.py --steps 2000            # start (or resume if out/ckpt.pt exists)
python train.py --steps 3000 --resume   # carry on to step 3000
python train.py --sample "Once upon a time" --temperature 0.8 --top_p 0.9
"""

import argparse
import json
import os
import time

import torch

from model import GPT, GPTConfig

OUT = "out"


def load_data(path="data/input.txt", val_fraction=0.1):
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(data) * (1 - val_fraction))
    # the last 10% of the book is held out; it is never used for training
    return data[:n], data[n:], chars


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, iters=20):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, block_size, batch_size))[1].item() for _ in range(iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def save_checkpoint(model, opt, step, chars, best_val):
    os.makedirs(OUT, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "config": model.cfg.__dict__, "chars": chars, "best_val": best_val},
               os.path.join(OUT, "ckpt.pt"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sample", type=str, default=None, help="only generate from the checkpoint")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--tokens", type=int, default=300)
    args = p.parse_args()
    torch.manual_seed(1337)

    train_data, val_data, chars = load_data()
    ckpt_path = os.path.join(OUT, "ckpt.pt")
    cfg = GPTConfig(vocab_size=len(chars))
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    step, best_val = 0, float("inf")
    if (args.resume or args.sample) and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model = GPT(GPTConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        opt.load_state_dict(ckpt["opt"])
        step, best_val, chars = ckpt["step"], ckpt["best_val"], ckpt["chars"]
        print(f"resumed from step {step}")

    if args.sample is not None:
        stoi = {c: i for i, c in enumerate(chars)}
        idx = torch.tensor([[stoi.get(c, 0) for c in args.sample]])
        model.eval()
        out = model.generate(idx, args.tokens, args.temperature, args.top_p)
        print("".join(chars[i] for i in out[0].tolist()))
        return

    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, vocab {len(chars)}")
    log = open(os.path.join("results", "train_log.jsonl"), "a")
    t0 = time.time()
    while step < args.steps:
        if step % args.eval_every == 0:
            losses = estimate_loss(model, {"train": train_data, "val": val_data}, cfg.block_size, args.batch_size)
            print(f"step {step}: train {losses['train']:.3f}  val {losses['val']:.3f}  ({time.time() - t0:.0f}s)")
            log.write(json.dumps({"step": step, **losses}) + "\n")
            log.flush()
            if losses["val"] < best_val:
                best_val = losses["val"]
            save_checkpoint(model, opt, step, chars, best_val)
        x, y = get_batch(train_data, cfg.block_size, args.batch_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
    losses = estimate_loss(model, {"train": train_data, "val": val_data}, cfg.block_size, args.batch_size)
    print(f"step {step}: train {losses['train']:.3f}  val {losses['val']:.3f}")
    log.write(json.dumps({"step": step, **losses}) + "\n")
    save_checkpoint(model, opt, step, chars, min(best_val, losses["val"]))


if __name__ == "__main__":
    main()
