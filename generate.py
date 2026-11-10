"""Generate text from the saved checkpoint.

python generate.py "Once upon a time" --temperature 0.8 --top_p 0.9 --tokens 300
"""

import argparse

import torch

from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default="Once upon a time")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--checkpoint", default="out/ckpt.pt")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    chars = ckpt["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    unknown = [c for c in args.prompt if c not in stoi]
    if unknown:
        print(f"Note: these characters were not in the training text and are skipped: {unknown}")
    idx = torch.tensor([[stoi[c] for c in args.prompt if c in stoi] or [0]])
    torch.manual_seed(0)
    out = model.generate(idx, args.tokens, args.temperature, args.top_p)
    print("".join(chars[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
