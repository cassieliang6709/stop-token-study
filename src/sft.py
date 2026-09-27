"""Chat SFT on top of a pretrained arm, with the same data and hyperparameters for every run.

mode lora        MiniMind LoRA (rank 16 on the square Linear layers); embeddings / lm_head frozen,
                 i.e. the <|im_end|> row cannot change.
mode lora_embed  lora + trainable token embeddings (tied to lm_head), the fix found on Qwen2.5-0.5B.

Usage: python src/sft.py --base out/pretrain_B.pth --mode lora --data data/sft_train.jsonl --out out/sft_B_lora.pth
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

MINIMIND = os.environ.get("MINIMIND_DIR", "minimind")
sys.path.insert(0, MINIMIND)
from dataset.lm_dataset import SFTDataset  # noqa: E402
from model.model_lora import apply_lora  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


def build(base, mode, hidden_size=768, layers=8):
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=layers))
    model.load_state_dict(torch.load(base, map_location="cpu"))
    model = model.cuda()
    apply_lora(model, rank=16)
    for name, p in model.named_parameters():
        p.requires_grad = "lora" in name or (mode == "lora_embed" and "embed_tokens" in name)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--mode", choices=["lora", "lora_embed"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=340)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(os.path.join(MINIMIND, "model"))
    model = build(args.base, args.mode)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"{args.mode}: {sum(p.numel() for p in trainable) / 1e6:.2f}M trainable", flush=True)

    ds = SFTDataset(args.data, tok, max_length=args.max_len)
    g = torch.Generator().manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=8, drop_last=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    log = open(args.out.replace(".pth", "_loss.jsonl"), "w")
    step, t0 = 0, time.time()
    model.train()
    for _ in range(args.epochs):
        for x, y in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 100 == 0:
                rec = {"step": step, "total": len(dl) * args.epochs, "loss": loss.item(), "min": round((time.time() - t0) / 60, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
            step += 1
    torch.save(model.state_dict(), args.out)  # includes the lora sub-modules
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
