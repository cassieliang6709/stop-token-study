"""Pretrain a MiniMind model from scratch under one of two document-boundary schemes.

arm A (MiniMind default): <|im_start|> text <|im_end|>
    The chat end-of-turn token is a pretraining target at every document end.
arm B (Qwen-base-like):   text <|endoftext|>
    Chat tokens never appear (any literal occurrences are stripped from the text), so
    <|im_end|> is only ever seen as a non-target row of the tied embedding / lm_head.

Everything else (data order, model, optimizer, schedule, seed) is identical.

Usage: python src/pretrain.py --arm A --data data/pretrain_t2t_mini.jsonl --out out/pretrain_A.pth
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

MINIMIND = os.environ.get("MINIMIND_DIR", "minimind")
sys.path.insert(0, MINIMIND)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


class DocDataset(Dataset):
    def __init__(self, path, tok, arm, max_len):
        self.texts = [json.loads(line)["text"] for line in open(path, encoding="utf-8")]
        self.tok, self.arm, self.max_len = tok, arm, max_len
        self.bos, self.eos = tok.bos_token_id, tok.eos_token_id  # <|im_start|>, <|im_end|>
        self.eot = tok.convert_tokens_to_ids("<|endoftext|>")
        self.pad = tok.pad_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        text = self.texts[i]
        if self.arm == "B":
            text = text.replace("<|im_start|>", "").replace("<|im_end|>", "")
        ids = self.tok(text, add_special_tokens=False, max_length=self.max_len - 2, truncation=True).input_ids
        ids = [self.bos] + ids + [self.eos] if self.arm == "A" else ids + [self.eot]
        n = len(ids)
        x = ids + [self.pad] * (self.max_len - n)
        # mask by position, not token id: in arm B the document end <|endoftext|> is also the pad id
        y = ids + [-100] * (self.max_len - n)
        return torch.tensor(x), torch.tensor(y)


def lr_at(step, total, lr):
    # MiniMind's schedule: cosine from lr down to 0.1 * lr
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max_len", type=int, default=340)
    ap.add_argument("--hidden_size", type=int, default=768)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(os.path.join(MINIMIND, "model"))
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.layers)).cuda()
    print(f"arm {args.arm}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params", flush=True)

    ds = DocDataset(args.data, tok, args.arm, args.max_len)
    im_marks = sum(t.count("<|im_end|>") for t in ds.texts)
    print(f"{len(ds)} documents; literal <|im_end|> in raw text: {im_marks}", flush=True)
    g = torch.Generator().manual_seed(args.seed)  # same document order in both arms
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=8, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)  # weight_decay 0.01, as in MiniMind
    total = args.epochs * len(dl)
    log = open(args.out.replace(".pth", "_loss.jsonl"), "w")
    step, t0 = 0, time.time()
    model.train()
    for _ in range(args.epochs):
        for x, y in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total, args.lr)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, labels=y).loss / args.accum
            loss.backward()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step % 200 == 0:
                rec = {"step": step, "total": total, "loss": loss.item() * args.accum, "min": round((time.time() - t0) / 60, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
            step += 1
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
