"""Measure whether each model can end its turn.

For every model: norm of the <|im_end|> row (tied embedding / lm_head) vs. ordinary tokens,
P(<|im_end|>) right after a held-out reference answer, and the runaway rate: the share of
held-out prompts where greedy decoding never emits <|im_end|> within --max_new tokens.

Usage: python src/probe.py --holdout data/sft_holdout.jsonl --out results/probe.json \
           A_pretrain=out/pretrain_A.pth B_lora=out/pretrain_B.pth:out/sft_B_lora.pth ...
       (name=pretrain.pth for a pretrained model, name=pretrain.pth:sft.pth for an SFT run)
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer

MINIMIND = os.environ.get("MINIMIND_DIR", "minimind")
sys.path.insert(0, MINIMIND)
sys.path.insert(0, os.path.dirname(__file__))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from sft import build  # noqa: E402


def load(spec):
    if ":" in spec:
        base, sft = spec.split(":")
        model = build(base, "lora")
        model.load_state_dict(torch.load(sft, map_location="cuda"))
    else:
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
        model.load_state_dict(torch.load(spec, map_location="cpu"))
        model = model.cuda()
    return model.eval()


def first_turn(conv):
    user = next(m for m in conv if m["role"] == "user")
    answer = next(m for m in conv if m["role"] == "assistant")
    return user["content"], answer["content"]


@torch.no_grad()
def probe(model, tok, rows, max_new):
    end = tok.eos_token_id
    w = model.lm_head.weight.float()
    norms = w.norm(dim=1)
    ordinary = norms[36:].median().item()  # ids 0-35 are special tokens
    out = {"im_end_norm": norms[end].item(), "im_start_norm": norms[tok.bos_token_id].item(),
           "endoftext_norm": norms[0].item(), "ordinary_median_norm": ordinary}

    probs, ranks, runaway = [], [], 0
    for q, a in rows:
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt + a, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        p = model(ids).logits[0, -1].float().softmax(-1)
        probs.append(p[end].item())
        ranks.append(int((p > p[end]).sum()) + 1)

        pids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        gen = model.generate(pids, max_new_tokens=max_new, do_sample=False, eos_token_id=end)
        runaway += int(end not in gen[0, pids.shape[1]:].tolist())
    probs.sort()
    ranks.sort()
    n = len(rows)
    out.update({"p_end_median": probs[n // 2], "p_end_min": probs[0], "p_end_max": probs[-1],
                "end_rank_median": ranks[n // 2], "runaway": runaway, "n": n, "max_new": max_new})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("models", nargs="+")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(os.path.join(MINIMIND, "model"))
    rows = []
    for line in open(args.holdout, encoding="utf-8"):
        q, a = first_turn(json.loads(line)["conversations"])
        if len(tok(q + a).input_ids) < 300:  # stay inside the 340-token training length
            rows.append((q, a))
        if len(rows) == args.n:
            break
    results = {}
    for arg in args.models:
        name, spec = arg.split("=", 1)
        results[name] = probe(load(spec), tok, rows, args.max_new)
        print(name, json.dumps(results[name]), flush=True)
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
