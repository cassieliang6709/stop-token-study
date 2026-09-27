"""Run 2 data: keep only conversations that fit inside the SFT training length, so every training
example ends with <|im_end|>. Also reports how many run-1 examples were truncated.

Usage: python src/filter_sft.py data/sft_t2t_mini.jsonl data/sft_train_fit.jsonl
"""
import json
import os
import sys

from transformers import AutoTokenizer

MINIMIND = os.environ.get("MINIMIND_DIR", "minimind")
LIMIT = 300    # SFT max_len is 340; leave room for the random system prompt MiniMind adds to 20% of samples
N_TRAIN = 100_000
HOLDOUT = 2_000  # the last 2,000 lines are the probe holdout; never train on them

src, dst = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(os.path.join(MINIMIND, "model"))
lines = open(src, encoding="utf-8").readlines()[:-HOLDOUT]


def n_tokens(line):
    conv = json.loads(line)["conversations"]
    msgs = [{"role": m["role"], "content": m["content"]} for m in conv]
    text = tok.apply_chat_template(msgs, tokenize=False).replace("<think>\n\n</think>\n\n", "")
    return len(tok(text).input_ids)


run1_long = sum(n_tokens(l) > 340 for l in lines[:N_TRAIN])
kept = []
for line in lines:
    if n_tokens(line) <= LIMIT:
        kept.append(line)
        if len(kept) == N_TRAIN:
            break
open(dst, "w", encoding="utf-8").writelines(kept)
print(f"run 1: {run1_long}/{N_TRAIN} training conversations exceeded 340 tokens (end token truncated)")
print(f"run 2: kept {len(kept)} conversations <= {LIMIT} tokens")
