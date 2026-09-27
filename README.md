# Does a chat model stop if its end token never appeared in pretraining?

Write-up: [writeup.md](writeup.md) (also on [liangyue.site](https://liangyue.site/en/#reading)).

On Qwen2.5-0.5B, plain LoRA SFT produced a model that almost never stopped: its `<|im_end|>` row
was never trained in the base model, and LoRA cannot update embeddings or the LM head
([ai-infra-gsm8k](https://github.com/cassieliang6709/ai-infra-gsm8k)). That result is
observational. This repo tests the cause directly by pretraining two 64M-parameter
[MiniMind](https://github.com/jingyaogong/minimind) models from scratch that differ in one
thing only: whether the chat end token appears in pretraining.

| Arm | Document format in pretraining | `<|im_end|>` during pretraining |
|---|---|---|
| A (MiniMind default) | `<|im_start|> text <|im_end|>` | a prediction target at every document end |
| B (Qwen-base-like) | `text <|endoftext|>` | never appears (0 literal occurrences in the corpus) |

Same corpus (1,270,238 documents, `pretrain_t2t_mini`), document order, seed, optimizer and
schedule; final pretraining loss 2.07 (A) vs. 2.06 (B). Each arm then gets identical LoRA SFT
(rank 16 on the square projections; embeddings / LM head frozen) on 100k MiniMind chat
conversations. Arm B also gets a second run with the token embeddings trainable (tied to the
LM head), the fix that worked on Qwen.

The training loops (`src/pretrain.py`, `src/sft.py`) are plain PyTorch; only MiniMind's model
definition, tokenizer, chat dataset and LoRA module are reused.

## Results

200 held-out prompts, greedy decoding. A response is a runaway if `<|im_end|>` is not produced
within 512 tokens. Run 2 repeats the SFT step on conversations that fit in the training length
(see caveats); the pretrained checkpoints are the same.

| Model | runaway, run 1 | runaway, run 2 | P(`<|im_end|>`) after answer, median (run 2) | rank of `<|im_end|>` (run 2) |
|---|---|---|---|---|
| A, pretrained only | 0 / 200 | — | 0.034 (run 1) | 5 (run 1) |
| B, pretrained only | 200 / 200 | — | 0.0000004 (run 1) | 5,261 (run 1) |
| A + LoRA SFT | 96 / 200 | **52 / 200** | 0.010 | 17 |
| B + LoRA SFT | 149 / 200 | **93 / 200** | 0.000019 | 330 |
| B + LoRA SFT, embeddings trainable | 133 / 200 | 78 / 200 | 0.0014 | 50 |

- **The end token has to be learned in pretraining for LoRA SFT to use it.** With identical LoRA
  SFT, the model whose pretraining never contained `<|im_end|>` runs away almost twice as often:
  93 vs. 52 of 200 in run 2 (z = 4.4), 149 vs. 96 in run 1 (z = 5.7). At the end of a reference
  answer it assigns the token about 500× less probability (median 0.000019 vs. 0.010).
- **Training the embeddings only partly repairs it**: rank 330 → 50 and runaways 93 → 78 in
  run 2, 149 → 133 in run 1. Neither runaway difference is significant at n = 200 (z = 1.5, 1.8).
  On Qwen2.5-0.5B the same fix was close to complete (P(stop) 0.0003 → 0.9998); a 64M model
  trained for 3k SFT steps recovers much less.
- Pretraining loss is unaffected by the choice of document-end token (2.07 vs. 2.06), and SFT
  loss is matched between A and B (1.85 vs. 1.86 in run 2).

## Caveats

- **Run 1 SFT truncated the end token from 49% of its training conversations** (49,096 of
  100,000 exceed the 340-token length), which teaches every arm to keep going. Run 2 trains only on
  conversations of ≤ 300 tokens; runaways roughly halve in every arm and the A/B gap remains.
- Even arm A runs away on 52/200 prompts in run 2. The generation budget (512) exceeds the
  training length (≤ 340), and greedy decoding in a 64M model may fall into repetition loops (not yet checked in the outputs).
- One seed per arm; n = 200 prompts.

## Reproduce

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind.git   # tested at commit in results/minimind_commit.txt
bash scripts/run.sh     # run 1: data, pretrain A and B in parallel, 3 SFT runs, probe (≈ 2.7 h on one RTX 4090)
bash scripts/run2.sh    # run 2: filtered SFT data, reuses the pretrained checkpoints (≈ 40 min)
```

| Path | Contents |
|---|---|
| `src/pretrain.py` | From-scratch pretraining, arm A or B |
| `src/sft.py` | LoRA / LoRA + embedding chat SFT |
| `src/probe.py` | Row norms, P(end) after reference answers, runaway rate |
| `src/filter_sft.py` | Run 2 data: conversations that fit in the training length |
| `results/probe.json`, `results/run2/probe.json` | Raw probe output, run 1 and run 2 |
| `results/*_loss.jsonl` | Training loss curves |
| `logs/` | Full training and probe logs, timing |
