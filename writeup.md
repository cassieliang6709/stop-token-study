# My LoRA model never stopped talking. The cause was one row of the embedding matrix.

*Yue (Cassie) Liang · September 2026 · code and raw results:
[ai-infra-gsm8k](https://github.com/cassieliang6709/ai-infra-gsm8k),
[stop-token-study](https://github.com/cassieliang6709/stop-token-study)*

I fine-tuned Qwen2.5-0.5B on GSM8K with LoRA and got a model that barely beat the base model:
35.5% on strict match, vs. 34.0% for the base model with a raw-text few-shot prompt. The headline number hid what was actually going on. Most of
the model's wrong answers weren't wrong reasoning. The model gave its answer and then kept
generating until it hit the 2,000-token limit.

This post covers how I traced that to a single token, and how I then tested the explanation by
pretraining two small models from scratch that differ only in that token.

## The symptom

I sorted every miss on the 1,319-question test set into four buckets with simple rules:
runaway (≥ 2,000 tokens), format, arithmetic, and reasoning.

| SFT variant | wrong | runaway |
|---|---|---|
| Base model, chat template | 932 | 84 (9%) |
| LoRA | 851 | **525 (62%)** |
| Full fine-tune | 883 | 54 (6%) |

LoRA made the model fail to stop far more often than either the base model or full fine-tuning.
It also explained an odd gap between the two metrics: lm-eval's strict match reads the *first*
`#### <answer>`, while flexible extract reads the *last* number in the output. For LoRA the two
disagreed by 7.6 points (35.5% vs. 27.9%), because the last number came from text generated
after the answer.

## The hypothesis

The chat template ends every assistant turn with `<|im_end|>`. To stop, the model has to put
high probability on that token. Two facts made me suspect it couldn't:

1. In a base model, chat-template tokens may never have been trained.
2. LoRA on the linear layers doesn't touch the embedding matrix or the LM head. Qwen2.5-0.5B ties
   the two, so the `<|im_end|>` row stays exactly as it was.

I checked the rows directly and measured the probability of `<|im_end|>` right after a
finished reference answer, on 50 training problems.

| Model | `<|im_end|>` row norm | P(`<|im_end|>`) after answer | rank |
|---|---|---|---|
| Base | 0.301 (median token: 0.461) | 0.0000 | 141,238 |
| LoRA | 0.301 (unchanged) | 0.0003 | 356 |
| LoRA + trainable embeddings / LM head | 0.306 | **0.9998** | 1 |

In the base model, `<|im_end|>` and `<|im_start|>` have exactly the same norm, 0.301, well
below the typical token. That pattern fits rows that were initialized and never updated. LoRA
could raise the token from rank 141,238 to rank 356 by changing the hidden states, but not far
enough to be sampled. Once I made the embeddings and LM head trainable
(`modules_to_save`), the probability went to 0.9998, and runaway misses dropped from 525 to 34.

This is a clean story, but it's observational. I changed a training setting and the behavior
changed. That doesn't show the cause was the token having never been trained in *pretraining*.
The fix also changes the number of trainable parameters, the optimizer state and weight tying.

## The controlled experiment

To test the cause, I needed two base models that were identical except for whether the end
token appeared in pretraining. You can't get that from released checkpoints, so I pretrained
two 64M-parameter [MiniMind](https://github.com/jingyaogong/minimind) models myself, using
plain PyTorch training loops.

- **Arm A**: every pretraining document is wrapped as `<|im_start|> text <|im_end|>`, so the end
  token is a prediction target 1.27 million times.
- **Arm B**: documents end with `<|endoftext|>` instead, and `<|im_end|>` never appears. This is
  the situation I suspected in Qwen's base model.

Everything else was the same: corpus (1,270,238 documents, zero literal `<|im_end|>` strings),
document order, seed, optimizer and schedule. Final pretraining loss came out at 2.07 and 2.06.
Both arms then got identical LoRA SFT on 100k chat conversations, and I measured how often each
fails to stop on 200 held-out prompts.

### Run 1 had a bug in the experiment, not the code

| | runaway (of 200) |
|---|---|
| A + LoRA | 96 |
| B + LoRA | 149 |
| B + LoRA + trainable embeddings | 133 |

The gap between A and B was large, but so was A's own runaway rate. A model that had seen the
end token over a million times shouldn't fail to stop half the time. The cause was in my SFT
data. I truncated conversations at 340 tokens, and **49% of them were longer than that**. Half
of the training examples had their `<|im_end|>` cut off, so every arm was being taught that
responses don't end.

### Run 2: every training example ends with the token

I reran SFT on conversations of 300 tokens or fewer, reusing the same two pretrained checkpoints.

| | runaway (of 200) | P(`<|im_end|>`) after answer, median | rank |
|---|---|---|---|
| A + LoRA | **52** | 0.010 | 17 |
| B + LoRA | **93** | 0.000019 | 330 |
| B + LoRA + trainable embeddings | 78 | 0.0014 | 50 |

Runaways roughly halved in every arm, and the A/B gap held: 93 vs. 52 (z = 4.4; run 1 gave
z = 5.7). At the end of an answer, the model that never saw the end token in pretraining gives it
about 500× less probability.

## What didn't reproduce

On Qwen, making the embeddings trainable fixed the problem almost completely. On the 64M model
it only partly helped: the token's rank improved from 330 to 50, but runaways went from 93 to
only 78, a difference that isn't significant at n = 200 (z = 1.5). My guess is that a tiny model
with 3k SFT steps can't learn a new token's role from SFT alone as well as a 0.5B model can. I
haven't tested that. Each arm also has one seed, and even arm A still runs away on 26% of prompts,
partly because the 512-token generation budget is longer than anything it was trained on.

## Takeaways

- **Check the special-token rows before LoRA fine-tuning.** If the chat end token looks untrained
  (its norm matches other unused tokens), either make the embeddings and LM head trainable or pick
  an end token the base model learned in pretraining.
- **Truncation silently removes end tokens.** When a model won't stop, count how many of your SFT
  examples still contain the end token after truncation.
- **Break down aggregate metrics by failure mode.** 35.5% (LoRA) vs. 33.1% (full fine-tuning) looked
  like "about the same." The error breakdown showed they failed in completely different
  ways.

## What's next

Knowing when to stop matters more for agents than for single math answers. An agent that doesn't
stop burns tool calls, and one that stops too early and claims it's done is worse. I'm planning to
build a small sandboxed coding-agent environment where test results decide whether a task is
actually finished. That lets me measure how often an agent claims completion when the tests fail,
and whether prompting alone fixes it.
