#!/bin/bash
# Full run on one RTX 4090: data -> 2 pretrain arms in parallel -> 3 LoRA SFT runs -> probe -> pack -> shutdown
cd /root/autodl-tmp/stop-token-study; mkdir -p data out results logs
export PATH=/root/miniconda3/bin:$PATH HF_ENDPOINT=https://hf-mirror.com MINIMIND_DIR=$PWD/minimind
T=logs/timing.log
t() { local n=$1; shift; local S=$(date +%s); "$@"; local rc=$?; echo "[$(date +%T)] $n $(( $(date +%s)-S ))s rc=$rc" >> $T; return $rc; }

[ -s data/sft_holdout.jsonl ] || {
  t download hf download jingyaogong/minimind_dataset pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl --repo-type dataset --local-dir data
  head -n 100000 data/sft_t2t_mini.jsonl > data/sft_train.jsonl   # SFT on the first 100k conversations
  tail -n 2000 data/sft_t2t_mini.jsonl > data/sft_holdout.jsonl   # probe on conversations never trained on
}
git -C minimind rev-parse HEAD > results/minimind_commit.txt

t pretrain bash -c "python src/pretrain.py --arm A --data data/pretrain_t2t_mini.jsonl --out out/pretrain_A.pth > logs/pretrain_A.log 2>&1 &
                    python src/pretrain.py --arm B --data data/pretrain_t2t_mini.jsonl --out out/pretrain_B.pth > logs/pretrain_B.log 2>&1 &
                    wait"
for run in "A lora" "B lora" "B lora_embed"; do
  set -- $run
  t sft_$1_$2 bash -c "python src/sft.py --base out/pretrain_$1.pth --mode $2 --data data/sft_train.jsonl --out out/sft_$1_$2.pth > logs/sft_$1_$2.log 2>&1"
done
t probe bash -c "python src/probe.py --holdout data/sft_holdout.jsonl --out results/probe.json --max_new 512 \
  A_pretrain=out/pretrain_A.pth B_pretrain=out/pretrain_B.pth \
  A_lora=out/pretrain_A.pth:out/sft_A_lora.pth B_lora=out/pretrain_B.pth:out/sft_B_lora.pth \
  B_lora_embed=out/pretrain_B.pth:out/sft_B_lora_embed.pth > logs/probe.log 2>&1"
cp out/*_loss.jsonl results/
tar czf /root/autodl-tmp/stop_token_results.tgz results logs src scripts
echo "[$(date +%T)] PACKED" >> $T
touch /root/autodl-tmp/STOP_DONE
sleep 900; shutdown
