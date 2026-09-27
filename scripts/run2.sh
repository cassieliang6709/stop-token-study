#!/bin/bash
# Run 2: same pretrained checkpoints, SFT only on conversations that fit in max_len -> probe -> pack -> shutdown
cd /root/autodl-tmp/stop-token-study; mkdir -p results/run2 logs/run2
export PATH=/root/miniconda3/bin:$PATH MINIMIND_DIR=$PWD/minimind
T=logs/run2/timing.log
t() { local n=$1; shift; local S=$(date +%s); "$@"; local rc=$?; echo "[$(date +%T)] $n $(( $(date +%s)-S ))s rc=$rc" >> $T; return $rc; }

t filter bash -c "python src/filter_sft.py data/sft_t2t_mini.jsonl data/sft_train_fit.jsonl 2>&1 | grep -E '^run' | tee results/run2/filter_stats.txt"
for run in "A lora" "B lora" "B lora_embed"; do
  set -- $run
  t sft2_$1_$2 bash -c "python src/sft.py --base out/pretrain_$1.pth --mode $2 --data data/sft_train_fit.jsonl --out out/sft2_$1_$2.pth > logs/run2/sft_$1_$2.log 2>&1"
done
t probe2 bash -c "python src/probe.py --holdout data/sft_holdout.jsonl --out results/run2/probe.json --max_new 512 \
  A_lora=out/pretrain_A.pth:out/sft2_A_lora.pth B_lora=out/pretrain_B.pth:out/sft2_B_lora.pth \
  B_lora_embed=out/pretrain_B.pth:out/sft2_B_lora_embed.pth > logs/run2/probe.log 2>&1"
for f in out/sft2_*_loss.jsonl; do cp $f results/run2/; done
tar czf /root/autodl-tmp/stop_token_run2.tgz results logs src scripts
echo "[$(date +%T)] PACKED" >> $T
touch /root/autodl-tmp/RUN2_DONE
sleep 900; shutdown
