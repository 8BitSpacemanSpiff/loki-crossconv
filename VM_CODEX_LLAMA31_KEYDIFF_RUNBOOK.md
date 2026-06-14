# VM runbook: Llama 3.1 KeyDiff / CrossCov eviction

This is the command history needed to reproduce the working VM setup for the
`crosscov-ext` branch. It covers the repo checkout, Python environment, Llama 3.1
compatibility shim, portable C4 calibration, CrossCov basis construction, and the
KeyDiff comparison runs.

## 1. Checkout the branch

```bash
git clone -b crosscov-ext https://github.com/8BitSpacemanSpiff/loki-crossconv.git
cd loki-crossconv
```

If the repo already exists:

```bash
cd ~/loki-crossconv
git checkout crosscov-ext
git pull --ff-only
```

## 2. Create the Python environment

Use Python 3.10 if available.

```bash
python3.10 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install sentencepiece
pip install "numpy<2"
```

The repo currently pins `transformers==4.40.2`. Avoid upgrading it unless you are
ready to update the local Llama attention monkeypatches.

`No module named 'axonn'` is harmless for the single-GPU/non-AxoNN runs below.

## 3. Download Llama 3.1 8B Instruct

Llama 3.1 is gated, so authenticate first if needed.

```bash
hf auth login

mkdir -p /home/models

hf download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir /home/models/Llama-3.1-8B-Instruct \
  --local-dir-use-symlinks False
```

Set common variables:

```bash
mkdir -p logs

export MODEL_ID=/home/models/Llama-3.1-8B-Instruct
export MODEL_TYPE=llama
export LAYERS=32
export SEQ=8192
export DATASET_TEST=wikitext-test
export DATASET_CAL=c4
export TRANSFORM_DATASET=c4
export ROT=prerotary
export CAL=64
```

## 4. Patch Llama 3.1 config for 8K runs

`transformers==4.40.2` cannot parse Llama 3.1's extended `rope_scaling` config:

```text
ValueError: `rope_scaling` must be a dictionary with two fields, `type` and `factor`
```

For the 8K WikiText/C4 runs here, patch the local model config once:

```bash
python tools/llama31_8k_compat_config.py /home/models/Llama-3.1-8B-Instruct
```

This backs up the original config to:

```text
/home/models/Llama-3.1-8B-Instruct/config.json.llama31_rope_backup
```

Restore before any >8K experiment:

```bash
python tools/llama31_8k_compat_config.py /home/models/Llama-3.1-8B-Instruct --restore
```

## 5. Optional no-calibration KeyDiff sanity runs

These do not need a CrossCov basis. They verify that the modified Llama path loads
and that KeyDiff/key-norm diagnostics work.

```bash
for MODE in keydiff keynorm; do
  python evaluate_tasks.py \
    --model-id $MODEL_ID \
    --model-type $MODEL_TYPE \
    --sequence-length $SEQ \
    --dataset $DATASET_TEST \
    --use-pca-topk \
    --use-crosscov \
    --crosscov-mode $MODE \
    --keep-tokens 2048 \
    --sink-tokens 16 \
    --recent-window 64 \
    --quiet-diagnostics \
    --log-mass-recall \
    2>&1 | tee logs/llama31_${MODE}_keep2048_no_calib.log
done
```

Observed at `keep=2048`:

```text
keydiff: PPL 152.4489, mass_kept 0.414210
keynorm: PPL 262.5208, mass_kept 0.254861
```

## 6. Save C4 calibration tensors

The original repo had a hardcoded C4 path under `/pscratch`. The branch now checks
`LOKI_C4_PATH` first, then falls back to streaming C4 validation from Hugging Face.

If you have a local C4 sample:

```bash
export LOKI_C4_PATH=/path/to/c4-sample
```

Otherwise stream a bounded number of samples:

```bash
export LOKI_C4_STREAM_SAMPLES=20000
```

Save pre-RoPE calibration tensors:

```bash
export TENSOR_DIR=./tensors_llama31_8b_c4_pre_seq8192_fit${CAL}
export TENSOR_ROOT=${TENSOR_DIR}/${ROT}

LOKI_SKIP_CHUNKS=0 LOKI_MAX_CHUNKS=$CAL python evaluate_tasks.py \
  --sequence-length $SEQ \
  --model-id $MODEL_ID \
  --model-type $MODEL_TYPE \
  --dataset $DATASET_CAL \
  --save-tensors \
  --tensors-dir $TENSOR_DIR \
  --use-topk \
  --top-k 1 \
  --rotary-type $ROT \
  2>&1 | tee logs/llama31_c4_pre_seq8192_fit${CAL}_save_tensors.log
```

Expected tensor root:

```text
./tensors_llama31_8b_c4_pre_seq8192_fit64/prerotary
```

## 7. Build CrossCov basis and Rq

```bash
python pca_analysis/crosscov.py $LAYERS "$TENSOR_ROOT" ./crosscov_llama31_8b_c4_pre_seq8192_fit${CAL}/${ROT}_pooled \
  --pool-gqa \
  --emit-rq \
  --device cuda \
  2>&1 | tee logs/llama31_crosscov_c4_pre_seq8192_fit${CAL}_pooled_emit_rq.log
```

## 8. Prepare runtime basis folder

```bash
rm -rf ./runtime_crosscov_llama31_8b_c4_pre_seq8192_fit${CAL}

mkdir -p ./runtime_crosscov_llama31_8b_c4_pre_seq8192_fit${CAL}/Llama-3.1-8B-Instruct-PCA/$TRANSFORM_DATASET/$ROT

cp -r ./crosscov_llama31_8b_c4_pre_seq8192_fit${CAL}/${ROT}_pooled/key \
  ./runtime_crosscov_llama31_8b_c4_pre_seq8192_fit${CAL}/Llama-3.1-8B-Instruct-PCA/$TRANSFORM_DATASET/$ROT/key

cp -r ./crosscov_llama31_8b_c4_pre_seq8192_fit${CAL}/${ROT}_pooled/query \
  ./runtime_crosscov_llama31_8b_c4_pre_seq8192_fit${CAL}/Llama-3.1-8B-Instruct-PCA/$TRANSFORM_DATASET/$ROT/query
```

Sanity check that Rq was copied:

```bash
ls ./runtime_crosscov_llama31_8b_c4_pre_seq8192_fit${CAL}/Llama-3.1-8B-Instruct-PCA/$TRANSFORM_DATASET/$ROT/key/rq_gram/rq_gram_layer_0.pt
```

## 9. Main eviction comparison

At `keep=2048`:

```bash
PCA_DATA_PATH=./runtime_crosscov_llama31_8b_c4_pre_seq8192_fit${CAL} python run_evict_comparison.py \
  --model-id $MODEL_ID \
  --model-type $MODEL_TYPE \
  --dataset $DATASET_TEST \
  --sequence-length $SEQ \
  --top-r 32 \
  --keep-tokens 2048 \
  --rotary-type $ROT \
  --transform-dataset $TRANSFORM_DATASET \
  --sink-tokens 16 \
  --recent-window 64 \
  --modes evict,keynorm,keydiff \
  --out outputs_evict_llama31_keep2048.json \
  2>&1 | tee logs/evict_llama31_keep2048.log
```

Observed:

```text
Budget: 2048 tokens

metric                 evict     keynorm     keydiff
----------------------------------------------------
mass_kept             0.3554      0.2549      0.4142
ppl                 143.2311    262.5208    152.4489
```

## 10. High-keep KeyDiff comparison

This is the regime where the KeyDiff paper reports strong accuracy.

```bash
for KEEP in 6144 7168 7680 8192; do
  PCA_DATA_PATH=./runtime_crosscov_llama31_8b_c4_pre_seq8192_fit${CAL} python run_evict_comparison.py \
    --model-id $MODEL_ID \
    --model-type $MODEL_TYPE \
    --dataset $DATASET_TEST \
    --sequence-length $SEQ \
    --top-r 32 \
    --keep-tokens $KEEP \
    --rotary-type $ROT \
    --transform-dataset $TRANSFORM_DATASET \
    --sink-tokens 16 \
    --recent-window 64 \
    --modes evict,keynorm,keydiff \
    --out outputs_evict_llama31_keep${KEEP}.json \
    2>&1 | tee logs/evict_llama31_keep${KEEP}.log
done
```

Observed:

```text
keep   evicted   evict PPL   keynorm PPL   keydiff PPL
6144   25.00%    10.3294     10.9555       7.9738
7168   12.50%    7.7713      8.0664        6.8398
7680   6.25%     7.1296      7.2339        6.6584
8192   0.00%     6.5830      6.5830        6.5830
```

Mass kept:

```text
keep   evict mass   keynorm mass   keydiff mass
6144   0.8236       0.7144         0.8938
7168   0.9129       0.8411         0.9554
7680   0.9537       0.9041         0.9799
8192   1.0000       1.0000         1.0000
```

Interpretation:

```text
KeyDiff wins clearly in the high-keep regime.
CrossCov evict beats key-norm, but it does not beat KeyDiff where KeyDiff is designed to work.
At extreme eviction (keep=2048), CrossCov evict slightly beats KeyDiff on PPL despite lower mass kept.
```

## 11. Useful log extraction

```bash
grep -hE "Budget:|mass_kept|ppl" logs/evict_llama31_keep*.log
```

JSON outputs are written as:

```text
outputs_evict_llama31_keep2048.json
outputs_evict_llama31_keep6144.json
outputs_evict_llama31_keep7168.json
outputs_evict_llama31_keep7680.json
outputs_evict_llama31_keep8192.json
```
