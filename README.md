# Loki CrossCov

This repository is a research fork of [hpcgroup/loki](https://github.com/hpcgroup/loki) for evaluating a CrossCov-SVD / query-aware basis for sparse attention.

The original Loki method builds a low-rank basis from the key auto-covariance, roughly `E[k k^T]`, and uses that basis to select the top-k attention keys cheaply. This fork adds a CrossCov basis built from query-key interaction, roughly `E[k q^T]`, and a deployable grouped-query attention (GQA) variant that pools query-head information into one basis per KV head.

The main goal of the fork is to compare:

- Full dense attention
- Loki PCA TopK attention
- CrossCov pooled GQA TopK attention

on Mistral-7B, especially at aggressive compression.

## What Was Added

Important additions in this fork:

- `pca_analysis/crosscov.py`
  - Computes CrossCov-SVD bases.
  - Supports `--pool-gqa` for deployable GQA models such as Mistral.
  - Supports `--device cuda` for faster basis computation.

- `pca_analysis/recall_eval.py`
  - Offline recall harness for Loki and CrossCov bases.
  - Supports runtime-compatible fixed-k recall.

- `methods/pca_topk/crosscov_utils.py`
  - Runtime CrossCov basis loader and sparse selection path.

- `methods/pca_topk/modify_mistral.py`
  - CrossCov runtime path for Mistral/Mixtral-style attention.
  - Optional attention-output reconstruction diagnostics.
  - Uses raw selected attention scores for apples-to-apples Loki vs CrossCov comparison.

- `evaluate_tasks.py`
  - Adds:
    - `--use-crosscov`
    - `--log-recall`
    - `--log-output-error`
    - `--log-mass-recall`
    - `--quiet-diagnostics`

- `methods/common/ppl.py`
  - Adds:
    - `LOKI_MAX_CHUNKS` to cap evaluation/calibration chunks.
    - `LOKI_SKIP_CHUNKS` to evaluate on held-out chunks.
  - Removes forced Hugging Face re-downloads so local model caches are reused.

## Environment

The code was tested on a Jarvis AI Lab instance with an H100 GPU.

Use Python 3.10 if possible.

```bash
cd ~/loki-crossconv

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install sentencepiece
```

For LM Harness downstream tasks, install this additionally:

```bash
pip install lm-eval
```

The original Loki code pins specific versions of `torch`, `transformers`, `datasets`, and related packages. Avoid upgrading `transformers` unless you are ready to update the attention patches.

## Reuse Model Downloads

Downloading Mistral-7B repeatedly wastes a lot of time. Download once and use the local path as `--model-id`.

```bash
mkdir -p /home/models

hf download mistralai/Mistral-7B-v0.1 \
  --local-dir /home/models/Mistral-7B-v0.1 \
  --local-dir-use-symlinks False
```

Use:

```bash
export MODEL_ID=/home/models/Mistral-7B-v0.1
```

The runtime basis folder name is derived from the basename of `MODEL_ID`, so this local path maps to:

```text
Mistral-7B-v0.1-PCA
```

Optional persistent Hugging Face caches:

```bash
mkdir -p /home/hf_cache /home/hf_datasets

cat >> ~/.bashrc <<'EOF'
export HF_HOME=/home/hf_cache
export HF_DATASETS_CACHE=/home/hf_datasets
export TRANSFORMERS_CACHE=/home/hf_cache/transformers
EOF

source ~/.bashrc
```

## Common Variables

Most commands below assume Mistral-7B.

```bash
cd ~/loki-crossconv
mkdir -p logs

export MODEL_ID=/home/models/Mistral-7B-v0.1
export MODEL_TYPE=mistral
export LAYERS=32
```

For paper-style WikiText-2 runs:

```bash
export SEQ=4096
export DATASET_TEST=wikitext-test
export DATASET_CAL=wikitext-valid
export TRANSFORM_DATASET=wikitext
export ROT=prerotary
export CAL=64
```

## Important Flags

Use `--use-pca-topk` for Loki/CrossCov runtime evaluation.

Do not use `--use-topk` for Loki/CrossCov evaluation. `--use-topk` runs the plain TopK baseline and does not load PCA/CrossCov bases.

Loki:

```bash
--use-pca-topk
```

CrossCov:

```bash
--use-pca-topk --use-crosscov
```

Compression parameters:

- `--top-k 0.25` means keep 25% of keys.
- `--top-r 32` means keep 32 projection dimensions.
- For Mistral head dimension 128:
  - `r=8` means `df=0.0625`
  - `r=16` means `df=0.125`
  - `r=32` means `df=0.25`

## Calibration vs Testing

For the main WikiText-2 experiments:

```text
Calibration / basis fitting: WikiText-2 validation split
Testing / PPL reporting:    WikiText-2 test split
```

This keeps the reported PPL held-out. The CrossCov and Loki bases are fit on validation tensors and evaluated on test text.

For chunk-level diagnostics on the same split, use:

- `LOKI_SKIP_CHUNKS=0 LOKI_MAX_CHUNKS=N` for early chunks.
- `LOKI_SKIP_CHUNKS=N LOKI_MAX_CHUNKS=M` for a disjoint held-out chunk range.

## 1. Full Attention Baseline on WikiText-2 Test

This reproduces the full-attention Mistral-7B WikiText-2 test baseline from the Loki paper.

```bash
python evaluate_tasks.py \
  --sequence-length $SEQ \
  --model-id $MODEL_ID \
  --model-type $MODEL_TYPE \
  --dataset $DATASET_TEST \
  2>&1 | tee logs/mistral_wikitext_test_full_seq4096.log
```

Expected result from our run:

```text
tensor(4.9140, device='cuda:0')
```

This matches the Loki paper's reported Mistral-7B full-attention PPL of about `4.91`.

## 2. Save Calibration Tensors

Save pre-RoPE key/query tensors from WikiText-2 validation.

```bash
export TENSOR_DIR=./tensors_mistral7b_loki_pre_seq4096_fit${CAL}
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
  2>&1 | tee logs/mistral_loki_pre_seq4096_fit${CAL}_save_tensors.log
```

Expected tensor layout:

```text
./tensors_mistral7b_loki_pre_seq4096_fit64/prerotary/key
./tensors_mistral7b_loki_pre_seq4096_fit64/prerotary/query
```

## 3. Build Loki PCA Basis

```bash
python pca_analysis/pca.py key $LAYERS "$TENSOR_ROOT" ./loki_pca_mistral7b_pre_seq4096_fit${CAL}/$ROT \
  2>&1 | tee logs/mistral_loki_pre_seq4096_fit${CAL}_pca.log
```

Prepare the runtime folder:

```bash
rm -rf ./runtime_loki_pre_seq4096_fit${CAL}
mkdir -p ./runtime_loki_pre_seq4096_fit${CAL}/Mistral-7B-v0.1-PCA/$TRANSFORM_DATASET/$ROT

cp -r ./loki_pca_mistral7b_pre_seq4096_fit${CAL}/$ROT/key \
  ./runtime_loki_pre_seq4096_fit${CAL}/Mistral-7B-v0.1-PCA/$TRANSFORM_DATASET/$ROT/key
```

## 4. Build CrossCov Pooled GQA Basis

Use the same calibration tensors as Loki.

```bash
python pca_analysis/crosscov.py $LAYERS "$TENSOR_ROOT" ./crosscov_mistral7b_pre_seq4096_fit${CAL}/${ROT}_pooled \
  --pool-gqa --device cuda \
  2>&1 | tee logs/mistral_crosscov_pre_seq4096_fit${CAL}_pooled.log
```

Prepare the runtime folder:

```bash
rm -rf ./runtime_crosscov_pre_seq4096_fit${CAL}
mkdir -p ./runtime_crosscov_pre_seq4096_fit${CAL}/Mistral-7B-v0.1-PCA/$TRANSFORM_DATASET/$ROT

cp -r ./crosscov_mistral7b_pre_seq4096_fit${CAL}/${ROT}_pooled/key \
  ./runtime_crosscov_pre_seq4096_fit${CAL}/Mistral-7B-v0.1-PCA/$TRANSFORM_DATASET/$ROT/key

cp -r ./crosscov_mistral7b_pre_seq4096_fit${CAL}/${ROT}_pooled/query \
  ./runtime_crosscov_pre_seq4096_fit${CAL}/Mistral-7B-v0.1-PCA/$TRANSFORM_DATASET/$ROT/query
```

Why `--pool-gqa` matters:

Mistral uses GQA with 32 query heads and 8 KV heads. A deployable compressed KV cache should store one projected key stream per KV head, not four different projected copies per query head. The pooled CrossCov construction creates one basis per KV head by pooling the query-head cross-covariance energy within each GQA group.

## 5. Replicate Loki Paper Setting

The Loki paper setting:

```text
Mistral-7B
WikiText-2 test
sequence length 4096
pre-RoPE basis
kf = 0.25
df = 0.25
```

For Mistral, `df=0.25` corresponds to `r=32`.

Run Loki:

```bash
PCA_DATA_PATH=./runtime_loki_pre_seq4096_fit${CAL} python evaluate_tasks.py \
  --sequence-length $SEQ \
  --model-id $MODEL_ID \
  --model-type $MODEL_TYPE \
  --dataset $DATASET_TEST \
  --use-pca-topk \
  --top-k 0.25 \
  --top-r 32 \
  --rotary-type $ROT \
  --transform-dataset $TRANSFORM_DATASET \
  2>&1 | tee logs/mistral_wikitext_test_loki_pre_kf0.25_r32_seq4096_fit${CAL}.log
```

Run CrossCov:

```bash
PCA_DATA_PATH=./runtime_crosscov_pre_seq4096_fit${CAL} python evaluate_tasks.py \
  --sequence-length $SEQ \
  --model-id $MODEL_ID \
  --model-type $MODEL_TYPE \
  --dataset $DATASET_TEST \
  --use-pca-topk \
  --use-crosscov \
  --top-k 0.25 \
  --top-r 32 \
  --rotary-type $ROT \
  --transform-dataset $TRANSFORM_DATASET \
  2>&1 | tee logs/mistral_wikitext_test_crosscov_pre_kf0.25_r32_seq4096_fit${CAL}.log
```

Extract results:

```bash
grep -hE "tensor\\(" \
  logs/mistral_wikitext_test_full_seq4096.log \
  logs/mistral_wikitext_test_loki_pre_kf0.25_r32_seq4096_fit${CAL}.log \
  logs/mistral_wikitext_test_crosscov_pre_kf0.25_r32_seq4096_fit${CAL}.log
```

Our result:

```text
Full attention:                    4.9140
Loki pre, kf=0.25, df=0.25:         4.9230
CrossCov pooled pre, same setting:  4.9226
```

CrossCov slightly improves over Loki at the exact paper setting, but the gap is small because Loki is already extremely close to full attention at this mild compression point.

## 6. Aggressive Compression Grid

The stronger CrossCov result appears at smaller projection budgets.

Run the grid:

```bash
export LOKI_PATH=./runtime_loki_pre_seq4096_fit${CAL}
export CROSSCOV_PATH=./runtime_crosscov_pre_seq4096_fit${CAL}

for KF in 0.125 0.25; do
  for R in 8 16 32; do
    echo "===== Loki KF=$KF R=$R ====="
    PCA_DATA_PATH=$LOKI_PATH python evaluate_tasks.py \
      --sequence-length $SEQ \
      --model-id $MODEL_ID \
      --model-type $MODEL_TYPE \
      --dataset $DATASET_TEST \
      --use-pca-topk \
      --top-k $KF \
      --top-r $R \
      --rotary-type $ROT \
      --transform-dataset $TRANSFORM_DATASET \
      2>&1 | tee logs/mistral_wikitext_test_loki_pre_kf${KF}_r${R}_seq4096_fit${CAL}.log

    echo "===== CrossCov KF=$KF R=$R ====="
    PCA_DATA_PATH=$CROSSCOV_PATH python evaluate_tasks.py \
      --sequence-length $SEQ \
      --model-id $MODEL_ID \
      --model-type $MODEL_TYPE \
      --dataset $DATASET_TEST \
      --use-pca-topk \
      --use-crosscov \
      --top-k $KF \
      --top-r $R \
      --rotary-type $ROT \
      --transform-dataset $TRANSFORM_DATASET \
      2>&1 | tee logs/mistral_wikitext_test_crosscov_pre_kf${KF}_r${R}_seq4096_fit${CAL}.log
  done
done
```

Extract results:

```bash
for f in logs/mistral_wikitext_test_*_pre_kf*_r*_seq4096_fit${CAL}.log; do
  echo -n "$f  "
  grep -E "tensor\\(" "$f" | tail -n 1
done

echo "logs/mistral_wikitext_test_full_seq4096.log  $(grep -E "tensor\\(" logs/mistral_wikitext_test_full_seq4096.log | tail -n 1)"
```

Our WikiText-2 test results, calibrated on WikiText-2 validation:

| kf | r | df | Loki PPL | CrossCov PPL | CrossCov gain |
|---:|---:|---:|---:|---:|---:|
| 0.125 | 8 | 0.0625 | 7.8481 | 5.4314 | -2.4167 |
| 0.125 | 16 | 0.125 | 5.5736 | 5.0635 | -0.5101 |
| 0.125 | 32 | 0.25 | 4.9636 | 4.9560 | -0.0076 |
| 0.25 | 8 | 0.0625 | 5.2205 | 5.0195 | -0.2010 |
| 0.25 | 16 | 0.125 | 4.9900 | 4.9420 | -0.0480 |
| 0.25 | 32 | 0.25 | 4.9230 | 4.9226 | -0.0004 |

Full attention PPL:

```text
4.9140
```

Interpretation:

CrossCov beats Loki at every tested point. The gain is small when Loki is already near dense attention, but large at aggressive compression.

## 7. Held-Out Fidelity Diagnostics

These diagnostics test whether CrossCov's recall advantage actually improves the attention output.

The diagnostic metrics are:

- PPL gap vs full attention.
- `mass_recall`: fraction of true softmax attention mass recovered by the selected tokens.
- `attn_out_rel_l2`: relative L2 error of sparse attention output vs dense attention output.
- `attn_out_cos`: cosine similarity of sparse attention output vs dense attention output.

For a chunk-level held-out split, fit on chunks `0..31`, evaluate in-sample on chunks `0..15`, and held-out on chunks `32..47`.

```bash
export SEQ=2048
export ROT=postrotary
export DATASET=wikitext-valid
export CAL=32
export EVAL=16
export TENSOR_DIR=./tensors_mistral7b_fit${CAL}
export TENSOR_ROOT=${TENSOR_DIR}/${ROT}
```

Save calibration tensors:

```bash
LOKI_SKIP_CHUNKS=0 LOKI_MAX_CHUNKS=$CAL python evaluate_tasks.py \
  --sequence-length $SEQ \
  --model-id $MODEL_ID \
  --model-type $MODEL_TYPE \
  --dataset $DATASET \
  --save-tensors \
  --tensors-dir $TENSOR_DIR \
  --use-topk \
  --top-k 1 \
  --rotary-type $ROT \
  2>&1 | tee logs/fit${CAL}_save_tensors.log
```

Build bases:

```bash
python pca_analysis/pca.py key $LAYERS "$TENSOR_ROOT" ./loki_pca_mistral7b_fit${CAL}/${ROT} \
  2>&1 | tee logs/fit${CAL}_loki_pca.log

python pca_analysis/crosscov.py $LAYERS "$TENSOR_ROOT" ./crosscov_mistral7b_fit${CAL}/${ROT}_pooled \
  --pool-gqa --device cuda \
  2>&1 | tee logs/fit${CAL}_crosscov_pooled.log
```

Prepare runtime folders:

```bash
rm -rf ./runtime_loki_fit${CAL} ./runtime_crosscov_fit${CAL}

mkdir -p ./runtime_loki_fit${CAL}/Mistral-7B-v0.1-PCA/$DATASET/$ROT
cp -r ./loki_pca_mistral7b_fit${CAL}/${ROT}/key \
  ./runtime_loki_fit${CAL}/Mistral-7B-v0.1-PCA/$DATASET/$ROT/key

mkdir -p ./runtime_crosscov_fit${CAL}/Mistral-7B-v0.1-PCA/$DATASET/$ROT
cp -r ./crosscov_mistral7b_fit${CAL}/${ROT}_pooled/key \
  ./runtime_crosscov_fit${CAL}/Mistral-7B-v0.1-PCA/$DATASET/$ROT/key
cp -r ./crosscov_mistral7b_fit${CAL}/${ROT}_pooled/query \
  ./runtime_crosscov_fit${CAL}/Mistral-7B-v0.1-PCA/$DATASET/$ROT/query
```

Run full baselines:

```bash
LOKI_SKIP_CHUNKS=0 LOKI_MAX_CHUNKS=$EVAL python evaluate_tasks.py \
  --sequence-length $SEQ \
  --model-id $MODEL_ID \
  --model-type $MODEL_TYPE \
  --dataset $DATASET \
  2>&1 | tee logs/full_insample_fit${CAL}.log

LOKI_SKIP_CHUNKS=$CAL LOKI_MAX_CHUNKS=$EVAL python evaluate_tasks.py \
  --sequence-length $SEQ \
  --model-id $MODEL_ID \
  --model-type $MODEL_TYPE \
  --dataset $DATASET \
  2>&1 | tee logs/full_heldout_fit${CAL}.log
```

Run Loki and CrossCov diagnostics at `r=8`, `top_k=0.125`:

```bash
for SPLIT in insample heldout; do
  if [ "$SPLIT" = "insample" ]; then SKIP=0; else SKIP=$CAL; fi

  PCA_DATA_PATH=./runtime_loki_fit${CAL} \
  LOKI_SKIP_CHUNKS=$SKIP \
  LOKI_MAX_CHUNKS=$EVAL \
  python evaluate_tasks.py \
    --sequence-length $SEQ \
    --model-id $MODEL_ID \
    --model-type $MODEL_TYPE \
    --dataset $DATASET \
    --use-pca-topk \
    --top-k 0.125 \
    --top-r 8 \
    --rotary-type $ROT \
    --transform-dataset $DATASET \
    --log-output-error \
    --log-mass-recall \
    --quiet-diagnostics \
    2>&1 | tee logs/loki_${SPLIT}_r8_k0125_fit${CAL}_pca.log

  PCA_DATA_PATH=./runtime_crosscov_fit${CAL} \
  LOKI_SKIP_CHUNKS=$SKIP \
  LOKI_MAX_CHUNKS=$EVAL \
  python evaluate_tasks.py \
    --sequence-length $SEQ \
    --model-id $MODEL_ID \
    --model-type $MODEL_TYPE \
    --dataset $DATASET \
    --use-pca-topk \
    --use-crosscov \
    --top-k 0.125 \
    --top-r 8 \
    --rotary-type $ROT \
    --transform-dataset $DATASET \
    --log-output-error \
    --log-mass-recall \
    --quiet-diagnostics \
    2>&1 | tee logs/crosscov_${SPLIT}_r8_k0125_fit${CAL}_pca.log
done
```

Extract summaries:

```bash
for f in logs/*_fit${CAL}_pca.log logs/full_*_fit${CAL}.log; do
  echo
  echo "===== $f ====="
  grep -E "Running Base|Modifying|CrossCov|Fetching|tensor\\(|Diagnostics summary|attn_out|mass_recall|samples|overall" "$f" | tail -n 40
done
```

Our diagnostic result:

| Split | Full PPL | Loki PPL | CrossCov PPL | Loki relL2 | CrossCov relL2 | Loki mass | CrossCov mass |
|---|---:|---:|---:|---:|---:|---:|
| In-sample | 4.9317 | 6.3195 | 4.9708 | 0.4701 | 0.1424 | 0.8682 | 0.9257 |
| Held-out | 5.2585 | 6.5564 | 5.3115 | 0.4718 | 0.1492 | 0.8599 | 0.9176 |

Interpretation:

The CrossCov advantage does not collapse held-out. It recovers more attention mass, has lower attention-output error, and has a much smaller PPL gap vs full attention.

## 8. Offline Recall Evaluation

After saving tensors and building bases, run recall evaluation.

Loki:

```bash
python pca_analysis/recall_eval.py \
  --tensor-root "$TENSOR_ROOT" \
  --key-basis ./loki_pca_mistral7b_fit${CAL}/${ROT}/key \
  --num-layers $LAYERS \
  --top-r 8 16 32 64 \
  --top-k 0.125 \
  --method loki \
  --max-seqs 16 \
  --runtime-fixed-k \
  --device cuda
```

CrossCov:

```bash
python pca_analysis/recall_eval.py \
  --tensor-root "$TENSOR_ROOT" \
  --key-basis ./crosscov_mistral7b_fit${CAL}/${ROT}_pooled/key \
  --query-basis ./crosscov_mistral7b_fit${CAL}/${ROT}_pooled/query \
  --num-layers $LAYERS \
  --top-r 8 16 32 64 \
  --top-k 0.125 \
  --method crosscov \
  --max-seqs 16 \
  --runtime-fixed-k \
  --device cuda
```

Use `--runtime-fixed-k` when comparing offline recall with the live model path, because the runtime path uses `k=int(top_k * full_sequence_length)`.

## 9. LM Harness Downstream Tasks

The repo has built-in LM Harness support in `evaluate_tasks.py`.

Tasks currently hardcoded:

- MMLU
- GSM8K
- HellaSwag
- WinoGrande
- TruthfulQA MC2
- ARC-Challenge

Install:

```bash
pip install lm-eval
```

Full attention:

```bash
python evaluate_tasks.py \
  --sequence-length 4096 \
  --model-id $MODEL_ID \
  --model-type $MODEL_TYPE \
  --lm-harness-eval \
  2>&1 | tee logs/mistral_lm_harness_full.log
```

Loki:

```bash
PCA_DATA_PATH=./runtime_loki_pre_seq4096_fit${CAL} python evaluate_tasks.py \
  --sequence-length 4096 \
  --model-id $MODEL_ID \
  --model-type $MODEL_TYPE \
  --use-pca-topk \
  --top-k 0.125 \
  --top-r 16 \
  --rotary-type prerotary \
  --transform-dataset wikitext \
  --lm-harness-eval \
  2>&1 | tee logs/mistral_lm_harness_loki_pre_kf0125_r16.log
```

CrossCov:

```bash
PCA_DATA_PATH=./runtime_crosscov_pre_seq4096_fit${CAL} python evaluate_tasks.py \
  --sequence-length 4096 \
  --model-id $MODEL_ID \
  --model-type $MODEL_TYPE \
  --use-pca-topk \
  --use-crosscov \
  --top-k 0.125 \
  --top-r 16 \
  --rotary-type prerotary \
  --transform-dataset wikitext \
  --lm-harness-eval \
  2>&1 | tee logs/mistral_lm_harness_crosscov_pre_kf0125_r16.log
```

## 10. C4 and BookCorpus Notes

The original repo supports `--dataset c4` and `--dataset bookcorpus`, but the local paths are hardcoded in `methods/common/ppl.py`:

```text
/pscratch/sd/p/prajwal/c4-sample
/pscratch/sd/p/prajwal/bookcorpus-sample
```

On a non-Perlmutter machine, either:

- Create those directories/symlinks, or
- Patch `methods/common/ppl.py` to read dataset paths from environment variables.

Helper scripts exist:

```text
helper/downloadc4split.py
helper/downloadbooksplit.py
```

## 11. Compute Benchmark

The original Loki compute benchmark is still available:

```bash
mkdir -p compute_files
python evaluate_compute.py
```

This benchmarks the attention kernel path and writes JSON files under `compute_files`.

## Troubleshooting

### `--use-topk` gives identical Loki/CrossCov results

Use `--use-pca-topk`, not `--use-topk`.

`--use-topk` runs the plain TopK baseline and does not load PCA/CrossCov bases.

### Runtime basis path not found

The loader expects:

```text
$PCA_DATA_PATH/Mistral-7B-v0.1-PCA/<transform_dataset>/<rotary_type>/key/...
```

For CrossCov it also expects:

```text
$PCA_DATA_PATH/Mistral-7B-v0.1-PCA/<transform_dataset>/<rotary_type>/query/...
```

If using local model path `/home/models/Mistral-7B-v0.1`, the model folder remains:

```text
Mistral-7B-v0.1-PCA
```

Make sure `--transform-dataset` matches the folder name used when copying bases. For the paper-style runs above, use:

```bash
--transform-dataset wikitext
```

### `sentencepiece` tokenizer error

Install:

```bash
pip install sentencepiece
```

### Hugging Face CLI

Newer Hugging Face installations use:

```bash
hf auth login
```

instead of:

```bash
huggingface-cli login
```

### `np.float_` removed in NumPy 2.0

Use a NumPy version compatible with this stack, for example:

```bash
pip install "numpy<2"
```

## Summary of Current Main Result

On Mistral-7B WikiText-2 test at sequence length 4096, with bases calibrated on WikiText-2 validation:

- CrossCov matches or slightly improves over Loki at the Loki paper's mild compression setting.
- CrossCov strongly improves over Loki at aggressive compression.
- Held-out diagnostics show the improvement is not just in token recall; CrossCov recovers more true attention mass and produces lower attention-output error.

The clearest headline setting so far:

```text
kf=0.125, r=16, pre-RoPE:
Loki PPL:     5.5736
CrossCov PPL: 5.0635
Full PPL:     4.9140
```

This suggests the CrossCov basis is most useful when the projection budget is small.
