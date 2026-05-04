#Weight-Only Quantization Testbed:

This document explains what is implemented in this repo, how experiments are run, what the saved metrics/plots mean , etc.

## 1) Scope and Objective

This testbed compares **weight-only** quantization methods:

- FP32 baseline
- AbsMax
- ZeroPoint
- MXFP (implemented but somthing is wrong)
- GPTQ (tried implementing)
- NestQuant
- NestQuant + LDLQ (tried but not working properly )



- We quantize weights, reconstruct to FP32, and run normal FP32 CPU inference.
- This just allows us to check and compare algotithms based on errors.
- It does **not** give us any inference speedups.

## 2) Pipeline

For each method and model:

1. Load FP32 model.
2. (If needed) collect calibration statistics.(example we will need to know the original distributions to caliberate for beta)
3. Quantize each target weight tensor once (in-place weight replacement with reconstructed FP32 tensor).
4. Evaluate task metric:
   - CNN: Top-1 / Top-5
   - LLM: Perplexity
5. Save:
   - `results.csv` (model-level summary)
   - `layer_metrics.csv` (per-layer diagnostics)
   - plots (automatic in `run_experiment.py`, and report-grade plots via `eval/report_plots.py`)

## 3) File Responsibilities

- `run_experiment.py`
  - Main orchestrator for single experiment run (LLM or CNN).
  - Handles config, dataset/model loading, quantization loop, evaluation, CSV writing, basic plots.
- `run_llm_orchestrator.py`
  - Sweep runner for LLM experiments (bits/k/m/grid sweeps).
  - Aggregates run outputs.
- `base.py`
  - Base quantizer interface, quantization wrapper, common per-layer report fields, SNR/MSE utilities.
- `surgery.py`
  - Finds quantizable layers and applies quantization to weights.
  - Supports `Linear`, `Conv2d`, and Hugging Face GPT-2 `Conv1D`.
- `metrics.py`
  - CNN accuracy metrics and LLM perplexity routine.
- `plot.py`
  - Basic plot utilities used by `run_experiment.py`.
- `report_plots.py`
  - Report plots from existing CSV runs.
- `quant_testbed\methods\*.py`
  - Method implementations:
    - `absmax.py`
    - `zeropoint.py`
    - `mxfp.py`
    - `gptq.py`
    - `nestquant/quantizer.py`, `nestquant/calibration.py`, `nestquant/e8.py`, `nestquant/hadamard.py`, `nestquant/qa_ldlq.py`

## 4) Models, Datasets, and Current Experimental Choices

## LLM side

- Model: `gpt2` (HF AutoModelForCausalLM)
- Tokenizer: `AutoTokenizer.from_pretrained("gpt2")`
  - For GPT-2 this resolves to GPT-2 tokenizer assets .

- Dataset for calibration/eval: `wikitext`, config `wikitext-103-raw-v1`
  - Calibration data: `train` split (in `get_calibration_data` default)
  - Evaluation PPL: configurable split, typically `validation`

## CNN side

- Model path supports `alexnet`(dataset is too large so not implemented but code can quantize alexnet also ) and `resnet18`.
- Current primary setup: **ResNet-18 on CIFAR-10**.
- Pretrained checkpoint source:
  - Repo: `Phoenix21/resnet18-cifar10-baseline`
  - File: `resnet18_cifar10_baseline.pth`

## 5) Quantization Methods Implemented

## AbsMax (`methods/absmax.py`)

- Symmetric quantization with signed integer range.
- Scale from absolute max.
- Supports global/layer/block granularity.

## ZeroPoint (`methods/zeropoint.py`)

- Asymmetric min-max quantization with zero-point.
- Uses unsigned integer code range `[0, 2^bits-1]`.
- Supports global/layer/block granularity.

## MXFP (`methods/mxfp.py`)
(will be added later)

## GPTQ (`methods/gptq.py`)
(will be added later)

## NestQuant (`methods/nestquant/quantizer.py`)

Implemented (weight-only):

1. Flatten weight rows.
2. Hadamard rotation.(deterministic)
3. Row normalization to target norm `sqrt(n)`.
4. Split into 8D blocks.
5. Build candidate beta pool from block norms:
   - linear or log grid (`grid_type`)
   - range derived from observed norms (not hardcoded beta)
6. For each candidate beta, precompute:
   - block MSE
   - overload membership via finite-codebook test
7. First-beta assignment rule.
8. DP selects `k` scales from `m` candidates.
9. Reconstruct, inverse normalize, inverse rotation.


## NestQuant + LDLQ (`use_ldlq=True`)

(will be added later)


## 6) Beta Universe Calibration: 

 In this implementation, beta candidates are derived from **actual observed block norms** of the current layer's transformed weights:

- Candidate lower/upper bounds are computed from quantiles/max of `||v|| / q`.
- Candidate set is generated as linear/log grid of size `m`.
- So beta is not hardcoded globally; it is layer/data-driven.

## 7) Perplexity Calculation Procedure 

In `eval/metrics.py::calculate_perplexity(...)`:

1. Load dataset split (default validation for eval calls).
2. Remove empty text rows.
3. Concatenate all text into one long string.
4. Tokenize once to one long token vector.
5. Create **non-overlapping** chunks of length `seq_len`.
6. Use up to `num_samples` chunks.
7. For each chunk:
   - call `model(input_ids=chunk, labels=chunk)`
   - collect `outputs.loss` (causal LM cross-entropy internal shift).
8. Compute mean NLL over chunks.
9. Return `exp(mean_nll)`.

So:

- `PPL = exp(avg CE loss per token over sampled chunks)`.
- This is a standard and valid approximation, but not a sliding-window full-corpus perplexity.

## 8) Metric Definitions Used

- Top-1 / Top-5: standard top-k accuracy.
- Accuracy drops:
  - `top1_drop = top1_fp32 - top1_quant`
  - `top5_drop = top5_fp32 - top5_quant`
- Perplexity ratio:
  - `ppl_ratio = ppl_quant / ppl_fp32`
- Loss increase (LLM):
  - `loss_increase = log(ppl_quant / ppl_fp32)`
- Per-layer weight MSE: mean squared error between original and reconstructed weights.
- BPW:
  - method-estimated bits per weight 
- Compression ratio:
  - `32 / bpw` relative to FP32.
- Quantization runtime:
  - measured wall-clock quantization time on CPU during fake quant.





## 9) How to Run

## Single run (LLM)

```powershell
python run_experiment.py --task llm --model gpt2 --methods fp32,absmax,zeropoint,nestquant --bits 4 --k 4 --m 32 --grid-type log --calib-samples 128 --eval-samples 100 --output-dir outputs\llm_4bit_run
```

## Single run (CNN, ResNet18 CIFAR-10 checkpoint)

```powershell
python run_experiment.py --task cnn --cnn-model resnet18 --cnn-dataset cifar10 --cnn-data-dir "D:\appa\Btech Project\quant_testbed\data" --cnn-checkpoint "D:\appa\Btech Project\quant_testbed\outputs\cache\hf_cnn_checkpoints\resnet18_cifar10_baseline.pth" --methods fp32,absmax,zeropoint,nestquant --bits 4 --k 4 --m 32 --grid-type log --eval-samples 10000 --output-dir outputs\resnet18_cifar10_4bit_fixed
```

## Aggregated report plots from CSV (MXFP,GPTQ excluded)

```powershell
python .\eval\report_plots.py --output-dir outputs\report_ready_plots --llm-runs-root outputs\llm_orchestrator\runs
```

## 10) Outputs and How to Interpret

- `results.csv`:
  - one row per method summary.
- `layer_metrics.csv`:
  - one row per quantized layer.

For report plots in `outputs/report_ready_plots`:

- `cnn_pareto_accuracy_vs_bpw.png`:
  - method curves over bit-width; best trade-off should stay high on y-axis at low bpw.
- `cnn_accuracy_drop_4bit.png`:
  - lower bar is better (less degradation).
- `cnn_layerwise_mse_lines_4bit.png`:
  - reveals which depth regions are sensitive.
- `llm_pareto_perplexity_vs_bpw.png`:
  - lower perplexity at low bpw is better.
- `cross_task_retention_3bit.png`:
  - compares method generalization across CNN and LLM in normalized form.
- `nestquant_dp_gain_vs_bits.png`:
  - directly shows DP advantage over greedy.


## 11) Next Improvements

- Add GPTQ and NestQuant+LDLQ .
- Quantize Activation and KV Cache

---

