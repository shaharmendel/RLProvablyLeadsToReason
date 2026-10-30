Code for reproducing the experiments in the paper "Outcome-Based RL Provably Leads Transformers to Reason, but Only With the Right Data".

Each setting uses its own environment. Start its setup commands from the repository root.

## 1. Real-world pre-training experiments

### Environment

To set up the environment, run the following commands from the repository root:

```bash
conda create -n real_world_llm python=3.11
conda activate real_world_llm
pip install torch==2.10.0
pip install numpy==2.2.6 psutil packaging ninja
pip install -r requirements_real_world_llm.txt --no-build-isolation
```

Run from `real_world_llm_experiments/` in `real_world_llm`.

`Qwen/Qwen2.5-3B-Instruct` is publicly accessible on Hugging Face.
To use `meta-llama/Llama-3.2-3B-Instruct`, request access through your
Hugging Face account:

1. Sign in and open
   [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
2. Fill in the "LLAMA 3.2 COMMUNITY LICENSE AGREEMENT" form on the model card
   and submit it. Requests are reviewed by Meta, usually within a few hours.
3. Once the card shows you have been granted access, create a token with `read`
   permission at
   [Settings → Access Tokens](https://huggingface.co/settings/tokens).

Export that token before training; `scripts/train.sh` uses it to warm the
Hugging Face cache before the ranks start.

```bash
export HF_TOKEN=hf_...
cd real_world_llm_experiments
```

### Train

To train all three configurations for a model family, pass `qwen` or `llama` to
`./scripts/run_all_training.sh`:

```bash
./scripts/run_all_training.sh qwen
./scripts/run_all_training.sh llama
```

To train a specific configuration, pass any configuration file from the table
below to `./scripts/train.sh`. For example:

```bash
./scripts/train.sh configs/qwen-15-uniform.yaml
```

| Config | Model | Nodes | Training distribution |
| --- | --- | --- | --- |
| `configs/qwen-10-uniform.yaml` | Qwen2.5-3B-Instruct | 10 | 10-uniform |
| `configs/qwen-15-uniform.yaml` | Qwen2.5-3B-Instruct | 15 | 15-uniform |
| `configs/qwen-15-hard.yaml` | Qwen2.5-3B-Instruct | 15 | 15-hard |
| `configs/llama-10-uniform.yaml` | Llama-3.2-3B-Instruct | 10 | 10-uniform |
| `configs/llama-15-uniform.yaml` | Llama-3.2-3B-Instruct | 15 | 15-uniform |
| `configs/llama-15-hard.yaml` | Llama-3.2-3B-Instruct | 15 | 15-hard |

Checkpoints go to `runs/<config-name>/`, sampled generations to
`completion_samples/completion_samples_dag_<timestamp>.txt`, console output to
`logs/`. Override the defaults with `NUM_PROCS`, `CUDA_VISIBLE_DEVICES`, and
`ACCEL_CFG`.

### Evaluate

To evaluate these configurations, pass `qwen` or `llama` to
`./scripts/run_eval_matrix.sh`. This evaluates each checkpoint against every
test distribution, using 1000 greedy samples per cell:

```bash
./scripts/run_eval_matrix.sh qwen
./scripts/run_eval_matrix.sh llama
```

Results go to `evaluation_results/<family>_matrix_<timestamp>/`.

### Plot

```bash
./scripts/plot_uniform_vs_hard.sh \
    completion_samples/completion_samples_dag_<uniform_ts>.txt \
    completion_samples/completion_samples_dag_<hard_ts>.txt \
    figures/uniform_vs_hard.png
```

### Ablation: effect of training-data diversity

Compare two uniform mixtures of three ancestor-depth levels using
Qwen2.5-3B-Instruct. Both use 15-node graphs and the same 800-step training budget.

| Config | Ancestor depths |
| --- | --- |
| `configs/qwen-5-10-15-uniform.yaml` | 5, 10, 15 |
| `configs/qwen-13-15-uniform.yaml` | 13, 14, 15 |

From `real_world_llm_experiments/`, pass either configuration from the table
above to `./scripts/train.sh`:

```bash
./scripts/train.sh configs/qwen-5-10-15-uniform.yaml
./scripts/train.sh configs/qwen-13-15-uniform.yaml
```

Evaluate both checkpoints on `15-hard` with 1000 greedy samples each:

```bash
./scripts/run_eval_depth_variants.sh
```

Results go to `evaluation_results/qwen_depth_variants_<timestamp>/`.

To plot training-time accuracy on maximal-depth examples:

```bash
DEPTH=14 \
LABEL1="5-10-15-Uniform" \
LABEL2="13-15-Uniform" \
./scripts/plot_uniform_vs_hard.sh \
    completion_samples/completion_samples_dag_<5-10-15-uniform_ts>.txt \
    completion_samples/completion_samples_dag_<13-15-uniform_ts>.txt \
    figures/5-10-15-uniform_vs_13-15-uniform.png
```

The configs measure ancestor depth in nodes, while `DEPTH` measures it in
edges; therefore a configured depth of 15 is plotted with `DEPTH=14`.

---

## 2. Synthetic pre-training experiments

### Environment

To set up the environment, run the following commands from the repository root:

```bash
cd synthetic_llm_experiments
./bootstrap.sh
./scripts/setup_env.sh
conda activate interplay-rl
```

Run from `synthetic_llm_experiments/` in `interplay-rl`.

Download the pinned dataset and model, build the training files, and
authenticate to the WandB project `composition-10B-op-RL`:

```bash
HF_BIN="$(command -v hf)" ./scripts/download_assets.sh
./scripts/build_datasets.sh
wandb login
./scripts/test.sh
```

### Run

To run a specific experiment, pass any experiment name from the table below
to `./run.sh`. For example:

```bash
./run.sh uniform-op10-20
```

| Experiment | Operation counts | Training rows |
| --- | --- | --- |
| `hard-op20` | 20 | 500,000 |
| `uniform-op10-17` | 10-17 | 400,000 |
| `uniform-op10-20` | 10-20 | 550,000 |

Logs go to `logs/` and checkpoints to `checkpoints/<experiment>/`. The launcher
refuses a nonempty checkpoint directory unless you pass `ALLOW_EXISTING=1`.

### Ablation: effect of training-data diversity

Compare two equally weighted mixtures of four operation counts. Both use
409,196 training examples, the same operation-20 training examples (one quarter of
the data), and 200 training steps.

| Experiment | Operation counts |
| --- | --- |
| `appendix-op10-13-17-20` | 10, 13, 17, 20 |
| `appendix-op17-18-19-20` | 17, 18, 19, 20 |

From `synthetic_llm_experiments/`, prepare the balanced data, then pass either
experiment name from the table above to `./run.sh`:

```bash
python scripts/prepare_diversity_data.py
./run.sh appendix-op10-13-17-20
./run.sh appendix-op17-18-19-20
```

---

## 3. Theoretical setting

### Environment

To set up the environment, run the following commands from the repository root:

```bash
conda create -n trans_rl python=3.9.23
conda activate trans_rl
pip install -r requirements.txt
```

### Run

From the repository root in `trans_rl`, run the script for the experiment you
want. The examples below use `--linear` (the default); replace it with
`--softmax` to use softmax attention:

```bash
# Emergence of efficient reasoning
python theoretically_inspired/exp_train_on_D.py --linear

# Out-of-distribution generalization
python theoretically_inspired/exp_train_on_D_m.py --linear

# Complex tasks require simple tasks
python theoretically_inspired/exp_vanish_grad.py --linear
```

Results go to `result/theoretically_inspired/<script>_linear/` for `--linear`
and `result/theoretically_inspired/<script>/` for `--softmax`.

---

## Credits

The real-world pre-training experiments started from Phil Schmid's
[Mini-R1: Reproduce Deepseek R1 "aha moment"](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb)
tutorial and the
[`run_r1_grpo.py`](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/scripts/run_r1_grpo.py)
script it ships.
The synthetic pre-training experiments build on
[Interplay-LM-Reasoning](https://github.com/Interplay-LM-Reasoning/Interplay-LM-Reasoning).
`bootstrap.sh` checks out revision `40b9d29`, and the runs use their
[composition](https://huggingface.co/datasets/Interplay-LM-Reasoning/composition)
dataset and
[extrapolation_rl](https://huggingface.co/Interplay-LM-Reasoning/extrapolation_rl)
base model. All three pinned revisions are recorded in
`synthetic_llm_experiments/manifests/sources.json`.
