# Hands-off — Probabilistic-circuit property rewrite

## Status: DONE (pending your full-pipeline run)

The active circuit module `src/probabilistic_circuits/probabilistic_circuits.py` was
rewritten so the per-class density circuits satisfy the four PC properties (smoothness,
decomposability, determinism handled explicitly, structured decomposability) and are
correctly normalized (Z = 1). See `NOTE.md` for the full rationale and before/after.

`probabilistic_circuits_tree.py` is **old code — left untouched on purpose** (nothing
active imports it).

## What changed (one-liners)
- Removed the `InputNode` learnable `gate` multiplier that broke normalization (`gate`
  kept = 1.0 for back-compat only).
- Replaced the depth-parity `prod_layer`/`sum_layer`/`branchwise` builder with a
  **vtree-driven region graph** (`random_balanced_vtree` + `_build_region`).
- Added explicit `scope` tracking on every node; constructors raise on violation.
- Added validators: `validate_smoothness/decomposability/circuit/determinism/
  structured_decomposability` + `ProbabilisticCircuit.validate()`.
- Made `evaluate` DAG-safe via `Node._memoize`.

## Public API — unchanged
`PCNet(...)`, `init_network(inputs, labels)`, `evaluate(x) → (B, n_classes, H, W)`.
Callers (`pipeline.py`, `experiment.py`, GE builders) need no edits. Note: `max_depth` is
now inert (depth follows data dimensionality); `max_branching` still sets default region
width.

## How to verify

**Correct env: `~/miniconda3/envs/cvad_venv` — it has torch + cv2 + skimage.**
(Do NOT use `expllm_env`; that belongs to the ExpLLMble project and lacks cv2/skimage.)

### 1. Quick structural test (no dataset needed)

```bash
cd /Users/eriknielsen/Documents/UNITN/PHD/MAIN_Project/unsup_CV-AD/GitHub/CVxPC
~/miniconda3/envs/cvad_venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "src")
import torch
import probabilistic_circuits.probabilistic_circuits as pc   # same path experiment.py uses
x = torch.randn(4, 6, 8, 8); labels = torch.randint(0,2,(4,1,8,8))
net = pc.PCNet(input_size=(8,8), n_classes=2, device="cpu", max_branching=2, seed=1)
net.init_network(x, labels)
print("shape:", tuple(net.evaluate(x).shape))            # (4, 2, 8, 8)
net.validate()                                            # built-in property check
for c in net.root.children:
    pc.validate_circuit(c)
    pc.validate_structured_decomposability(c, net.vtree)
print("smoothness + decomposability + structured decomp: OK")
PY
```
Expected: shape `(4, 2, 8, 8)`; "PC validation passed …"; "OK". (Determinism is
intentionally not satisfied.)

### 2. Full training pipeline (needs the dataset)
Entry point is `src/main.py <config.json> <seed>`. The PC algorithms that exercise the
rewritten module are `pcnet`, `pcnet_ge`, `pcnet_ge_unsupervised` (all import
`probabilistic_circuits.probabilistic_circuits`). E.g.:

```bash
cd /Users/eriknielsen/Documents/UNITN/PHD/MAIN_Project/unsup_CV-AD/GitHub/CVxPC
~/miniconda3/envs/cvad_venv/bin/python src/main.py src/config/pcnet.json 90
```

**Requires the dataset** at the config's `data_path` (e.g.
`data/kaggle/crack_segmentation_dataset/train`). There is currently **no `data/` directory**
in the repo — place the dataset there (or edit `data_path`) before running. `validate()` is
already called inside the pipeline after `init_network`, so a real run asserts the properties
on actual feature tensors.

## Remaining / open items
- **Full training run not executed here.** Run `unsup_train` / `train` on a real dataset
  (env with `cv2`, `skimage`, `torch`) to confirm convergence behaves sensibly without the
  gate multiplier. Likelihood magnitudes will differ from old checkpoints (the gate scaling
  is gone) — **old checkpoints are not numerically comparable**; retrain.
- **Marginal queries** (the TPM-2022 outlier-explanation use case) are now sound but a
  helper to marginalize an arbitrary feature subset `S` is not yet written — add a
  `marginal(x, subset)` that drops leaves outside `S` and re-evaluates if/when needed.
- Confirm the chosen vtree (balanced random) is adequate; a data-driven vtree could improve
  structure quality without affecting validity.

## Env notes
- **`~/miniconda3/envs/cvad_venv`** — the project env (torch 2.8.0 + cv2 4.12.0 + skimage).
- `expllm_env` is the *ExpLLMble* project's env (no cv2/skimage) — not for CVxPC.
- This dir (`CVxPC`) is a git repo; `probabilistic_circuits.py` is modified (uncommitted).
  Not committed — commit when you're ready.
