# NOTE — Probabilistic-circuit rewrite for property compliance

**Scope of this change:** `src/probabilistic_circuits/probabilistic_circuits.py` (the
canonical/active module). The legacy `probabilistic_circuits_tree.py` is **old code and
was intentionally left untouched**; nothing active imports it.

**Goal:** make the per-pixel class-conditional density circuits *valid, tractable*
probabilistic circuits — i.e. provably satisfy the four PC structural properties
(Choi, Vergari & Van den Broeck, 2020). Tractable arbitrary **marginals** are the explicit
requirement of the outlier-explanation method of Khosravi, Vergari & Van den Broeck,
*"Why Is This an Outlier?"* (TPM 2022, https://openreview.net/pdf?id=dLoGOry8sJR): querying
`p_θ(x_S)` for many feature subsets `S` is exact and linear-time **iff** the circuit is
smooth + decomposable. The old structure destroyed both, so those queries were unsound.

---

## The four properties and how each is now enforced

| # | Property | Enforced in | Checked by |
|---|----------|-------------|-----------|
| 1 | **Smoothness** (sum children share one scope) | `SumNode.__init__`, `GateNode`, `ResidualNode` | `validate_smoothness` |
| 2 | **Decomposability** (product children have disjoint scopes) | `ProductNode.__init__` | `validate_decomposability` |
| 3 | **Determinism** (≤1 non-zero child per sum) | *intentionally NOT satisfied* — soft mixtures of full-support leaves | `validate_determinism` (documents the expected failure) |
| 4 | **Structured decomposability** (all products split scope per one vtree) | built top-down from one shared vtree | `validate_structured_decomposability` |

**Normalization (Z = 1)** is now a *consequence* of (1)+(2)+normalized leaves+softmax
mixture weights, not a separate hack.

---

## What was broken before (HEAD) and what fixed it

### 1. `InputNode` multiplied its log-density by a learnable `gate` → Z ≠ 1
- **Before:** `self.gate = nn.Parameter(1.0)`, `return log_mix * self.gate`. Scaling a
  normalized log-density by an arbitrary learnable scalar destroys normalization; the
  circuit was no longer a density.
- **After:** the gate scaling is removed. Leaves are normalized univariate mixtures over
  {Gaussian, Laplace, Student-t}. The `gate` attribute is **kept and fixed to 1.0** only
  for checkpoint / diagnostic back-compat.

### 2. Layered `pairwise` / `prod_layer` / `branchwise` / `sum_layer` build → smoothness + decomposability broken
- **Before:** product/sum layers were stacked by depth parity over the *same* feature
  nodes with **no scope tracking**. Product children shared variables (not decomposable)
  and sum children had mismatched scopes (not smooth). No vtree existed, so structured
  decomposability was undefined.
- **After:** structure is built **top-down from a single shared balanced vtree** over the
  `C` channel features (`random_balanced_vtree`, `VtreeLeaf` / `VtreeInternal`):
  - `_build_region(vnode)` returns nodes whose scope is exactly `vnode.scope`.
  - Leaf region → `n_leaf_components` `InputNode`s for that one feature.
  - Internal region → `ProductNode`s over (left × right) **disjoint** scopes, mixed by
    `n_sum_components` `SumNode`s of **identical** scope.
  - Every product follows the same vtree split → structured decomposability by construction.

### 3. Scope was never represented → no property could be checked
- **After:** every `Node` carries `self.scope: FrozenSet[int]`; constructors **raise** on
  violation; standalone validators (`validate_smoothness`, `validate_decomposability`,
  `validate_circuit`, `validate_determinism`, `validate_structured_decomposability`)
  recompute scopes with id-memoization (DAG-safe) and assert each property.
- `ProbabilisticCircuit.validate()` runs smoothness + decomposability (+ structured
  decomposability when `self.vtree` is set) on each per-class density circuit.

### 4. `ClassifierNode` clarified as a discriminative head
- It stacks the per-class **density** circuits into `(B, n_classes, H, W)` class-conditional
  log-likelihoods (argmax → segmentation). The four properties are asserted on its
  *children* (the real PCs), not on the stack — documented in code and in `validate()`.

---

## DAG correctness

Vtree sub-circuits are shared across many parents, so the circuit is a DAG. Naive recursion
would recompute shared nodes exponentially. Every `evaluate` now uses `Node._memoize`
(keyed by `id(self)` per forward pass); scope computation and all validators are likewise
id-memoized.

---

## Public contract preserved (no caller changes needed)

- `PCNet(n_classes, distribution, device, max_depth, max_branching, seed, cv_module, ...)`
  — same signature. `max_depth` is retained but **no longer caps recursion** (capping would
  drop variables and break scope coverage); depth follows data dimensionality. `max_branching`
  still sets default region width when `n_sum_components` / `n_leaf_components` are `None`.
- `init_network(inputs (B,C,H,W), labels)` and `evaluate(x) → (B, n_classes, H, W)` unchanged.
- `MultiPCNet` / `SubPCNet` / selector nodes retained.

---

## Verification (current state)

Structural smoke test passes on a freshly built `PCNet`:
- `evaluate` output shape `(B, 2, H, W)` ✓
- smoothness + decomposability + structured decomposability hold on every class circuit ✓
- determinism correctly **fails** (expected — soft mixtures) ✓

See `hands_off.md` for how to re-run.
