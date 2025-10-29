import torch
import torch.nn.functional as F
import numpy as np
import random, math, os, cv2, time, gc
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Union
import torch.multiprocessing as mp
from multiprocessing import Manager
from torch.utils.data import DataLoader, TensorDataset
from joblib import Parallel, delayed
from probabilistic_circuits.probabilistic_circuits import *

# ============================================================
# --- Grammar specifications for tree/DAG generation ---
# ============================================================

@dataclass
class LeafSpec:
    kind: str
    feature_idx: int

@dataclass
class SumSpec:
    kind: str
    k: int
    children: List["TreeSpec"]

@dataclass
class ProdSpec:
    kind: str
    children: List["TreeSpec"]

@dataclass
class GateSpec:
    kind: str
    children: List["TreeSpec"]

@dataclass
class ResidualSpec:
    kind: str
    children: List["TreeSpec"]

TreeSpec = Union[LeafSpec, SumSpec, ProdSpec]


# ============================================================
# --- Mapper (Codons → DAG grammar) ---
# ============================================================

class GEMapper:
    """
    Grammatical Evolution mapper adapted for DAG-compatible PCNet.
    Sum nodes can now reuse children across different groups (branchwise overlap).
    """
    def __init__(self, max_wraps=2):
        self.max_wraps = max_wraps

    def map_model(self, codons: List[int], n_features: int, max_depth: int, max_branching: int):
        """Map a genome into multiple DAG-compatible trees (e.g. one per class)."""
        specs, idx, wraps = [], 0, 0
        for _ in range(2):  # binary classifier → 2 roots
            tree, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth=0)
            specs.append(tree)
        return specs

    # --------------------- internals ---------------------

    def _next(self, codons, idx, wraps):
        if idx >= len(codons):
            wraps += 1
            if wraps > self.max_wraps:
                return None, idx, wraps, True
            idx = 0
        return codons[idx], idx + 1, wraps, False

    def _fallback_feature(self, idx, depth, n_features):
        if n_features <= 0: return 0
        return (depth * 17 + idx) % n_features

    def _expand(self, codons, idx, wraps, n_features, max_depth, max_branching, depth):
        if depth >= max_depth:
            f = self._fallback_feature(idx, depth, n_features)
            return LeafSpec("input", f), idx, wraps

        c, idx, wraps, exhausted = self._next(codons, idx, wraps)
        if exhausted:
            f = self._fallback_feature(idx, depth, n_features)
            return LeafSpec("input", f), idx, wraps

        node_type = c % 5  # 0 = leaf, 1 = sum, 2 = prod

        # Leaf node
        if node_type == 0:
            f, idx, wraps, exhausted2 = self._next(codons, idx, wraps)
            if exhausted2:
                f = self._fallback_feature(idx, depth, n_features)
            else:
                f = f % n_features
            return LeafSpec("input", f), idx, wraps

        # Sum node (branching + overlapping possible)
        elif node_type == 1:
            kcod, idx, wraps, _ = self._next(codons, idx, wraps)
            k = 2 + (kcod % max(1, max_branching - 1))
            children = []
            for _ in range(k):
                ch, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
                children.append(ch)
            # Randomly duplicate a few children to simulate overlapping connections
            if len(children) > 2 and random.random() < 0.4:
                dup = random.choice(children)
                children.insert(random.randint(0, len(children) - 1), dup)
            return SumSpec("sum", k, children), idx, wraps

        # Product node (can have multiple children now, not just binary)
        elif node_type == 3:  # Gate node (2 children)
            a, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
            b, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
            return GateSpec("gate", [a, b]), idx, wraps

        elif node_type == 4:  # Residual node (2 children)
            base, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
            sub, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
            return ResidualSpec("res", [base, sub]), idx, wraps
        else:
            n_child = 2 + (c % max(1, max_branching - 1))
            children = []
            for _ in range(n_child):
                ch, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
                children.append(ch)
            return ProdSpec("prod", children), idx, wraps


# ============================================================
# --- PC Builder (TreeSpec → PCNet DAG) ---
# ============================================================

class PCBuilder:
    """Builds a DAG-compatible PCNet from grammar specs."""
    def __init__(self, pcnet):
        self.pc = pcnet
        self.input_cache = {}

    def build_classifier(self, trees, inputs):
        feats = sorted(set(f for t in trees for f in self._collect_features(t)))
        for f in feats:
            if f not in self.input_cache:
                node = InputNode(f, device=self.pc.device)
                node.fit(inputs)
                self.input_cache[f] = node
                self.pc.add_node(node)

        subcircuits = [self._build_subtree(t) for t in trees]
        root = ClassifierNode(subcircuits)
        self.pc.add_node(root)
        for c in subcircuits:
            self.pc.add_edge(root, c)
        self.pc.set_root(root)

        # collect params
        self.pc.params = []
        for n in self.pc.get_nodes():
            if isinstance(n, nn.Module):
                if hasattr(n, "get_learnable_params"):
                    self.pc.params += list(n.get_learnable_params())
                else:
                    # Fallback: include all module parameters
                    for _, p in n.named_parameters(recurse=False):
                        if p.requires_grad:
                            self.pc.params.append(p)
        return root

    def _collect_features(self, t):
        out = []
        def walk(x):
            if isinstance(x, LeafSpec): out.append(x.feature_idx)
            elif hasattr(x, "children"):
                for c in x.children:
                    walk(c)
        walk(t)
        return out

    def _build_subtree(self, t):
        if isinstance(t, LeafSpec):
            return self.input_cache[t.feature_idx]
        elif isinstance(t, SumSpec):
            kids = [self._build_subtree(c) for c in t.children]
            weights = torch.rand(len(kids), device=self.pc.device)
            node = SumNode(kids, weights=weights, device=self.pc.device)
            self.pc.add_node(node)
            for c in kids:
                self.pc.add_edge(node, c)
            return node
        elif isinstance(t, ProdSpec):
            kids = [self._build_subtree(c) for c in t.children]
            node = ProductNode(kids)
            self.pc.add_node(node)
            for c in kids:
                self.pc.add_edge(node, c)
            return node
        elif isinstance(t, GateSpec):
            kids = [self._build_subtree(c) for c in t.children]
            node = GateNode(kids[0], kids[1], device=self.pc.device)
            self.pc.add_node(node)
            for c in kids: self.pc.add_edge(node, c)
            return node

        elif isinstance(t, ResidualSpec):
            kids = [self._build_subtree(c) for c in t.children]
            node = ResidualNode(kids[0], kids[1], device=self.pc.device)
            self.pc.add_node(node)
            for c in kids: self.pc.add_edge(node, c)
            return node
        else:
            raise ValueError(f"Unknown spec type: {type(t)}")


# ============================================================
# --- GE Optimizer ---
# ============================================================

class GEOptimizer:
    """
    Parallel, DAG-compatible GE optimizer for the new PCNet structure.
    """
    def __init__(
        self,
        pop_size=16,
        genome_len=64,
        max_depth=4,
        max_branching=4,
        crossover_rate=0.9,
        mutation_rate=0.05,
        elite_k=2,
        n_classes=1,
        device="cpu",
        seed=42,
        n_jobs=None,
        report=None
    ):
        self.pop_size = pop_size
        self.genome_len = genome_len
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_k = elite_k
        self.n_classes = n_classes
        self.device = device
        self.mapper = GEMapper(max_wraps=2)
        self.rng = np.random.default_rng(seed)
        self.n_jobs = n_jobs or os.cpu_count()
        self.report = report

    # -----------------------------------------------------
    # GA primitives
    # -----------------------------------------------------
    def init_population(self):
        return [self.rng.integers(0, 256, size=self.genome_len, dtype=np.int32).tolist()
                for _ in range(self.pop_size)]

    def mutate(self, genome):
        return [random.randint(0, 255) if random.random() < self.mutation_rate else g for g in genome]

    def crossover(self, g1, g2):
        if random.random() > self.crossover_rate:
            return g1[:], g2[:]
        cut = random.randint(1, len(g1) - 2)
        return g1[:cut] + g2[cut:], g2[:cut] + g1[cut:]

    def get_dataloader(self, inputs, labels, batch_size=32):
        dataset = TensorDataset(inputs, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # -----------------------------------------------------
    # Core fitness evaluation
    # -----------------------------------------------------
    def _evaluate_single(self, genome, inputs, labels, criterion, n_classes, train_steps, lr, seed_offset=0, global_index=None, param_bank=None):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        sig = hash(tuple(genome))
        random.seed(seed_offset)
        np.random.seed(seed_offset)
        torch.manual_seed(seed_offset)

        if self.report:
            with open(self.report, "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] Genome {global_index} started.\n")

        # Map → DAG grammar
        trees = self.mapper.map_model(genome, inputs.shape[1], self.max_depth, self.max_branching)

        pc = PCNet(input_size=inputs.shape[1:], n_classes=n_classes, device=self.device,
                   max_depth=self.max_depth, max_branching=self.max_branching)
        builder = PCBuilder(pc)
        builder.build_classifier(trees, inputs)

        if param_bank is not None and sig in param_bank:
            try:
                pc.load_state_dict(param_bank[sig], strict=False)
            except Exception as e:
                print(f"⚠️ Reuse failed: {e}")
        # train/val split
        split = int(0.8 * len(inputs))
        
        x_tr, y_tr = inputs[:split].to(self.device), labels[:split].to(self.device)
        x_va, y_va = inputs[split:].to(self.device), labels[split:].to(self.device)
        opt = torch.optim.Adam(pc.params, lr=lr, weight_decay=1e-4)

        train_loader = self.get_dataloader(x_tr, y_tr)
        val_loader = self.get_dataloader(x_va, y_va)

        for _ in range(train_steps):
            for xb, yb in train_loader:
                logits = pc.evaluate(xb)
                loss = criterion(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # validation
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in val_loader:
                logits = pc.evaluate(xb)
                val_loss += criterion(logits, yb).item()
            val_loss /= max(1, len(val_loader))

        if param_bank is not None:
            param_bank[sig] = deepcopy(pc.state_dict())

        del pc, builder, opt, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return val_loss, global_index

    # -----------------------------------------------------
    # Population evaluation (parallel)
    # -----------------------------------------------------
    def evaluate_population(self, population, inputs, labels, criterion, n_classes=2, train_steps=20, lr=1e-2, param_bank=None):
        n_cpus = min(self.n_jobs, len(population))
        queues = [mp.Queue() for _ in range(len(population))]
        procs = []

        for i, genome in enumerate(population):
            p = mp.Process(target=self._worker_eval,
                           args=(i, genome, inputs, labels, queues[i], criterion, n_classes, train_steps, lr, param_bank))
            p.start()
            procs.append(p)
            if len(procs) >= n_cpus:
                for p in procs: p.join()
                procs.clear()

        for p in procs:
            p.join()

        results = []
        for q in queues:
            try:
                results.append(q.get(timeout=30))
            except:
                results.append((float("inf"), len(results)))

        losses = [None] * len(population)
        for loss, idx in results:
            losses[idx] = loss
        return losses

    def _worker_eval(self, idx, genome, inputs, labels, queue, criterion, n_classes, train_steps, lr, param_bank):
        try:
            loss, gid = self._evaluate_single(genome, inputs, labels, criterion, n_classes, train_steps, lr, idx, idx, param_bank)
            queue.put((loss, gid))
        except Exception as e:
            print(f"[Worker {idx}] Error: {e}")
            queue.put((float("inf"), idx))

    # -----------------------------------------------------
    # Evolutionary loop
    # -----------------------------------------------------
    def evolve(self, inputs, labels, criterion, generations=10, n_classes=1, train_steps=20, lr=1e-2):
        manager = Manager()
        param_bank = manager.dict()
        population = self.init_population()
        best_score, best_genome, history = math.inf, None, []
        complexity_hist = []
        if labels.dim() == 4 and labels.size(1) == 1:
            labels = labels.squeeze(1)

        for gen in range(generations):
            print(f"🧬 Generation {gen+1}/{generations}")
            losses = self.evaluate_population(population, inputs, labels, criterion, n_classes, train_steps, lr, param_bank)

            # --- Evaluate complexity ---
            complexities = []
            for genome in population:
                trees = self.mapper.map_model(genome, inputs.shape[1], self.max_depth, self.max_branching)
                dummy_pc = PCNet(input_size=inputs.shape[1:], n_classes=n_classes, device=self.device,
                                max_depth=self.max_depth, max_branching=self.max_branching)
                PCBuilder(dummy_pc).build_classifier(trees, inputs)
                complexities.append(sum(p.numel() for p in getattr(dummy_pc, "params", [])))

            # --- Pareto sorting ---
            scored = list(zip(losses, complexities, population))
            scored.sort(key=lambda x: (x[0], x[1]))  # loss first, then smaller model

            best_loss, best_complexity, best = scored[0]
            history.append(best_loss)
            complexity_hist.append(best_complexity)

            if best_loss < best_score:
                best_score, best_genome = best_loss, deepcopy(best)

            elites = [deepcopy(g) for _, _, g in scored[:self.elite_k]]
            new_pop = elites[:]

            def select():
                i, j = random.randrange(len(scored)), random.randrange(len(scored))
                return scored[i] if (scored[i][0] < scored[j][0] or
                                    (scored[i][0] == scored[j][0] and scored[i][1] < scored[j][1])) else scored[j]

            while len(new_pop) < self.pop_size:
                p1, p2 = select()[2], select()[2]
                c1, c2 = self.crossover(p1, p2)

                # --- Weight inheritance ---
                sig1, sig2 = hash(tuple(p1)), hash(tuple(p2))
                if sig1 in param_bank and sig2 in param_bank:
                    blended = {}
                    for k in param_bank[sig1]:
                        if k in param_bank[sig2]:
                            blended[k] = 0.5 * (param_bank[sig1][k] + param_bank[sig2][k])
                    param_bank[hash(tuple(c1))] = blended

                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self.mutate(c2))

            population = new_pop
            print(f"✅ Gen {gen+1}: best loss={best_score:.4f}, complexity={best_complexity}")

        # --- Final refinement of best genome ---
        trees = self.mapper.map_model(best_genome, inputs.shape[1], self.max_depth, self.max_branching)
        final_pc = PCNet(input_size=inputs.shape[1:], n_classes=n_classes,
                        device=self.device, max_depth=self.max_depth, max_branching=self.max_branching)
        PCBuilder(final_pc).build_classifier(trees, inputs)

        # short fine-tune of the best model
        opt = torch.optim.Adam(final_pc.params, lr=lr)
        for _ in range(10):
            logits = final_pc.evaluate(inputs)
            loss = criterion(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
        
        return final_pc, best_genome, {"loss": history, "complexity": complexity_hist}

