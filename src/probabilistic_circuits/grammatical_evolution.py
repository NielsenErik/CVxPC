# ============================================================
# Parallel HPC-optimized GE components
# ============================================================

from dataclasses import dataclass
from typing import List, Union
import random, math, numpy as np, torch, tqdm
from joblib import Parallel, delayed
from copy import deepcopy
import os
from probabilistic_circuits.probabilistic_circuits import *

# ============================================================
# Tree grammar specs
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
    left: "TreeSpec"
    right: "TreeSpec"

TreeSpec = Union[LeafSpec, SumSpec, ProdSpec]


# ============================================================
# Mapper
# ============================================================

class GEMappingError(Exception): pass

class GEMapper:
    """Grammar-based mapper from codon sequence to TreeSpec."""
    def __init__(self, max_wraps=2):
        self.max_wraps = max_wraps

    def map_model(self, codons: List[int], n_features: int, max_depth: int, max_branching: int):
        specs, idx, wraps = [], 0, 0
        for _ in range(2):  # binary classifier = 2 trees
            tree, idx, wraps = self._expand_tree(codons, idx, wraps, n_features, max_depth, max_branching, 0)
            specs.append(tree)
        return specs

    def _next_codon(self, codons, idx, wraps):
        if idx >= len(codons):
            wraps += 1
            if wraps > self.max_wraps:
                raise GEMappingError("Exceeded max codon wraps.")
            idx = 0
        return codons[idx], idx + 1, wraps

    def _expand_tree(self, codons, idx, wraps, n_features, max_depth, max_branching, depth):
        if depth >= max_depth:
            feature, idx, wraps = self._pick_feature(codons, idx, wraps, n_features)
            return LeafSpec("input", feature), idx, wraps

        c, idx, wraps = self._next_codon(codons, idx, wraps)
        node_type = c % 3

        if node_type == 0:  # leaf
            f, idx, wraps = self._pick_feature(codons, idx, wraps, n_features)
            return LeafSpec("input", f), idx, wraps

        elif node_type == 1:  # sum
            kcod, idx, wraps = self._next_codon(codons, idx, wraps)
            k = 2 + (kcod % max(1, max_branching - 1))
            children = []
            for _ in range(k):
                ch, idx, wraps = self._expand_tree(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
                children.append(ch)
            return SumSpec("sum", k, children), idx, wraps

        else:  # prod
            left, idx, wraps = self._expand_tree(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
            right, idx, wraps = self._expand_tree(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
            return ProdSpec("prod", left, right), idx, wraps

    def _pick_feature(self, codons, idx, wraps, n_features):
        c, idx, wraps = self._next_codon(codons, idx, wraps)
        return c % n_features, idx, wraps


# ============================================================
# PC Builder
# ============================================================

class PCBuilder:
    """Parallel-safe circuit builder from TreeSpec."""
    def __init__(self, pcnet):
        self.pc = pcnet
        self.input_cache = {}

    def build_classifier(self, trees, inputs):
        feats = sorted(set(f for t in trees for f in self._collect_features(t)))
        # Shared initialization of input nodes
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
        params = []
        for n in self.pc.get_nodes():
            if isinstance(n, SumNode):
                params.append(n.weights)
            elif isinstance(n, InputNode):
                params.extend([n.mu, n.log_sigma, n.log_nu, n.logits, n.gate])
        self.pc.params = params
        return root

    def _collect_features(self, t):
        out = []
        def walk(x):
            if isinstance(x, LeafSpec): out.append(x.feature_idx)
            elif isinstance(x, SumSpec): [walk(c) for c in x.children]
            elif isinstance(x, ProdSpec): walk(x.left); walk(x.right)
        walk(t)
        return out

    def _build_subtree(self, t):
        if isinstance(t, LeafSpec):
            return self.input_cache[t.feature_idx]
        elif isinstance(t, SumSpec):
            kids = [self._build_subtree(c) for c in t.children]
            w = torch.rand(len(kids), device=self.pc.device)
            node = SumNode(kids, weights=w / (w.sum() + 1e-8), device=self.pc.device)
            self.pc.add_node(node)
            [self.pc.add_edge(node, c) for c in kids]
            return node
        elif isinstance(t, ProdSpec):
            left, right = self._build_subtree(t.left), self._build_subtree(t.right)
            node = ProductNode([left, right])
            self.pc.add_node(node)
            self.pc.add_edge(node, left)
            self.pc.add_edge(node, right)
            return node
        raise ValueError("Unknown TreeSpec.")


# ============================================================
# GE Optimizer (parallelized)
# ============================================================

class GEOptimizer:
    def __init__(
        self,
        pop_size=16,
        genome_len=64,
        max_depth=4,
        max_branching=4,
        crossover_rate=0.9,
        mutation_rate=0.05,
        elite_k=2,
        device="cpu",
        seed=42,
        n_jobs=None
    ):
        self.pop_size = pop_size
        self.genome_len = genome_len
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_k = elite_k
        self.device = device
        self.mapper = GEMapper(max_wraps=2)
        self.rng = np.random.default_rng(seed)
        self.n_jobs = n_jobs or os.cpu_count()

    # ----------------------------------------------
    # Genetic operators
    # ----------------------------------------------
    def init_population(self):
        return [self.rng.integers(0, 256, size=self.genome_len, dtype=np.int32).tolist()
                for _ in range(self.pop_size)]

    def mutate(self, genome):
        g = genome[:]
        for i in range(len(g)):
            if random.random() < self.mutation_rate:
                g[i] = random.randint(0, 255)
        return g

    def crossover(self, g1, g2):
        if random.random() > self.crossover_rate:
            return g1[:], g2[:]
        cut = random.randint(1, len(g1) - 2)
        return g1[:cut] + g2[cut:], g2[:cut] + g1[cut:]

    # ----------------------------------------------
    # Fitness evaluation
    # ----------------------------------------------
    def _evaluate_single(self, genome, inputs, labels, n_classes, train_steps, lr, seed_offset=0):
        # Independent RNG for each worker
        torch.manual_seed(seed_offset)
        np.random.seed(seed_offset)
        random.seed(seed_offset)

        try:
            trees = self.mapper.map_model(genome, inputs.shape[1], self.max_depth, self.max_branching)
        except GEMappingError:
            return math.inf, None

        pc = PCNet(input_size=inputs.shape[1:], n_classes=n_classes, device=self.device,
                   max_depth=self.max_depth, max_branching=self.max_branching)
        builder = PCBuilder(pc)
        builder.build_classifier(trees, inputs)
        if labels.dim() == 4 and labels.size(1) == 1:
            labels = labels.squeeze(1)
        optimizer = torch.optim.Adam(pc.params, lr=lr)
        flat = labels.view(-1)
        counts = torch.bincount(flat, minlength=n_classes).float()
        counts[counts == 0] = 1.0
        balance = (1.0 / counts)
        balance = balance / balance.sum()
        criterion = DiceCELoss(pos_weight=balance.to(self.device)).to(self.device)
        split = int(len(inputs) * 0.8)
        # shuffle inputs and label pairs

        rand_perm = torch.randperm(len(inputs))
        inputs, labels = inputs[rand_perm], labels[rand_perm]

        x, y = inputs[:split], labels[:split]
        x, y = x.to(self.device), y.to(self.device)
        x_val, y_val = inputs[split:], labels[split:]
        x_val, y_val = x_val.to(self.device), y_val.to(self.device)
        
        for _ in range(train_steps):
            optimizer.zero_grad()
            logits = pc.evaluate(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits = pc.evaluate(x_val)
            val_loss = criterion(logits, y_val).item()
        return val_loss, pc

    def evaluate_population(self, population, inputs, labels, n_classes=2, train_steps=50, lr=1e-2):
        """Parallel evaluation of population genomes."""
        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(self._evaluate_single)(genome, inputs, labels, n_classes, train_steps, lr, i)
            for i, genome in enumerate(population)
        )
        return results


    # ----------------------------------------------
    # Evolutionary loop
    # ----------------------------------------------
    def evolve(self, train_inputs, train_labels, generations=10):
        population = self.init_population()
        history, best_score, best_pc, best_genome = [], math.inf, None, None

        for gen in range(generations):
            print(f"🧬 Generation {gen+1}/{generations} (evaluating {len(population)} genomes in parallel)")
            results = self.evaluate_population(population, train_inputs, train_labels)
            scored = [(s, g, p) for (s, p), g in zip(results, population)]

            scored.sort(key=lambda x: x[0])
            history.append(scored[0][0])

            if scored[0][0] < best_score:
                best_score, best_genome, best_pc = scored[0]

            elites = [deepcopy(scored[i][1]) for i in range(min(self.elite_k, len(scored)))]
            new_pop = elites[:]

            # tournament selection
            def pick():
                i, j = random.randrange(len(scored)), random.randrange(len(scored))
                return scored[i] if scored[i][0] < scored[j][0] else scored[j]

            while len(new_pop) < self.pop_size:
                p1 = pick()[1]
                p2 = pick()[1]
                c1, c2 = self.crossover(p1, p2)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self.mutate(c2))

            population = new_pop
            print(f"✅ Gen {gen+1}: best loss={best_score:.4f}")

        return best_pc, best_genome, history
