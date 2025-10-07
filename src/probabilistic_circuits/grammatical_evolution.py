"""
Grammar define as follows:

<model>      ::= Classifier(<tree>, <tree>)           ; binary segmentation
<tree>       ::= <sum> | <prod> | <leaf>
<sum>        ::= SUM[k]( <tree>,...,<tree> )          ; 2 ≤ k ≤ max_branching
<prod>       ::= PROD( <tree>, <tree> )               ; binary product (pairwise)
<leaf>       ::= INPUT[f]                              ; choose feature/channel f
; Depth-limiting is enforced by the mapper (at max_depth → <leaf> only).

"""

# ============================================================
# GE mapper: genome -> tree spec
# ============================================================
from dataclasses import dataclass
from typing import List, Optional, Union
import torch
from probabilistic_circuits.probabilistic_circuits import *
import random
import numpy as np
import tqdm
import math
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LeafSpec:
    kind: str           # 'input'
    feature_idx: int

@dataclass
class SumSpec:
    kind: str           # 'sum'
    k: int
    children: List["TreeSpec"]

@dataclass
class ProdSpec:
    kind: str           # 'prod'
    left: "TreeSpec"
    right: "TreeSpec"

TreeSpec = Union[LeafSpec, SumSpec, ProdSpec]

class GEMappingError(Exception):
    pass

class GEMapper:
    """
    Maps a list of integers (codons) to a TreeSpec following the grammar above.
    Uses wrap-around with a maximum number of wraps to avoid infinite expansions.
    """
    def __init__(self, max_wraps=2):
        self.max_wraps = max_wraps

    def map_model(self, codons: List[int], n_features: int, max_depth: int, max_branching: int) -> List[TreeSpec]:
        # Binary classifier: produce two independent trees (one per class)
        specs = []
        idx, wraps = 0, 0
        for _ in range(2):
            spec, idx, wraps = self._expand_tree(codons, idx, wraps, n_features, max_depth, max_branching, depth=0)
            specs.append(spec)
        return specs

    def _next_codon(self, codons, idx, wraps):
        if idx >= len(codons):
            if wraps >= self.max_wraps:
                raise GEMappingError("Exceeded maximum codon wrap-around.")
            idx, wraps = 0, wraps + 1
        val = codons[idx]
        return val, idx + 1, wraps

    def _expand_tree(self, codons, idx, wraps, n_features, max_depth, max_branching, depth):
        # If max depth reached → force leaf
        if depth >= max_depth:
            feature, idx, wraps = self._pick_feature(codons, idx, wraps, n_features)
            return LeafSpec(kind="input", feature_idx=feature), idx, wraps

        # Choose node type: 0=input, 1=sum, 2=prod
        choice, idx, wraps = self._next_codon(codons, idx, wraps)
        node_type = choice % 3

        if node_type == 0:  # leaf
            feature, idx, wraps = self._pick_feature(codons, idx, wraps, n_features)
            return LeafSpec(kind="input", feature_idx=feature), idx, wraps

        elif node_type == 1:  # sum
            kcod, idx, wraps = self._next_codon(codons, idx, wraps)
            k = 2 + (kcod % max(1, (max_branching - 1)))      # in [2, max_branching]
            children = []
            for _ in range(k):
                ch, idx, wraps = self._expand_tree(codons, idx, wraps, n_features, max_depth, max_branching, depth+1)
                children.append(ch)
            return SumSpec(kind="sum", k=k, children=children), idx, wraps

        else:  # prod (binary)
            left, idx, wraps = self._expand_tree(codons, idx, wraps, n_features, max_depth, max_branching, depth+1)
            right, idx, wraps = self._expand_tree(codons, idx, wraps, n_features, max_depth, max_branching, depth+1)
            return ProdSpec(kind="prod", left=left, right=right), idx, wraps

    def _pick_feature(self, codons, idx, wraps, n_features):
        c, idx, wraps = self._next_codon(codons, idx, wraps)
        return c % n_features, idx, wraps

# ============================================================
# ============================================================
# Build a PCNet from a TreeSpec
# ============================================================
class PCBuilder:
    """
    Creates nodes and edges inside a given PCNet instance from a TreeSpec.
    Reuses your InputNode/SumNode/ProductNode/ClassifierNode.
    """
    def __init__(self, pcnet: PCNet):
        self.pc = pcnet
        self.input_cache = {}  # feature_idx -> InputNode

    def build_classifier(self, trees: List[TreeSpec], inputs: torch.Tensor):
        # 1) ensure InputNodes exist & are fit() once
        self._ensure_inputs(trees, inputs)

        # 2) build subcircuits
        child_nodes = [self._build_subtree(t) for t in trees]

        # 3) root classifier
        root = ClassifierNode(child_nodes)
        self.pc.add_node(root)
        for ch in child_nodes:
            self.pc.add_edge(root, ch)
        self.pc.set_root(root)

        # 4) collect parameters (sum weights + input distribution params)
        params = []
        for n in self.pc.get_nodes():
            if isinstance(n, SumNode):
                params.append(n.weights)
            elif isinstance(n, InputNode):
                params.extend([n.mu, n.log_sigma, n.log_nu, n.logits])
        self.pc.params = params
        return root

    def _ensure_inputs(self, trees: List[TreeSpec], inputs: torch.Tensor):
        # Flatten all feature indices from all trees
        feats = sorted(set(
            f
            for t in trees if t is not None
            for f in self._collect_features(t)
        ))

        for f in feats:
            if f not in self.input_cache:
                node = InputNode(f, device=self.pc.device)
                node.fit(inputs)  # robust init
                self.input_cache[f] = node
                self.pc.add_node(node)


    def _collect_features(self, t: TreeSpec) -> List[int]:
        feats = []
        def walk(x):
            if isinstance(x, LeafSpec):
                feats.append(x.feature_idx)
            elif isinstance(x, SumSpec):
                for c in x.children: walk(c)
            elif isinstance(x, ProdSpec):
                walk(x.left); walk(x.right)
        walk(t)
        return feats

    def _build_subtree(self, t: TreeSpec):
        if isinstance(t, LeafSpec):
            return self.input_cache[t.feature_idx]

        if isinstance(t, SumSpec):
            # create children first
            kids = [self._build_subtree(c) for c in t.children]
            # random normalized weights
            w = torch.rand(len(kids), device=self.pc.device)
            w = w / (w.sum() + 1e-8)
            node = SumNode(kids, weights=w, device=self.pc.device)
            self.pc.add_node(node)
            for ch in kids:
                self.pc.add_edge(node, ch)
            return node

        if isinstance(t, ProdSpec):
            left = self._build_subtree(t.left)
            right = self._build_subtree(t.right)
            node = ProductNode([left, right])
            self.pc.add_node(node)
            self.pc.add_edge(node, left)
            self.pc.add_edge(node, right)
            return node

        raise ValueError("Unknown TreeSpec.")

# ============================================================
# GE optimizer
# ============================================================
import math
from copy import deepcopy

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
        seed=42
    ):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.pop_size = pop_size
        self.genome_len = genome_len
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_k = elite_k
        self.device = device

        self.mapper = GEMapper(max_wraps=2)

    def init_population(self):
        return [np.random.randint(0, 256, size=self.genome_len, dtype=np.int32).tolist()
                for _ in range(self.pop_size)]

    # -------- genetic operators --------
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
        c1 = g1[:cut] + g2[cut:]
        c2 = g2[:cut] + g1[cut:]
        return c1, c2
    # -------- fitness evaluation --------
    def evaluate_genome(self, genome, inputs, labels, n_classes=2, train_steps=50, lr=1e-2):
        """
        inputs: (B, C, H, W) float tensor
        labels: (B, H, W) long tensor
        """
        B, C, H, W = inputs.shape
        n_features = C

        # 1) Map genome to two trees
        try:
            trees = self.mapper.map_model(genome, n_features, self.max_depth, self.max_branching)
        except GEMappingError:
            return math.inf, None  # invalid (too many wraps)

        # 2) Build PC
        pc = PCNet(input_size=(C, H, W), n_classes=n_classes, device=self.device,
                   max_depth=self.max_depth, max_branching=self.max_branching)
        builder = PCBuilder(pc)
        builder.build_classifier(trees, inputs)

        # 3) quick train (likelihood-based) with DiceCE
        optim = torch.optim.Adam(pc.params, lr=lr)
        flat = labels.view(-1)
        counts = torch.bincount(flat, minlength=n_classes).float()
        counts[counts == 0] = 1.0
        balance = (1.0 / counts)
        balance = balance / balance.sum()

        # --- Loss ---
        criterion = DiceCELoss(pos_weight=balance.detach().clone().to(self.device)).to(self.device)
        x, y = inputs.to(self.device), labels.to(self.device)

        for _ in range(train_steps):
            optim.zero_grad()
            logits = pc.evaluate(x)  # (B, n_classes, H, W)
            mask = y.squeeze(1)
            loss = criterion(logits, mask)
            loss.backward()
            optim.step()

        # 4) validation score (lower is better here)
        with torch.no_grad():
            logits = pc.evaluate(x)
            mask = y.squeeze(1)
            val = criterion(logits, mask).item()

        return val, pc

    def evolve(self, train_inputs, train_labels, generations=10):
        population = self.init_population()
        history = []

        best_score = math.inf
        best_pc = None
        best_genome = None

        for gen in range(generations):
            scored = []
            for genome in population:
                score, pc = self.evaluate_genome(genome, train_inputs, train_labels)
                scored.append((score, genome, pc))

            scored.sort(key=lambda t: t[0])
            history.append(scored[0][0])

            # track best
            if scored[0][0] < best_score:
                best_score, best_genome, best_pc = scored[0]

            # Elitism
            new_pop = [deepcopy(scored[i][1]) for i in range(self.elite_k)]

            # Tournament selection for the rest
            def pick():
                i, j = random.randrange(self.pop_size), random.randrange(self.pop_size)
                return scored[i] if scored[i][0] < scored[j][0] else scored[j]

            while len(new_pop) < self.pop_size:
                p1 = pick()[1]
                p2 = pick()[1]
                c1, c2 = self.crossover(p1, p2)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self.mutate(c2))

            population = new_pop

            print(f"[GE] Gen {gen+1}/{generations}  best_loss={scored[0][0]:.4f}")

        return best_pc, best_genome, history
