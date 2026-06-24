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
                # Instead of returning None, fallback to pseudo-random codon
                return random.randint(0, 255), idx, wraps, True
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

        node_type = c % 3  # 0=leaf, 1=sum, 2=prod

        # Leaf
        if node_type == 0:
            f, idx, wraps, exhausted2 = self._next(codons, idx, wraps)
            if exhausted2:
                f = self._fallback_feature(idx, depth, n_features)
            else:
                f = f % max(1, n_features)
            return LeafSpec("input", f), idx, wraps

        # Sum (allow overlapping children)
        elif node_type == 1:
            kcod, idx, wraps, _ = self._next(codons, idx, wraps)
            k = 2 + (kcod % max(1, max_branching - 1))
            children = []
            for _ in range(k):
                ch, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
                children.append(ch)
            # overlap (duplicate a few children)
            if len(children) > 2 and random.random() < 0.4:
                dup = random.choice(children)
                children.insert(random.randint(0, len(children)), dup)
            return SumSpec("sum", len(children), children), idx, wraps

        # Product (now with overlapping children too)
        elif node_type == 2:
            n_child = 2 + (c % max(1, max_branching - 1))
            children = []
            for _ in range(n_child):
                ch, idx, wraps = self._expand(codons, idx, wraps, n_features, max_depth, max_branching, depth + 1)
                children.append(ch)
            # allow fan-out by duplicating references
            if len(children) > 2 and random.random() < 0.4:
                dup = random.choice(children)
                children.insert(random.randint(0, len(children)), dup)
            return ProdSpec("prod", children), idx, wraps


# ============================================================
# --- PC Builder (TreeSpec → PCNet DAG) ---
# ============================================================

class PCBuilder:
    """Builds a DAG-compatible PCNet from grammar specs."""
    def __init__(self, pcnet):
        self.pc = pcnet
        self.input_cache = {}
        self._memo = {}  # NEW: spec-id -> built PC node

    def _collect_features(self, t):
        """Recursively collect feature indices from a TreeSpec."""
        out = []

        def walk(x):
            if isinstance(x, LeafSpec):
                out.append(x.feature_idx)
            elif hasattr(x, "children"):
                for c in x.children:
                    walk(c)

        walk(t)
        return out

    def build_classifier(self, trees, inputs):
        feats = sorted(set(f for t in trees for f in self._collect_features(t)))
        
        # Instantiate InputNodes 
        # (the fit() method might need spatial format but we'll adapt depending on inputs dimension)
        self.pc.leaves = nn.ModuleList()
        current_inps = []
        for f in feats:
            if f not in self.input_cache:
                node = InputNode(f)
                node.fit(inputs)
                self.input_cache[f] = node
                current_inps.append(node)
                
        # Fill self.pc.leaves to track parameters
        self.pc.leaves.extend(current_inps)

        # Build once per unique subtree thanks to memoization
        subcircuits = [self._build_subtree(t) for t in trees]

        root = ClassifierNode(subcircuits)
        
        # In current PCNet, we just set root and track params inside
        self.pc.root = root

        # Re-initialize tracked params manually if there relies on it
        self.pc.params = []
        for n in self.pc.modules():
            if isinstance(n, (InputNode, SumNode, ProductNode, ClassifierNode, AttentionalSumNode, MultiHeadAttentionalSumNode)):
                for p in n.parameters():
                    if p.requires_grad:
                        if not any(p is existing_p for existing_p in self.pc.params):
                            self.pc.params.append(p)
        return root

    def _build_subtree(self, t):
        # --- memoization: reuse node instances for shared specs ---
        key = id(t)
        if key in self._memo:
            return self._memo[key]

        if isinstance(t, LeafSpec):
            node = self.input_cache[t.feature_idx]
            # Leaves are already shared via input_cache; still return early
            self._memo[key] = node
            return node

        elif isinstance(t, SumSpec):
            kids = [self._build_subtree(c) for c in t.children]
            
            # Use AttentionalSumNode if device implies Attentional PCNet, otherwise basic SumNode
            if hasattr(self.pc, "num_heads") and hasattr(self.pc, "embed_dim"):
                node = MultiHeadAttentionalSumNode(kids, input_dim=self.pc.in_channels, embed_dim=self.pc.embed_dim, num_heads=self.pc.num_heads)
            elif hasattr(self.pc, "embed_dim"):
                node = AttentionalSumNode(kids, input_dim=self.pc.in_channels, embed_dim=self.pc.embed_dim)
            else:
                node = SumNode(kids)
                
            self._memo[key] = node
            return node

        elif isinstance(t, ProdSpec):
            kids = [self._build_subtree(c) for c in t.children]
            node = ProductNode(kids)
            self._memo[key] = node
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
        n_classes=2,
        device="cpu",
        seed=42,
        n_jobs=None,
        report=None,
        log_path=None,
        model_type="pcnet",
        embed_dim=16,
        num_heads=4
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
        self.log_path = log_path
        self.model_type = model_type
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        if log_path:
            os.makedirs(log_path, exist_ok=True)
            
        self.evolution_report = os.path.join(log_path, "evolution_report.txt") if log_path else None
        if self.evolution_report:
            with open(self.evolution_report, "w") as f:
                f.write("Generation,Avg_Fitness,Best_Fitness,Avg_Complexity,Best_Complexity,All_Time_Best_Fitness\n")
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

        # Map → DAG grammar
        trees = self.mapper.map_model(genome, inputs.shape[1], self.max_depth, self.max_branching)
        
        # Instantiate base PCNet to hold the structure
        if self.model_type == "multihead_attentional_pcnet":
            pc = MultiHeadAttentionalPCNet(
                in_channels=inputs.shape[1], n_classes=n_classes, 
                max_depth=self.max_depth, max_branching=self.max_branching, 
                embed_dim=self.embed_dim, num_heads=self.num_heads
            )
        elif self.model_type == "attentional_pcnet":
            pc = AttentionalPCNet(
                in_channels=inputs.shape[1], n_classes=n_classes, 
                max_depth=self.max_depth, max_branching=self.max_branching, embed_dim=self.embed_dim
            )
        else:
            pc = PCNet(
                n_classes=n_classes, max_depth=self.max_depth, max_branching=self.max_branching
            )

        # Build structurally onto the pc using our Graph tools
        builder = PCBuilder(pc)
        builder.build_classifier(trees, inputs)
        pc = pc.to(self.device)

        if param_bank is not None and sig in param_bank:
            try:
                pc.load_state_dict(param_bank[sig], strict=False)
            except Exception as e:
                print(f"⚠️ Reuse failed: {e}")
                
        # Train/Val Split (80/20)
        split = int(0.8 * len(inputs))
        x_tr, y_tr = inputs[:split].to(self.device), labels[:split].to(self.device)
        x_va, y_va = inputs[split:].to(self.device), labels[split:].to(self.device)
        
        # In current PCNet design, builder populates pc.params
        opt = torch.optim.Adam(pc.params, lr=lr, weight_decay=1e-4)

        train_loader = self.get_dataloader(x_tr, y_tr)
        val_loader = self.get_dataloader(x_va, y_va)

        # Basic NLL Optimization Training Loop
        pc.train()
        for _ in range(train_steps):
            for xb, yb in train_loader:
                if yb.ndim == 3:
                    yb = yb.unsqueeze(1)
                log_probs = pc(xb)
                loss = criterion(log_probs, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Validation Evaluation
        pc.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in val_loader:
                if yb.ndim == 3:
                    yb = yb.unsqueeze(1)
                log_probs = pc(xb)
                val_loss += criterion(log_probs, yb).item()
            val_loss /= max(1, len(val_loader))

        if param_bank is not None:
            param_bank[sig] = deepcopy(pc.state_dict())

        del pc, train_loader, val_loader
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
                dummy_pc = PCNet(n_classes=n_classes, max_depth=self.max_depth, max_branching=self.max_branching)
                PCBuilder(dummy_pc).build_classifier(trees, inputs)
                complexities.append(sum(p.numel() for p in getattr(dummy_pc, "params", [])))

            # --- Pareto sorting ---
            scored = list(zip(losses, complexities, population))
            scored.sort(key=lambda x: (x[0], x[1]))  # loss first, then smaller model
            losses = [s[0] for s in scored]
            complexities = [s[1] for s in scored]
            avg_loss = sum(losses) / len(losses)
            avg_complexity = sum(complexities) / len(complexities)

            best_loss, best_complexity, best = scored[0]
            history.append(best_loss)
            complexity_hist.append(best_complexity)

            if best_score is None or best_loss <= best_score:
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
            with open(self.evolution_report, "a") as f:
                f.write(f"[{str(time.strftime('%H:%M:%S'))}]{gen+1},{avg_loss:.4f},{best_loss:.4f},{avg_complexity},{best_complexity},{best_score:.4f}\n")
            f.close()
        # --- Final refinement of best genome ---
        trees = self.mapper.map_model(best_genome, inputs.shape[1], self.max_depth, self.max_branching)
        
        if self.model_type == "multihead_attentional_pcnet":
            final_pc = MultiHeadAttentionalPCNet(
                in_channels=inputs.shape[1], n_classes=n_classes, 
                max_depth=self.max_depth, max_branching=self.max_branching, 
                embed_dim=self.embed_dim, num_heads=self.num_heads
            ).to(self.device)
        elif self.model_type == "attentional_pcnet":
            final_pc = AttentionalPCNet(
                in_channels=inputs.shape[1], n_classes=n_classes, 
                max_depth=self.max_depth, max_branching=self.max_branching, embed_dim=self.embed_dim
            ).to(self.device)
        else:
            final_pc = PCNet(
                n_classes=n_classes, max_depth=self.max_depth, max_branching=self.max_branching
            ).to(self.device)
            
        PCBuilder(final_pc).build_classifier(trees, inputs)
        
        return final_pc, best_genome, {"loss": history, "complexity": complexity_hist}