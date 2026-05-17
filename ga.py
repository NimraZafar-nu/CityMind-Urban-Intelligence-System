"""
CityMind – Challenge 3: Ambulance Placement via Genetic Algorithm
From scratch – no external libraries.

Chromosome  : tuple of 3 unique node IDs  (one per ambulance)
Fitness     : -(max Dijkstra distance from any citizen to nearest ambulance)
              Lower max-distance = better coverage = higher fitness
Selection   : Tournament (size 3)
Crossover   : Ordered crossover with duplicate repair
Mutation    : 10 % chance to replace one gene with a random unoccupied node
Elitism     : Top 2 chromosomes pass unchanged to next generation
"""
import random
import heapq
from graph import CityGraph


# ── Dijkstra (runs inside fitness) ────────────────────────────────────────────

def _dijkstra(graph: CityGraph, start_id: int) -> dict:
    """
    Shortest path from start_id to every reachable node.
    Uses effective_cost so risk multipliers affect distances.
    Returns {node_id: cost}.
    """
    dist = {start_id: 0.0}
    pq   = [(0.0, start_id)]
    while pq:
        cost, cur = heapq.heappop(pq)
        if cost > dist[cur]:
            continue
        for nb, edge in graph.neighbors(cur):          # skips blocked edges
            nc = cost + edge.effective_cost
            if nc < dist.get(nb.id, 1e18):
                dist[nb.id] = nc
                heapq.heappush(pq, (nc, nb.id))
    return dist


# ── Genetic Algorithm ─────────────────────────────────────────────────────────

class GeneticAmbulancePlacer:
    NUM_AMBULANCES = 3
    POP_SIZE       = 50
    GENERATIONS    = 100
    MUTATION_RATE  = 0.10
    ELITE_K        = 2
    TOURNAMENT_K   = 3

    def __init__(self, graph: CityGraph):
        self.graph           = graph
        self.placements      = []           # list of 3 node IDs (current best)
        self.worst_dist      = float("inf") # fitness value (lower = better)
        self.coverage_radius = {}           # nid -> (nearest_amb_id, dist)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, log_fn=None) -> list:
        """
        Full GA run. Safe to call repeatedly.
        Returns list of 3 node IDs.

        PERFORMANCE FIX: Reduce generations for large graphs to prevent freeze.
        With 50+ nodes, 100 generations × POP_SIZE=50 × Dijkstra each = very slow.
        We now cap generations at 30 for large graphs.
        """
        g       = self.graph
        all_ids = list(g.nodes.keys())

        if len(all_ids) < self.NUM_AMBULANCES:
            if log_fn:
                log_fn("GA: not enough nodes")
            return []

        depot_ids = [n.id for n in g.nodes.values() if n.type == "depot"]
        citizens  = [n.id for n in g.nodes.values() if n.type == "residential"]
        if not citizens:
            citizens = all_ids[:]

        # Adaptive generation count — fewer iterations for bigger graphs
        # This prevents the "Python not responding" freeze
        n_nodes = len(all_ids)
        if n_nodes > 60:
            gens = 20
        elif n_nodes > 30:
            gens = 40
        else:
            gens = self.GENERATIONS   # 100 for small graphs

        # Pre-compute Dijkstra from every node once
        dist_from = {nid: _dijkstra(g, nid) for nid in all_ids}

        def fitness(chrom):
            worst = 0.0
            for cid in citizens:
                nearest = min(dist_from[amb].get(cid, 1e18) for amb in chrom)
                if nearest > worst:
                    worst = nearest
            return worst

        population = [
            random.sample(all_ids, self.NUM_AMBULANCES)
            for _ in range(self.POP_SIZE)
        ]

        best_chrom = None
        best_fit   = 1e18

        for _gen in range(gens):   # ← use adaptive gens instead of GENERATIONS
            # Score every chromosome
            scored = sorted(
                [(fitness(c), c) for c in population],
                key=lambda x: x[0]
            )

            if scored[0][0] < best_fit:
                best_fit   = scored[0][0]
                best_chrom = scored[0][1][:]

            # Elitism: top ELITE_K survive unchanged
            new_pop = [c[:] for _, c in scored[:self.ELITE_K]]

            # Fill rest via tournament → crossover → mutation
            while len(new_pop) < self.POP_SIZE:
                # Tournament selection
                p1 = self._tournament(scored)
                p2 = self._tournament(scored)

                # Ordered crossover
                child = self._crossover(p1, p2, all_ids)

                # Mutation
                if random.random() < self.MUTATION_RATE:
                    idx = random.randint(0, self.NUM_AMBULANCES - 1)
                    others = [n for n in all_ids if n not in child]
                    if others:
                        child[idx] = random.choice(others)

                new_pop.append(child)

            population = new_pop

        # ── Finalise ─────────────────────────────────────────────────────────
        raw_best = best_chrom or random.sample(all_ids, self.NUM_AMBULANCES)

        # FIX C3: Snap each ambulance placement to nearest depot when possible
        # This ensures ambulances visually appear AT depot buildings
        final_placements = []
        used_depots = set()
        for amb_id in raw_best:
            # Find a depot not yet used that is reachable
            best_depot = None
            best_depot_dist = float('inf')
            for dep_id in depot_ids:
                if dep_id in used_depots:
                    continue
                d = dist_from[amb_id].get(dep_id, float('inf'))
                if d < best_depot_dist:
                    best_depot_dist = d
                    best_depot = dep_id
            if best_depot is not None and best_depot_dist < float('inf'):
                final_placements.append(best_depot)
                used_depots.add(best_depot)
            else:
                # No free depot reachable — keep original GA choice
                final_placements.append(amb_id)

        self.placements = final_placements
        self.worst_dist = best_fit
        self._build_coverage(dist_from, citizens)

        if log_fn:
            depot_count = sum(1 for nid in self.placements
                              if g.nodes.get(nid) and g.nodes[nid].type == "depot")
            log_fn(
                f"GA: ambulances at {self.placements} "
                f"({depot_count} at depots) | worst dist={best_fit:.2f}"
            )
        return self.placements

    def rerun(self, log_fn=None) -> list:
        """Alias – called every time risk weights change."""
        return self.run(log_fn)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _tournament(self, scored):
        """Return the chromosome with best fitness among TOURNAMENT_K random picks."""
        pool = random.sample(scored, min(self.TOURNAMENT_K, len(scored)))
        return min(pool, key=lambda x: x[0])[1][:]

    def _crossover(self, p1, p2, all_ids):
        """
        Ordered crossover:
        Take a slice from p1, fill remaining slots with p2 values
        not already in the slice (preserving order).
        Repair duplicates by adding random unused nodes.
        """
        n   = self.NUM_AMBULANCES
        cut = random.randint(1, n - 1)
        child = p1[:cut]
        for gene in p2:
            if gene not in child:
                child.append(gene)
            if len(child) == n:
                break
        # If still short (rare), fill with random unused
        used = set(child)
        for nid in all_ids:
            if len(child) == n:
                break
            if nid not in used:
                child.append(nid)
                used.add(nid)
        return child[:n]

    def _build_coverage(self, dist_from, citizens):
        """Record which ambulance each citizen is closest to."""
        self.coverage_radius = {}
        for cid in citizens:
            best_amb, best_d = None, 1e18
            for amb in self.placements:
                d = dist_from[amb].get(cid, 1e18)
                if d < best_d:
                    best_d   = d
                    best_amb = amb
            self.coverage_radius[cid] = (best_amb, best_d)
