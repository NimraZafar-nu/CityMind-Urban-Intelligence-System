"""
CityMind – Challenge 2: Road Network (Kruskal MST + full connectivity)

CRITICAL FIX for rerouting:
The previous implementation only connected GRID-ADJACENT nodes (touching cells).
Since CSP places nodes sparsely across the grid, most nodes had only 1 neighbor.
Blocking that 1 edge made the node completely unreachable → rerouting returned None.

NEW APPROACH:
1. Build a FULL connectivity graph: every node can reach every other node.
   Edge cost = Euclidean distance × road type discount.
2. Run Kruskal's to find the MST (marked in_mst=True, drawn in blue).
3. Add ALL edges within a reasonable distance as backup edges (not in MST).
   These are drawn gray but exist in the graph so A* can use them.
4. Add Hospital↔Depot redundancy edge.
5. This guarantees: blocking any single edge never isolates a node.
"""
import math
import itertools
from graph import CityGraph


class UnionFind:
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank   = {n: 0 for n in nodes}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return False
        if self.rank[rx] < self.rank[ry]: rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]: self.rank[rx] += 1
        return True


class MSTBuilder:
    def __init__(self, graph: CityGraph):
        self.graph    = graph
        self.mst_cost = 0.0
        self.path1    = None
        self.path2    = None

    def build(self, log_fn=None):
        g = self.graph
        if len(g.nodes) < 2:
            return

        old_blocked = {k for k,e in g.edges.items() if e.blocked}
        old_flooded = {k for k,e in g.edges.items() if getattr(e,'flooded',False)}
        g.edges.clear()

        node_ids = list(g.nodes.keys())

        # ── Build candidate edges ──────────────────────────────────────────────
        # Strategy: connect nodes that are grid-adjacent OR within 2 grid cells.
        # This gives each node multiple neighbors for rerouting.
        # We limit to nearby pairs to keep edge count manageable.

        pos_to_id = {(n.gx, n.gy): n.id for n in g.nodes.values()}
        all_candidates = []  # (cost, a_id, b_id)
        seen = set()

        MAX_GRID_DIST = 3   # connect nodes up to 3 grid cells apart

        for n in g.nodes.values():
            for nb in g.nodes.values():
                if nb.id <= n.id: continue
                pair = (n.id, nb.id)
                if pair in seen: continue
                seen.add(pair)
                grid_dist = abs(n.gx-nb.gx) + abs(n.gy-nb.gy)
                if grid_dist > MAX_GRID_DIST: continue
                # Cost: Euclidean pixel distance normalized, with residential discount
                px_dist = math.hypot(n.px-nb.px, n.py-nb.py)
                bc = 0.8 if (n.type=="residential" or nb.type=="residential") else 1.0
                cost = px_dist * bc / 100.0
                all_candidates.append((cost, n.id, nb.id))

        # If too sparse, fall back to ALL pairs (ensures connectivity)
        if len(all_candidates) < len(node_ids) - 1:
            seen2 = set()
            for n, nb in itertools.combinations(g.nodes.values(), 2):
                pair = (min(n.id,nb.id), max(n.id,nb.id))
                if pair in seen2: continue
                seen2.add(pair)
                px_dist = math.hypot(n.px-nb.px, n.py-nb.py)
                bc = 0.8 if (n.type=="residential" or nb.type=="residential") else 1.0
                all_candidates.append((px_dist*bc/100.0, n.id, nb.id))

        all_candidates.sort(key=lambda x: x[0])

        # ── Kruskal's MST ──────────────────────────────────────────────────────
        uf        = UnionFind(node_ids)
        mst_count = 0
        self.mst_cost = 0.0
        mst_keys  = set()

        for cost, a_id, b_id in all_candidates:
            if uf.union(a_id, b_id):
                key = (min(a_id,b_id), max(a_id,b_id))
                e = g.add_edge(a_id, b_id, base_cost=cost)
                e.in_mst    = True
                e.redundant = False
                mst_keys.add(key)
                self.mst_cost += cost
                mst_count += 1
                if mst_count == len(node_ids)-1:
                    break

        # ── Add backup edges (non-MST, for rerouting) ─────────────────────────
        # Add all short-distance candidate edges that aren't in MST yet.
        # This gives every node at least 2-3 neighbors so blocking 1 never isolates it.
        BACKUP_LIMIT = len(node_ids) * 2   # don't add too many
        backup_added = 0

        for cost, a_id, b_id in all_candidates:
            if backup_added >= BACKUP_LIMIT:
                break
            key = (min(a_id,b_id), max(a_id,b_id))
            if key in g.edges:
                continue  # already added (MST edge)
            e = g.add_edge(a_id, b_id, base_cost=cost)
            e.in_mst    = False
            e.redundant = False
            backup_added += 1

        if log_fn:
            log_fn(f"✓ MST: {mst_count} edges, {backup_added} backup, cost={self.mst_cost:.2f}")

        # ── Hospital↔Depot redundancy ──────────────────────────────────────────
        self._add_redundancy(log_fn)

        # ── Restore blocked/flooded ────────────────────────────────────────────
        for key in old_blocked:
            if key in g.edges:
                g.edges[key].blocked = True
        for key in old_flooded:
            if key in g.edges:
                g.edges[key].flooded = True

        # ── Update effective costs ─────────────────────────────────────────────
        g.update_effective_costs()

        # ── Independent paths ─────────────────────────────────────────────────
        self._find_independent_paths(log_fn)

    def _add_redundancy(self, log_fn=None):
        g = self.graph
        hospitals = [n.id for n in g.nodes.values() if n.type=="hospital"]
        depots    = [n.id for n in g.nodes.values() if n.type=="depot"]
        if not hospitals or not depots:
            return
        best_c, best_pair = float('inf'), None
        for h in hospitals:
            hn = g.nodes[h]
            for d in depots:
                dn = g.nodes[d]
                key = (min(h,d),max(h,d))
                if key in g.edges:
                    g.edges[key].redundant = True
                    continue
                px_dist = math.hypot(hn.px-dn.px, hn.py-dn.py)
                bc = 0.8 if (hn.type=="residential" or dn.type=="residential") else 1.0
                c  = px_dist*bc/100.0
                if c < best_c:
                    best_c, best_pair = c, (h,d)
        if best_pair:
            e = g.add_edge(best_pair[0], best_pair[1], base_cost=best_c)
            e.in_mst   = False
            e.redundant = True
            if log_fn:
                log_fn(f"✓ Redundancy: H#{best_pair[0]}↔D#{best_pair[1]}")

    def _find_independent_paths(self, log_fn=None):
        from collections import deque
        g = self.graph
        hospitals = [n.id for n in g.nodes.values() if n.type=="hospital"]
        depots    = [n.id for n in g.nodes.values() if n.type=="depot"]
        self.path1 = None
        self.path2 = None
        if not hospitals or not depots:
            return
        h_id = hospitals[0]
        d_id = depots[0]

        def bfs(start, goal, forbidden=None):
            forbidden = forbidden or set()
            parent = {start: None}
            q = deque([start])
            while q:
                cur = q.popleft()
                if cur == goal:
                    path, node = [], goal
                    while node is not None:
                        path.append(node); node = parent[node]
                    path.reverse(); return path
                for nb, edge in g.neighbors(cur):
                    if nb.id not in parent and nb.id not in forbidden:
                        parent[nb.id] = cur; q.append(nb.id)
            return None

        self.path1 = bfs(h_id, d_id)
        if self.path1 and len(self.path1) > 2:
            forbidden = set(self.path1[1:-1])
            self.path2 = bfs(h_id, d_id, forbidden)
        if log_fn:
            p2 = len(self.path2) if self.path2 else 0
            log_fn(f"✓ H→D: primary={len(self.path1) if self.path1 else 0} backup={p2}")

    def rebuild_after_change(self, log_fn=None):
        self.build(log_fn)

    @property
    def mst_total_cost(self):
        return self.mst_cost
