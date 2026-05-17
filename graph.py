"""
CityMind – Shared City Graph  (M×N grid support)
Single source of truth for all modules.
"""
import math
from collections import deque

TYPES = ["residential", "hospital", "school", "industrial", "powerplant", "depot"]

TYPE_COLORS = {
    "residential": (255, 160, 80),
    "hospital":    (255, 80,  100),
    "school":      (80,  200, 120),
    "industrial":  (180, 120, 60),
    "powerplant":  (255, 220, 60),
    "depot":       (80,  160, 255),
}

TYPE_LABELS = {
    "residential": "Residential",
    "hospital":    "Hospital",
    "school":      "School",
    "industrial":  "Industrial",
    "powerplant":  "Power Plant",
    "depot":       "Amb. Depot",
}

TYPE_ICONS = {
    "residential": "🏠",
    "hospital":    "🏥",
    "school":      "🏫",
    "industrial":  "🏭",
    "powerplant":  "⚡",
    "depot":       "🚑",
}

POP_LEVELS = {"low": 30, "medium": 65, "high": 90}


class Node:
    _id_counter = 0

    def __init__(self, gx, gy, ntype, px=0, py=0):
        Node._id_counter += 1
        self.id   = Node._id_counter
        self.gx   = gx        # grid col  (0 … cols-1)
        self.gy   = gy        # grid row  (0 … rows-1)
        self.px   = px        # pixel x
        self.py   = py        # pixel y
        self.type = ntype
        self.population_density = POP_LEVELS["medium"]
        self.risk_level      = "medium"
        self.risk_multiplier = 1.0
        self.industrial_distance = 999
        self.accessibility   = True
        self.pulse    = 0.0
        self.selected = False
        self.hover    = False

    def pos(self):
        return (self.px, self.py)


class Edge:
    def __init__(self, a, b, base_cost=1.0):
        self.a = a
        self.b = b
        self.base_cost     = base_cost
        self.effective_cost = base_cost
        self.blocked   = False
        self.in_mst    = False
        self.redundant = False

    def key(self):
        return (min(self.a, self.b), max(self.a, self.b))


class CityGraph:
    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.edges: dict[tuple, Edge] = {}
        # M×N grid — cols = columns (x-axis), rows = rows (y-axis)
        self.grid_cols = 6
        self.grid_rows = 6
        self.start_node_id = None
        self.end_node_id   = None

    @property
    def grid_size(self):
        """Legacy compat — returns cols (used where square assumed)."""
        return self.grid_cols

    def clear(self):
        self.nodes.clear()
        self.edges.clear()
        Node._id_counter = 0
        self.start_node_id = None
        self.end_node_id   = None

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def remove_node(self, nid: int):
        self.nodes.pop(nid, None)
        dead = [k for k, e in self.edges.items() if e.a == nid or e.b == nid]
        for k in dead:
            del self.edges[k]
        if self.start_node_id == nid: self.start_node_id = None
        if self.end_node_id   == nid: self.end_node_id   = None

    def add_edge(self, a_id, b_id, base_cost=1.0):
        key = (min(a_id, b_id), max(a_id, b_id))
        if key not in self.edges:
            self.edges[key] = Edge(a_id, b_id, base_cost)
        return self.edges[key]

    def get_edge(self, a_id, b_id):
        return self.edges.get((min(a_id, b_id), max(a_id, b_id)))

    def node_at(self, gx, gy):
        for n in self.nodes.values():
            if n.gx == gx and n.gy == gy:
                return n
        return None

    def neighbors(self, nid, include_blocked=False):
        result = []
        for e in self.edges.values():
            if e.blocked and not include_blocked:
                continue
            if e.a == nid and e.b in self.nodes:
                result.append((self.nodes[e.b], e))
            elif e.b == nid and e.a in self.nodes:
                result.append((self.nodes[e.a], e))
        return result

    def bfs_hops(self, start_id):
        dist = {start_id: 0}
        q = deque([start_id])
        while q:
            cur = q.popleft()
            for nb, _ in self.neighbors(cur):
                if nb.id not in dist:
                    dist[nb.id] = dist[cur] + 1
                    q.append(nb.id)
        return dist

    def nearest_node_px(self, px, py, max_dist=9999):
        best, bd = None, max_dist
        for n in self.nodes.values():
            d = math.hypot(n.px - px, n.py - py)
            if d < bd:
                bd, best = d, n
        return best, bd

    def nearest_edge_px(self, px, py, max_dist=30):
        best_e, best_d = None, max_dist
        for e in self.edges.values():
            na = self.nodes.get(e.a)
            nb = self.nodes.get(e.b)
            if na is None or nb is None:
                continue
            d = _pt_seg(px, py, na.px, na.py, nb.px, nb.py)
            if d < best_d:
                best_d, best_e = d, e
        return best_e, best_d

    def nodes_of_type(self, t):
        return [n for n in self.nodes.values() if n.type == t]

    def update_effective_costs(self):
        for e in self.edges.values():
            na = self.nodes.get(e.a)
            nb = self.nodes.get(e.b)
            if na and nb:
                mult = max(na.risk_multiplier, nb.risk_multiplier)
                bc = 0.8 if (na.type == "residential" or nb.type == "residential") else 1.0
                e.base_cost = bc
                e.effective_cost = bc * mult


def _pt_seg(px, py, ax, ay, bx, by):
    dx, dy = bx-ax, by-ay
    if dx == 0 and dy == 0:
        return math.hypot(px-ax, py-ay)
    t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)))
    return math.hypot(px-(ax+t*dx), py-(ay+t*dy))
