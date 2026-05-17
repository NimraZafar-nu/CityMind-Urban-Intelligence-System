"""
CityMind - Challenge 1: CSP City Layout Planner
Correctly implements all 3 hard constraints. Fast, no infinite loops.

CONSTRAINTS:
  C1: Industrial NOT adjacent (4-connected) to School or Hospital
  C2: Every Residential within 3 grid-hops of at least 1 Hospital
  C3: Every PowerPlant within 2 grid-hops of at least 1 Industrial zone

APPROACH (chosen for correctness AND speed):
  1. Shuffle grid positions, select N positions for N nodes
  2. Greedy type assignment respecting C1 adjacency (fast, never hangs)
  3. Enforce exact type counts by simple swaps (O(N), no dict copies)
  4. Auto-repair remaining violations iteratively (max 150 iterations)
  5. Retry up to 6 times, keep best result

Runtime heal (on node add/remove/toggle):
  - Build assignment from current graph
  - Run _auto_repair with pinned positions
  - Write fixed types back to graph nodes
  - Never moves nodes, only changes types
"""
import random
from collections import deque
from graph import CityGraph, Node, TYPES


# ── Adjacency helpers ─────────────────────────────────────────────────────────

def _nb4(gx, gy, cols, rows):
    for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
        nx, ny = gx+dx, gy+dy
        if 0 <= nx < cols and 0 <= ny < rows:
            yield (nx, ny)


def _grid_bfs(pos_set, start):
    """BFS over pos_set. Returns {pos: hop_count}."""
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
            nb = (cur[0]+dx, cur[1]+dy)
            if nb in pos_set and nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return dist


# ── Constraint checks ─────────────────────────────────────────────────────────

def _c1_ok(t1, t2):
    """True if types t1 and t2 can be adjacent."""
    BAD = {("industrial","school"),("industrial","hospital"),
           ("school","industrial"),("hospital","industrial")}
    return (t1, t2) not in BAD


def _adj_ok(pos, ntype, assignment, cols, rows):
    """C1 check: placing ntype at pos doesn't violate adjacency."""
    for nb in _nb4(*pos, cols, rows):
        if nb in assignment and not _c1_ok(ntype, assignment[nb]):
            return False
    return True


def check_all(assignment, cols, rows):
    """
    Returns list of violation dicts.
    Each dict has: constraint (C1/C2/C3), pos, rule (short), detail (long).
    """
    pos_set    = set(assignment)
    viols      = []
    hospitals  = {p for p,t in assignment.items() if t=="hospital"}
    industrials= {p for p,t in assignment.items() if t=="industrial"}

    # C1 – adjacency
    seen = set()
    for pos, t in assignment.items():
        if t == "industrial":
            for nb in _nb4(*pos, cols, rows):
                if nb in assignment and assignment[nb] in ("school","hospital"):
                    key = (min(pos,nb), max(pos,nb))
                    if key not in seen:
                        seen.add(key)
                        viols.append({"constraint":"C1","pos":pos,
                            "rule": f"Industrial adj to {assignment[nb]}",
                            "detail": f"Industrial {pos} next to {assignment[nb]} {nb}"})

    # C2 – residential within 3 hops of hospital
    # Pre-build BFS from each hospital once
    hosp_bfs = {h: _grid_bfs(pos_set, h) for h in hospitals}
    for pos, t in assignment.items():
        if t != "residential": continue
        if not hospitals:
            viols.append({"constraint":"C2","pos":pos,
                "rule":"No hospital in city",
                "detail":f"Residential {pos}: no hospital exists"})
        else:
            best = min((hosp_bfs[h].get(pos, 999) for h in hospitals), default=999)
            if best > 3:
                viols.append({"constraint":"C2","pos":pos,
                    "rule":f"Residential {best} hops from hospital (max 3)",
                    "detail":f"Residential {pos} is {best} hops from nearest hospital"})

    # C3 – powerplant within 2 hops of industrial
    ind_bfs = {i: _grid_bfs(pos_set, i) for i in industrials}
    for pos, t in assignment.items():
        if t != "powerplant": continue
        if not industrials:
            viols.append({"constraint":"C3","pos":pos,
                "rule":"No industrial zone in city",
                "detail":f"PowerPlant {pos}: no industrial zone exists"})
        else:
            best = min((ind_bfs[i].get(pos, 999) for i in industrials), default=999)
            if best > 2:
                viols.append({"constraint":"C3","pos":pos,
                    "rule":f"PowerPlant {best} hops from industrial (max 2)",
                    "detail":f"PowerPlant {pos} is {best} hops from nearest industrial"})

    return viols


def conflict_report(assignment, cols, rows):
    viols = check_all(assignment, cols, rows)
    if not viols:
        return "All CSP constraints satisfied."
    lines = [f"⚠ {len(viols)} violation(s):"]
    by_c = {}
    for v in viols:
        by_c.setdefault(v["constraint"], []).append(v["rule"])
    for c, rules in sorted(by_c.items()):
        lines.append(f"  {c}: {rules[0]}")
        if c=="C1": lines.append("    Fix: move industrial away from schools/hospitals")
        if c=="C2": lines.append("    Fix: add hospital within 3 hops of residential areas")
        if c=="C3": lines.append("    Fix: place industrial zone within 2 hops of power plant")
    lines.append("Auto-repair applied.")
    return "\n".join(lines)


# ── Fast greedy placer ────────────────────────────────────────────────────────

def _greedy_place(positions, cols, rows, type_counts):
    """
    Assign types greedily.
    - Builds a pool of types matching type_counts.
    - For each position tries to assign the next needed type that satisfies C1.
    - Falls back to 'residential' if nothing fits.
    - O(N*6) - never hangs.
    """
    # Build ordered pool: required types first, then fill with residential
    pool = []
    for t in ("hospital","depot","industrial","powerplant","school","residential"):
        cnt = type_counts.get(t, 0)
        pool.extend([t]*cnt)
    while len(pool) < len(positions):
        pool.append("residential")
    pool = pool[:len(positions)]

    # Shuffle positions for randomness
    pos_list = list(positions)
    random.shuffle(pos_list)

    assignment = {}
    for pos in pos_list:
        placed = False
        # Try each type in priority order (most constrained first)
        for t in ("hospital","depot","industrial","powerplant","school","residential"):
            if _adj_ok(pos, t, assignment, cols, rows):
                assignment[pos] = t
                placed = True
                break
        if not placed:
            assignment[pos] = "residential"   # absolute fallback

    return assignment


# ── Count enforcement ─────────────────────────────────────────────────────────

def _fix_counts(assignment, positions, type_counts):
    """
    Adjust assignment to match type_counts.
    Simple O(N) approach: no dict copies, no _local_ok calls.
    Just swap types directly. CSP repair will fix any C1 violations caused.
    """
    a = dict(assignment)

    # Count current
    cur = {}
    for t in TYPES: cur[t] = 0
    for t in a.values(): cur[t] = cur.get(t,0)+1

    # Over-represented types → convert to residential
    for t, want in type_counts.items():
        if t == "residential": continue
        surplus = cur.get(t,0) - want
        for p in list(positions):
            if surplus <= 0: break
            if a.get(p) == t:
                a[p] = "residential"
                cur[t] -= 1
                cur["residential"] = cur.get("residential",0)+1
                surplus -= 1

    # Under-represented types → convert from residential
    for t, want in type_counts.items():
        if t == "residential": continue
        deficit = want - cur.get(t,0)
        for p in list(positions):
            if deficit <= 0: break
            if a.get(p) == "residential":
                a[p] = t
                cur["residential"] -= 1
                cur[t] = cur.get(t,0)+1
                deficit -= 1

    # Guarantee at least 1 hospital, 1 depot, 1 industrial
    for req in ("hospital","depot","industrial"):
        if cur.get(req,0) == 0:
            for p in positions:
                if a.get(p) == "residential":
                    a[p] = req
                    cur["residential"] -= 1
                    cur[req] = cur.get(req,0)+1
                    break

    return a


# ── Self-healing repair ────────────────────────────────────────────────────────

def _auto_repair(assignment, cols, rows, pinned, log_fn=None):
    """
    Fix violations without moving pinned nodes.
    Max 150 iterations — guaranteed to terminate.
    """
    a = dict(assignment)
    pos_list = sorted(a.keys())

    for _iter in range(150):
        viols = check_all(a, cols, rows)
        if not viols:
            break
        v    = viols[0]
        vpos = v["pos"]
        c    = v["constraint"]

        if c == "C1":
            # Industrial violating adjacency
            if a.get(vpos) == "industrial" and vpos not in pinned:
                for safe in ("residential","depot","powerplant"):
                    if _adj_ok(vpos, safe, {p:t for p,t in a.items() if p!=vpos}, cols, rows):
                        if log_fn: log_fn(f"↻ C1: industrial→{safe} at {vpos}")
                        a[vpos] = safe
                        break
            else:
                # Fix the school/hospital neighbour
                for nb in _nb4(*vpos, cols, rows):
                    if nb in a and a[nb] in ("school","hospital") and nb not in pinned:
                        a[nb] = "residential"
                        if log_fn: log_fn(f"↻ C1: {a[nb]}→residential at {nb}")
                        break

        elif c == "C2":
            # Residential too far from hospital → convert nearest node to hospital
            cands = sorted(
                [p for p in pos_list if p not in pinned and p!=vpos
                 and a[p] not in ("industrial","hospital")],
                key=lambda p: abs(p[0]-vpos[0])+abs(p[1]-vpos[1]))
            placed = False
            for cp in cands:
                tmp = {p:t for p,t in a.items() if p!=cp}
                if _adj_ok(cp, "hospital", tmp, cols, rows):
                    if log_fn: log_fn(f"↻ C2: {a[cp]}→hospital at {cp}")
                    a[cp] = "hospital"
                    placed = True
                    break
            if not placed and vpos not in pinned:
                a[vpos] = "hospital"   # convert the residential itself
                if log_fn: log_fn(f"↻ C2: residential→hospital at {vpos}")

        elif c == "C3":
            # PowerPlant too far from industrial → convert nearest node to industrial
            cands = sorted(
                [p for p in pos_list if p not in pinned and p!=vpos
                 and a[p] not in ("industrial","school","hospital")],
                key=lambda p: abs(p[0]-vpos[0])+abs(p[1]-vpos[1]))
            placed = False
            for cp in cands:
                tmp = {p:t for p,t in a.items() if p!=cp}
                if _adj_ok(cp, "industrial", tmp, cols, rows):
                    if log_fn: log_fn(f"↻ C3: {a[cp]}→industrial at {cp}")
                    a[cp] = "industrial"
                    placed = True
                    break
            if not placed and vpos not in pinned:
                a[vpos] = "industrial"
                if log_fn: log_fn(f"↻ C3: powerplant→industrial at {vpos}")
        else:
            break

    return a


# ── Main CSP Solver ───────────────────────────────────────────────────────────

class CSPSolver:
    def __init__(self, graph: CityGraph):
        self.graph           = graph
        self.last_violations = []

    def solve(self, cols, rows, type_counts, city_rect, log_fn=None):
        """
        Place all nodes satisfying CSP constraints. Never hangs.
        Returns list of violations (empty = fully satisfied).
        """
        g = self.graph
        g.clear()
        g.grid_cols = cols
        g.grid_rows = rows

        ax, ay, aw, ah = city_rect
        cw, ch = aw/cols, ah/rows
        total = max(4, min(sum(type_counts.values()), cols*rows))

        best_asgn  = None
        best_viols = None

        for attempt in range(6):
            # Pick random grid positions
            all_pos = [(gx,gy) for gx in range(cols) for gy in range(rows)]
            random.shuffle(all_pos)
            chosen = all_pos[:total]

            # Greedy assignment (fast, always terminates)
            asgn = _greedy_place(chosen, cols, rows, type_counts)

            # Fix type counts (fast, no dict copies)
            asgn = _fix_counts(asgn, chosen, type_counts)

            # Repair constraint violations
            asgn  = _auto_repair(asgn, cols, rows, pinned=set(), log_fn=log_fn)
            viols = check_all(asgn, cols, rows)

            if best_viols is None or len(viols) < len(best_viols):
                best_asgn  = dict(asgn)
                best_viols = viols

            if not viols:
                break
            if attempt < 5 and log_fn:
                log_fn(f"↻ Attempt {attempt+1}: {len(viols)} violation(s), retrying…")

        # Place into graph
        g.nodes.clear()
        g.edges.clear()
        Node._id_counter = 0

        for (gx, gy), ntype in best_asgn.items():
            px = int(ax + (gx+0.5)*cw)
            py = int(ay + (gy+0.5)*ch)
            n  = Node(gx, gy, ntype, px, py)
            n.population_density = random.randint(20, 90)
            g.add_node(n)

        self.last_violations = best_viols or []
        if self.last_violations:
            if log_fn:
                log_fn(f"⚠ CSP: {len(self.last_violations)} violation(s) remain")
                log_fn(conflict_report(best_asgn, cols, rows))
        else:
            if log_fn:
                log_fn(f"✓ CSP: {len(g.nodes)} nodes on {cols}×{rows} — all constraints satisfied")

        return self.last_violations

    def heal(self, pinned_positions=None, log_fn=None):
        """
        Fix violations in current graph. Does NOT move nodes.
        Called after every node type change, add, or remove.
        """
        g    = self.graph
        cols = g.grid_cols
        rows = g.grid_rows

        id_map     = {(n.gx,n.gy): n.id   for n in g.nodes.values()}
        assignment = {(n.gx,n.gy): n.type for n in g.nodes.values()}
        pinned     = set(pinned_positions) if pinned_positions else set()

        viols = check_all(assignment, cols, rows)
        if not viols:
            self.last_violations = []
            if log_fn: log_fn("✓ CSP: all constraints satisfied")
            return []

        if log_fn: log_fn(f"↻ CSP heal: {len(viols)} violation(s)")
        fixed = _auto_repair(assignment, cols, rows, pinned, log_fn)

        # Write back to graph
        changed = 0
        for pos, ntype in fixed.items():
            nid = id_map.get(pos)
            if nid and nid in g.nodes and g.nodes[nid].type != ntype:
                g.nodes[nid].type = ntype
                changed += 1

        if changed and log_fn:
            log_fn(f"↻ CSP: healed {changed} node type(s)")

        self.last_violations = check_all(fixed, cols, rows)
        if not self.last_violations:
            if log_fn: log_fn("✓ CSP: all constraints satisfied")
        else:
            if log_fn:
                log_fn(f"⚠ CSP: {len(self.last_violations)} violation(s) remain")
                log_fn(conflict_report(fixed, cols, rows))

        return self.last_violations

    def validate_current(self):
        g = self.graph
        asgn = {(n.gx,n.gy): n.type for n in g.nodes.values()}
        self.last_violations = check_all(asgn, g.grid_cols, g.grid_rows)
        return self.last_violations

    def get_conflict_report(self):
        g    = self.graph
        asgn = {(n.gx,n.gy): n.type for n in g.nodes.values()}
        return conflict_report(asgn, g.grid_cols, g.grid_rows)
