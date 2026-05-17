"""
CityMind – Challenge 4: Emergency Routing (A* with real-time rerouting)
From scratch – no external libraries.

HOW ROUTING WORKS:
  1. User sets a TEAM START node (right-click → Set Team START, or Set Team button).
  2. User places CIVILIAN nodes (right-click → Add Civilian, or Place Civ button).
  3. User clicks SIMULATE.
  4. A* finds the shortest path from team_pos → first civilian.
  5. Every simulation step, team moves ONE node along the path.
  6. When team reaches a civilian → mark rescued → A* runs to next civilian.
  7. If ANY road on the current path is blocked (flood/manual), A* reruns IMMEDIATELY
     from the team's CURRENT position — NOT from start.
  8. Repeat until all civilians are rescued.

HEURISTIC: Manhattan distance (admissible → A* always finds optimal path).
COST: edge.effective_cost = base_cost × risk_multiplier (set by ML module).
"""
import heapq
from graph import CityGraph


# ─────────────────────────────────────────────────────────────────────────────
# Pure A* function
# ─────────────────────────────────────────────────────────────────────────────

def astar(graph: CityGraph, start_id: int, goal_id: int):
    """
    Find shortest path from start_id to goal_id using A*.

    Returns:
        (path, cost) where path = [start_id, ..., goal_id]
        (None, inf)  if no path exists (graph disconnected or all roads blocked)

    The heuristic h(n) = Manhattan distance on grid coordinates.
    This is ADMISSIBLE because every grid edge costs at least 0.8
    (residential road discount), so the heuristic never overestimates.
    """
    if start_id not in graph.nodes or goal_id not in graph.nodes:
        return None, float("inf")

    # If start == goal, we are already there
    if start_id == goal_id:
        return [start_id], 0.0

    goal_n = graph.nodes[goal_id]

    def h(nid):
        """Manhattan distance heuristic."""
        n = graph.nodes.get(nid)
        if n is None:
            return 0
        return abs(n.gx - goal_n.gx) + abs(n.gy - goal_n.gy)

    # Priority queue entries: (f_score, g_score, node_id)
    open_set = [(h(start_id), 0.0, start_id)]
    g_cost   = {start_id: 0.0}
    parent   = {}
    closed   = set()

    while open_set:
        f, g, cur = heapq.heappop(open_set)

        if cur == goal_id:
            # Reconstruct path by following parent pointers
            path = []
            node = goal_id
            while node in parent:
                path.append(node)
                node = parent[node]
            path.append(start_id)
            path.reverse()
            return path, g_cost[goal_id]

        if cur in closed:
            continue
        closed.add(cur)

        # Expand neighbors (graph.neighbors skips blocked edges automatically)
        for nb, edge in graph.neighbors(cur):
            if nb.id in closed:
                continue
            new_g = g_cost[cur] + edge.effective_cost
            if new_g < g_cost.get(nb.id, 1e18):
                g_cost[nb.id] = new_g
                parent[nb.id] = cur
                heapq.heappush(open_set, (new_g + h(nb.id), new_g, nb.id))

    return None, float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Emergency Router
# ─────────────────────────────────────────────────────────────────────────────

class EmergencyRouter:
    """
    Manages the medical team's mission end-to-end.

    State:
        team_pos     : int   — current node ID where the team is
        civilians    : list  — node IDs to rescue, in placement order
        visited      : list  — civilians already rescued
        current_path : list  — A* path to next target [team_pos, ..., target]
        current_cost : float — total cost of current_path
        active       : bool  — True while mission is running
        total_steps  : int   — how many steps taken so far
    """

    def __init__(self, graph: CityGraph):
        self.graph        = graph
        self.team_pos     = None
        self.civilians    = []       # all target node IDs (in order)
        self.visited      = []       # rescued node IDs
        self.current_path = []       # A* path: [cur, ..., next_target]
        self.current_cost = float("inf")
        self.active       = False
        self.status       = "Idle — set team START and place civilians"
        self.total_steps  = 0

    # ── Setup ─────────────────────────────────────────────────────────────────

    def set_start(self, node_id: int, log_fn=None):
        """Set the team's starting node. Recalculates path if mission is active."""
        if node_id not in self.graph.nodes:
            if log_fn:
                log_fn(f"[C4] Invalid start node #{node_id}")
            return
        self.team_pos = node_id
        self.status   = f"Team ready at #{node_id}"
        if log_fn:
            log_fn(f"[C4] Team START → #{node_id}")
        # If mission already running, recalc from new start
        if self.active:
            self._recalc(log_fn)

    def add_civilian(self, node_id: int, log_fn=None):
        """Add a civilian target. Duplicates are ignored."""
        if node_id in self.civilians:
            if log_fn:
                log_fn(f"[C4] Node #{node_id} already a civilian target")
            return
        self.civilians.append(node_id)
        if log_fn:
            log_fn(f"[C4] Civilian added at #{node_id} ({len(self.civilians)} total)")
        # If mission running and we have no current path, recalc
        if self.active and not self.current_path:
            self._recalc(log_fn)

    def remove_civilian(self, node_id: int, log_fn=None):
        """Remove a civilian target (useful if node is deleted)."""
        if node_id in self.civilians:
            self.civilians.remove(node_id)
        if node_id in self.visited:
            self.visited.remove(node_id)
        # If this was the current target, recalculate
        target = self.next_target()
        if self.current_path and target != (self.current_path[-1] if self.current_path else None):
            self._recalc(log_fn)

    def start_mission(self, log_fn=None) -> bool:
        """
        Begin the mission. Returns True if successfully started.
        Checks: team start set, at least 1 civilian placed.
        """
        if not self.team_pos:
            if log_fn:
                log_fn("[C4] ⚠ Set team START first (right-click a node → Set Team START)")
            return False
        if not self.civilians:
            if log_fn:
                log_fn("[C4] ⚠ Place at least 1 civilian first (right-click → Add Civilian)")
            return False
        if self.team_pos not in self.graph.nodes:
            if log_fn:
                log_fn(f"[C4] ⚠ Team start node #{self.team_pos} no longer exists")
            return False

        self.visited     = []
        self.total_steps = 0
        self.active      = True

        # Check if team is already sitting on a civilian node
        self._check_at_civilian(log_fn)

        # Calculate initial path
        if self.active:
            self._recalc(log_fn)

        return True

    def reset(self):
        """Full reset — clears all mission state."""
        self.team_pos     = None
        self.civilians    = []
        self.visited      = []
        self.current_path = []
        self.current_cost = float("inf")
        self.active       = False
        self.status       = "Idle"
        self.total_steps  = 0

    # ── Real-time rerouting ───────────────────────────────────────────────────

    def on_edge_blocked(self, edge, step: int, log_fn=None):
        """
        Called the MOMENT any edge is blocked (flood or manual).
        If the blocked edge is on the current A* path, reroutes IMMEDIATELY
        from the team's CURRENT position (not from start).
        """
        if not self.active or len(self.current_path) < 2:
            return

        # Build set of edges on current path
        path_edges = set()
        for i in range(len(self.current_path) - 1):
            a, b = self.current_path[i], self.current_path[i + 1]
            path_edges.add((min(a, b), max(a, b)))

        edge_key = (min(edge.a, edge.b), max(edge.a, edge.b))
        if edge_key in path_edges:
            if log_fn:
                log_fn(f"[Step {step}] 🚨 Road ({edge.a}↔{edge.b}) blocked! "
                       f"Rerouting A* from #{self.team_pos}…")
            self._recalc(log_fn, step=step)
        # If the blocked edge is NOT on the path, no rerouting needed

    # ── Simulation step ───────────────────────────────────────────────────────

    def step(self, log_fn=None):
        """
        Advance the team ONE node along current_path.

        Called once per simulation tick (every SIM_INTERVAL frames).

        Logic:
          1. If path is empty → try recalc (maybe a new path is needed).
          2. Move team from current_path[0] to current_path[1].
          3. Remove current_path[0] (team has left that node).
          4. Check if new position is a civilian → mark rescued.
          5. If all civilians rescued → mission complete.
          6. Otherwise → if we just rescued one, recalc path to next.
        """
        if not self.active:
            return

        # If no path, try recalculating
        if len(self.current_path) < 2:
            self._recalc(log_fn)
            if len(self.current_path) < 2:
                # Still no path — all routes may be blocked
                target = self.next_target()
                if target is not None:
                    self.status = f"⚠ No path to #{target} — all routes blocked?"
                return

        self.total_steps += 1

        # Move one step forward
        self.team_pos     = self.current_path[1]
        self.current_path = self.current_path[1:]   # drop the node we just left

        if log_fn:
            log_fn(f"[Step {self.total_steps}] Team → #{self.team_pos} "
                   f"(path remaining: {len(self.current_path)-1} more steps)")

        # Check if we just landed on a civilian
        self._check_at_civilian(log_fn)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_at_civilian(self, log_fn=None):
        """
        Check if the team's CURRENT position is a civilian target.
        If yes, mark them rescued and move to the next target.

        This is separate from step() so it can also be called at mission start
        (in case the team starts on a civilian node).
        """
        if not self.active:
            return

        # Keep checking in case multiple civilians are at the same node
        while True:
            target = self.next_target()
            if target is None:
                # All civilians rescued!
                self.active       = False
                self.current_path = []
                self.current_cost = 0.0
                self.status       = "🎉 Mission complete — all civilians rescued!"
                if log_fn:
                    log_fn(f"[C4] ✅ MISSION COMPLETE in {self.total_steps} steps!")
                return

            if self.team_pos == target:
                # Rescue this civilian
                self.visited.append(target)
                done  = len(self.visited)
                total = len(self.civilians)
                self.status = f"Rescued #{target}! ({done}/{total} done)"
                if log_fn:
                    log_fn(f"[Step {self.total_steps}] 🚑 RESCUED civilian #{target}! "
                           f"({done}/{total} done)")
            else:
                break   # not at a civilian — stop checking

        # Recalculate path to the next unvisited civilian
        if self.active:
            self._recalc(log_fn)

    def next_target(self):
        """Return the next unvisited civilian node ID, or None if all done."""
        for cid in self.civilians:
            if cid not in self.visited:
                return cid
        return None

    def _recalc(self, log_fn=None, step: int = 0):
        """
        Run A* from team_pos to the next unvisited civilian.
        Updates current_path and current_cost.
        Logs path details or warns if no path exists.
        """
        if not self.team_pos:
            return

        target = self.next_target()
        if target is None:
            # No more civilians — mission should be complete
            self.current_path = []
            self.current_cost = 0.0
            return

        # Verify target still exists in graph
        if target not in self.graph.nodes:
            if log_fn:
                log_fn(f"[C4] Civilian #{target} no longer in graph — skipping")
            self.visited.append(target)  # auto-skip deleted civilian
            self._recalc(log_fn, step)
            return

        path, cost = astar(self.graph, self.team_pos, target)
        self.current_path = path if path else []
        self.current_cost = cost

        if path:
            self.status = (f"→ #{target}: {len(path)-1} steps, "
                           f"cost={cost:.2f} | rescued {len(self.visited)}/{len(self.civilians)}")
            if log_fn:
                log_fn(f"[Step {step}] A* #{self.team_pos}→#{target}: "
                       f"{len(path)} nodes, cost={cost:.2f}")
        else:
            self.status = f"⚠ No path to #{target}! (all routes blocked?)"
            if log_fn:
                log_fn(f"[C4] ⚠ No path from #{self.team_pos} to #{target}!")
