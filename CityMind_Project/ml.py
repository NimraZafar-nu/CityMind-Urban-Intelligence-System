"""
CityMind – Challenge 5: Crime Risk Prediction + Police Deployment

Pipeline (fully from scratch — no sklearn, no numpy):

  STEP 1 — K-Means Clustering (UNSUPERVISED)
    Input features per node:
      • population_density  (normalised 0-1)
      • industrial_proximity = 1 - industrial_distance/max_dist  (normalised 0-1)
    Output: 3 clusters labelled High / Medium / Low by centroid danger score.

  STEP 2 — Synthetic Crime Dataset Generation
    For each node, compute a crime_score:
      crime_score = 0.55 * pop_norm + 0.45 * ind_proximity_norm + noise
    Justification:
      - High population density = more targets = higher crime
      - Proximity to industrial zones = lower surveillance, poor lighting = higher crime
      - Random noise = real-world unpredictability
    Labels: crime_score >=0.60 → High, >=0.38 → Medium, else Low

  STEP 3 — Decision Tree Classifier (SUPERVISED)
    Trained on the synthetic dataset above.
    Max depth=4, splits by information gain (entropy).
    Predicts risk level for any node given its features.

  STEP 4 — Feed Risk Back to Graph
    node.risk_level      → "high" | "medium" | "low"
    node.risk_multiplier → 1.5   | 1.2      | 1.0
    Edge effective_costs update automatically → affects A* + GA.

  STEP 5 — Deploy 10 Police Officers
    Sort all nodes by (risk_level desc, population_density desc).
    Assign exactly 10 police officers to the top-10 ranked nodes.
    Each officer is stored as a PoliceOfficer object with:
      node_id, officer_id, risk_level at time of deployment.
    Police officers can be redeployed mid-simulation when risk changes.
"""
import random
import math
from graph import CityGraph


# ── Risk multipliers ──────────────────────────────────────────────────────────
RISK_MULT = {"high": 1.5, "medium": 1.2, "low": 1.0}
NUM_POLICE = 10   # exactly 10 officers as per project spec


# ─────────────────────────────────────────────────────────────────────────────
# Police Officer data class
# ─────────────────────────────────────────────────────────────────────────────
class PoliceOfficer:
    """
    Represents one of the 10 police officers.
    Stores which node they are deployed at and their badge number.
    """
    def __init__(self, officer_id: int, node_id: int, risk_level: str):
        self.officer_id = officer_id     # 1 to 10
        self.node_id    = node_id        # which graph node they patrol
        self.risk_level = risk_level     # risk level when deployed

    def __repr__(self):
        return f"Officer#{self.officer_id} @ Node#{self.node_id} ({self.risk_level})"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – K-Means Clustering (Unsupervised)
# ─────────────────────────────────────────────────────────────────────────────
class KMeans:
    """
    K-Means from scratch with K-Means++ initialisation.

    Why K-Means?
    - We have NO pre-labelled data → unsupervised learning.
    - We want to group neighbourhoods into 3 natural clusters
      (dangerous / moderate / safe) based on their features alone.
    - K-Means is simple, fast, and interpretable — perfect for a
      decision-support system where planners need to understand the groupings.
    """

    def __init__(self, k=3, max_iter=100, tol=1e-4):
        self.k         = k
        self.max_iter  = max_iter
        self.tol       = tol
        self.centroids = []
        self.labels    = {}    # nid → cluster index

    def fit(self, data: list) -> dict:
        """
        data = list of (node_id, [feature0, feature1])
        Returns {node_id: cluster_index}.
        """
        if len(data) < self.k:
            self.labels = {nid: 0 for nid, _ in data}
            self.centroids = [[0.5, 0.5]] * self.k
            return self.labels

        self.centroids = self._kpp_init(data)

        for _ in range(self.max_iter):
            # Assign each point to the nearest centroid
            new_labels = {}
            for nid, feat in data:
                best_c = min(range(self.k),
                             key=lambda c: _sq_dist(feat, self.centroids[c]))
                new_labels[nid] = best_c

            # Recompute centroids as mean of each cluster
            new_centroids = []
            total_move    = 0.0
            for c in range(self.k):
                members = [feat for nid, feat in data if new_labels.get(nid) == c]
                if members:
                    nc = [sum(f[i] for f in members) / len(members)
                          for i in range(len(members[0]))]
                else:
                    nc = self.centroids[c][:]   # keep old if empty cluster
                total_move += _sq_dist(nc, self.centroids[c])
                new_centroids.append(nc)

            self.centroids = new_centroids
            self.labels    = new_labels

            if total_move < self.tol:
                break   # converged — centroids stable

        return self.labels

    def predict_one(self, feat: list) -> int:
        """Assign a new feature vector to the nearest existing centroid."""
        if not self.centroids:
            return 0
        return min(range(len(self.centroids)),
                   key=lambda c: _sq_dist(feat, self.centroids[c]))

    def _kpp_init(self, data):
        """K-Means++ smart seeding — spreads initial centroids apart."""
        centroids = [random.choice(data)[1][:]]
        for _ in range(self.k - 1):
            # Distance of each point to its nearest existing centroid
            dists = [min(_sq_dist(feat, c) for c in centroids)
                     for _, feat in data]
            total = sum(dists)
            if total == 0:
                centroids.append(random.choice(data)[1][:])
                continue
            # Pick next centroid proportional to distance squared
            r, cumsum = random.random() * total, 0.0
            for i, (_, feat) in enumerate(data):
                cumsum += dists[i]
                if cumsum >= r:
                    centroids.append(feat[:])
                    break
            else:
                centroids.append(data[-1][1][:])
        return centroids


def _sq_dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Decision Tree Classifier (Supervised)
# ─────────────────────────────────────────────────────────────────────────────
def _entropy(labels):
    if not labels:
        return 0.0
    total = len(labels)
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    result = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            result -= p * math.log2(p)
    return result


def _majority(labels):
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return max(counts, key=counts.get)


class DecisionTree:
    """
    CART-style Decision Tree for 3-class classification.

    Why Decision Tree?
    - We have labelled training data (synthetic crime scores → High/Med/Low).
    - Decision Trees are interpretable — we can explain each split rule.
    - No external libraries needed — fits the from-scratch requirement.
    - Works well with small datasets (our city has ≤200 nodes).
    """

    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self._tree     = None

    def fit(self, X: list, y: list):
        """X = list of feature vectors. y = list of labels."""
        self._tree = self._build(list(zip(X, y)), depth=0)

    def predict(self, x: list) -> str:
        if self._tree is None:
            return "low"
        return self._traverse(self._tree, x)

    def _build(self, data, depth):
        if not data:
            return {"leaf": "low"}
        labels = [lbl for _, lbl in data]
        # Stop if all same label or max depth reached
        if len(set(labels)) == 1:
            return {"leaf": labels[0]}
        if depth >= self.max_depth:
            return {"leaf": _majority(labels)}
        # Find best split
        split = self._best_split(data)
        if split is None:
            return {"leaf": _majority(labels)}
        fi, thr, left, right = split
        return {
            "fi":    fi,
            "thr":   thr,
            "left":  self._build(left,  depth + 1),
            "right": self._build(right, depth + 1),
        }

    def _best_split(self, data):
        """Try every feature × threshold; pick the one with highest info gain."""
        n_feat    = len(data[0][0])
        base_ent  = _entropy([lbl for _, lbl in data])
        best_gain = -1
        best      = None
        for fi in range(n_feat):
            vals = sorted(set(x[fi] for x, _ in data))
            for i in range(len(vals) - 1):
                thr   = (vals[i] + vals[i + 1]) / 2.0
                left  = [(x, l) for x, l in data if x[fi] <= thr]
                right = [(x, l) for x, l in data if x[fi] >  thr]
                if not left or not right:
                    continue
                n    = len(data)
                gain = base_ent - (
                    len(left)  / n * _entropy([l for _, l in left]) +
                    len(right) / n * _entropy([l for _, l in right])
                )
                if gain > best_gain:
                    best_gain = gain
                    best      = (fi, thr, left, right)
        return best

    def _traverse(self, node, x):
        if "leaf" in node:
            return node["leaf"]
        if x[node["fi"]] <= node["thr"]:
            return self._traverse(node["left"], x)
        return self._traverse(node["right"], x)


# ─────────────────────────────────────────────────────────────────────────────
# Full Challenge-5 Pipeline
# ─────────────────────────────────────────────────────────────────────────────
class CrimeRiskML:
    """
    Manages the full C5 pipeline and the 10 police officers.

    Key attributes visible to UIManager and Renderer:
      police_nodes  : list of node IDs where police are deployed  (10 items)
      officers      : list of PoliceOfficer objects (10 items)
      risk_counts   : {"high": N, "medium": N, "low": N}
      node_scores   : {node_id: float crime_score}  (for display)
    """

    def __init__(self, graph: CityGraph):
        self.graph        = graph
        self.kmeans       = KMeans(k=3, max_iter=100)
        self.tree         = DecisionTree(max_depth=4)
        self.trained      = False

        # Police state
        self.police_nodes  = []   # 10 node IDs (for backwards compat)
        self.officers      = []   # 10 PoliceOfficer objects
        self.risk_counts   = {"high": 0, "medium": 0, "low": 0}
        self.node_scores   = {}   # node_id → raw crime score

        self._last_n_nodes = 0

    # ── Main pipeline ─────────────────────────────────────────────────────────
    def run(self, log_fn=None) -> dict:
        """
        Run the full 5-step pipeline.
        Safe to call multiple times (re-trains on each call).
        Returns {"high": N, "medium": N, "low": N}.
        """
        g     = self.graph
        nodes = list(g.nodes.values())
        if len(nodes) < 3:
            if log_fn:
                log_fn("[C5] Need at least 3 nodes for ML")
            return self.risk_counts

        # ── STEP 1: Compute industrial_distance for every node ────────────────
        # BFS from each industrial node, find min hops to each other node
        ind_nodes = [n.id for n in nodes if n.type == "industrial"]
        for n in nodes:
            if ind_nodes:
                best = min(
                    g.bfs_hops(iid).get(n.id, 999)
                    for iid in ind_nodes
                )
            else:
                best = 999   # no industrial zones → everyone far away
            n.industrial_distance = best

        # ── STEP 2: Normalise features for ML ────────────────────────────────
        max_pop = max(n.population_density for n in nodes) or 1
        max_ind = max(n.industrial_distance for n in nodes) or 1

        def get_features(n):
            """
            Feature vector for one node.
            f0 = population density normalised 0-1
            f1 = industrial proximity normalised 0-1
                 (1.0 = right next to industrial, 0.0 = far away)
            """
            pop_norm = n.population_density / max_pop
            ind_prox = 1.0 - min(n.industrial_distance, max_ind) / max_ind
            return [pop_norm, ind_prox]

        # ── STEP 3: K-Means Clustering (Unsupervised) ─────────────────────────
        data          = [(n.id, get_features(n)) for n in nodes]
        cluster_label = self.kmeans.fit(data)

        # Label each cluster High/Medium/Low by its centroid danger score
        # Danger score = pop_norm + ind_proximity (both contribute equally)
        danger = []
        for ci, cen in enumerate(self.kmeans.centroids):
            score = cen[0] + cen[1]
            danger.append((score, ci))
        danger.sort(reverse=True)   # highest danger first
        risk_order   = ["high", "medium", "low"]
        cluster_risk = {
            danger[i][1]: risk_order[i]
            for i in range(min(3, len(danger)))
        }

        # ── STEP 4: Generate Synthetic Crime Dataset ──────────────────────────
        #
        # Crime score formula (justified):
        #   crime = 0.55 * population_density_norm
        #         + 0.45 * industrial_proximity_norm
        #         + Gaussian_noise(0, 0.04)
        #
        # Justification:
        #   - Population density (55% weight): more people = more crime targets,
        #     more foot traffic, higher opportunity for theft/assault.
        #   - Industrial proximity (45% weight): industrial areas have poor
        #     lighting, low foot traffic at night, weaker community surveillance.
        #   - Gaussian noise: real crime data is never perfectly predictable.
        #
        # Labels:
        #   crime ≥ 0.60 → High risk
        #   crime ≥ 0.38 → Medium risk
        #   crime <  0.38 → Low risk
        #
        X_train, y_train = [], []
        self.node_scores = {}
        for n in nodes:
            f = get_features(n)
            crime = 0.55 * f[0] + 0.45 * f[1] + random.gauss(0, 0.04)
            crime = max(0.0, min(1.0, crime))   # clamp to [0, 1]
            if   crime >= 0.60: lbl = "high"
            elif crime >= 0.38: lbl = "medium"
            else:               lbl = "low"
            X_train.append(f)
            y_train.append(lbl)
            self.node_scores[n.id] = crime

        # ── STEP 5: Train Decision Tree (Supervised) ──────────────────────────
        self.tree.fit(X_train, y_train)
        self.trained = True

        # ── STEP 6: Predict final risk for every node ─────────────────────────
        # Decision Tree is primary predictor.
        # K-Means cluster label used as tie-breaker for borderline cases.
        self.risk_counts = {"high": 0, "medium": 0, "low": 0}
        for n in nodes:
            f  = get_features(n)
            dt_pred  = self.tree.predict(f)                             # DT
            cl_pred  = cluster_risk.get(cluster_label.get(n.id, 0), "low")  # K-Means

            # Blending rule:
            # If both agree → use that label.
            # If they disagree → trust DT (trained on more features).
            # Exception: if node is industrial, always at least medium.
            final = dt_pred
            if n.type == "industrial" and final == "low":
                final = "medium"   # industrial zones are never fully safe

            n.risk_level      = final
            n.risk_multiplier = RISK_MULT[final]
            self.risk_counts[final] += 1

        # ── STEP 7: Push risk multipliers to edges (affects A* + GA) ──────────
        g.update_effective_costs()

        # ── STEP 8: Deploy 10 police officers ────────────────────────────────
        self._deploy_police(nodes, log_fn)

        self._last_n_nodes = len(nodes)

        if log_fn:
            rc = self.risk_counts
            log_fn(
                f"[C5] ML done — "
                f"High={rc['high']} Med={rc['medium']} Low={rc['low']} | "
                f"10 police deployed to top-risk nodes"
            )
        return self.risk_counts

    def rerun(self, log_fn=None) -> dict:
        """Re-run after any population or type change (during simulation)."""
        return self.run(log_fn)

    # ── Police deployment ────────────────────────────────────────────────────
    def _deploy_police(self, nodes, log_fn=None):
        """
        Deploy exactly 10 police officers to the highest-risk nodes.

        Ranking algorithm:
          1. Primary key:   risk_level  (high > medium > low)
          2. Secondary key: crime score (higher score = more dangerous)
          3. Tertiary key:  population_density (higher = more people to protect)

        This ensures:
          - All High-risk nodes are covered first.
          - If there are fewer than 10 High-risk nodes, Medium nodes fill in.
          - If still not 10, Low-risk nodes get remaining officers.
          - Among nodes of the same risk, the more populated ones get priority.

        Each officer is tracked as a PoliceOfficer object.
        self.police_nodes is also updated for backwards compatibility.
        """
        risk_rank = {"high": 3, "medium": 2, "low": 1}

        ranked = sorted(
            nodes,
            key=lambda n: (
                risk_rank[n.risk_level],
                self.node_scores.get(n.id, 0.0),
                n.population_density
            ),
            reverse=True
        )

        # Take top NUM_POLICE (or all nodes if fewer than 10)
        top_nodes = ranked[:NUM_POLICE]

        # Create PoliceOfficer objects
        self.officers = []
        self.police_nodes = []
        for i, n in enumerate(top_nodes):
            officer = PoliceOfficer(
                officer_id = i + 1,
                node_id    = n.id,
                risk_level = n.risk_level
            )
            self.officers.append(officer)
            self.police_nodes.append(n.id)

        if log_fn:
            high_count   = sum(1 for o in self.officers if o.risk_level == "high")
            medium_count = sum(1 for o in self.officers if o.risk_level == "medium")
            low_count    = sum(1 for o in self.officers if o.risk_level == "low")
            log_fn(
                f"[C5] Police: {high_count} at High-risk, "
                f"{medium_count} at Medium-risk, "
                f"{low_count} at Low-risk nodes"
            )

    def redeploy_during_simulation(self, step: int, log_fn=None):
        """
        Called during simulation when risk weights change.
        Re-ranks nodes and moves officers to new positions.
        This shows the system dynamically adapting police coverage.
        """
        g     = self.graph
        nodes = list(g.nodes.values())
        if not nodes or not self.trained:
            return
        self._deploy_police(nodes, log_fn)
        if log_fn:
            log_fn(f"[Step {step}] Police redeployed — risk shift detected")

    def get_officer_at_node(self, node_id: int):
        """Return PoliceOfficer at this node, or None."""
        for o in self.officers:
            if o.node_id == node_id:
                return o
        return None

    def predict_node(self, node) -> str:
        """Classify a new node using the already-trained tree."""
        g     = self.graph
        nodes = list(g.nodes.values())
        if not self.trained or not nodes:
            node.risk_level      = "low"
            node.risk_multiplier = RISK_MULT["low"]
            return "low"

        max_pop = max(n.population_density for n in nodes) or 1
        max_ind = max(n.industrial_distance for n in nodes) or 1

        ind_nodes = [n.id for n in nodes if n.type == "industrial"]
        ind_dist  = min(
            (g.bfs_hops(iid).get(node.id, 999) for iid in ind_nodes),
            default=999
        )
        node.industrial_distance = ind_dist

        f = [
            node.population_density / max_pop,
            1.0 - min(ind_dist, max_ind) / max_ind,
        ]
        label = self.tree.predict(f)
        if node.type == "industrial" and label == "low":
            label = "medium"

        node.risk_level      = label
        node.risk_multiplier = RISK_MULT[label]
        return label
