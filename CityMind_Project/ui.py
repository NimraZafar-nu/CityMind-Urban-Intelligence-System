"""
CityMind – UIManager (4 toggles, clean top bar, no popup on node click)
4 main view toggles:
  1. Road Network  – show/hide MST edges
  2. Ambulance     – show/hide coverage circles + sprites
  3. Heatmap       – show/hide risk overlay
  4. Clusters      – show/hide K-Means cluster colours + panel
"""
import pygame, math, random
from graph import CityGraph, TYPE_COLORS, TYPE_LABELS, TYPE_ICONS, TYPES
from csp     import CSPSolver, check_all as csp_check_all, _adj_ok
from mst     import MSTBuilder
from ga      import GeneticAmbulancePlacer
from routing import EmergencyRouter
from ml      import CrimeRiskML
from renderer import CityRenderer

PANEL_W = 280
LOG_H   = 130
TOP_H   = 52
SKY_H   = 140

DEFAULT_COUNTS = {
    "residential":5,"hospital":2,"school":2,
    "industrial":1,"powerplant":1,"depot":1,
}

# ─────────────────────────────────────────────────────────────────────────────
class Btn:
    def __init__(self,rect,label,color=(60,100,180),tc=(255,255,255),toggle=False):
        self.rect=pygame.Rect(rect); self.label=label
        self.color=color; self.tc=tc; self.toggle=toggle
        self.active=False; self.hover=False

    def draw(self,screen,font):
        col=self.color
        if self.active: col=tuple(min(255,c+70) for c in col)
        if self.hover:  col=tuple(min(255,c+30) for c in col)
        pygame.draw.rect(screen,col,self.rect,border_radius=8)
        pygame.draw.rect(screen,(255,255,255),self.rect,2,border_radius=8)
        s=font.render(self.label,True,self.tc)
        screen.blit(s,s.get_rect(center=self.rect.center))

    def hit(self,event):
        if event.type==pygame.MOUSEMOTION:
            self.hover=self.rect.collidepoint(event.pos)
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            if self.rect.collidepoint(event.pos):
                if self.toggle: self.active=not self.active
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
class UIManager:
    def __init__(self,screen,graph:CityGraph,renderer:CityRenderer):
        self.screen=screen; self.graph=graph; self.renderer=renderer
        self.csp=CSPSolver(graph); self.mst=MSTBuilder(graph)
        self.ga=GeneticAmbulancePlacer(graph)
        self.router=EmergencyRouter(graph); self.ml=CrimeRiskML(graph)

        self.sw,self.sh=screen.get_size()
        self.grid_cols=6; self.grid_rows=6
        self.type_counts=dict(DEFAULT_COUNTS)
        self.node_radius=24
        self.violations=[]; self.log=[]

        # Interaction
        self.selected_node=None; self.hovered_node=None; self.hovered_edge=None
        self.block_mode=False; self.civ_mode=False; self.team_mode=False
        self.ctx_menu=None; self.popup=None

        # 4 View toggles
        self.show_roads     = True   # Toggle 1: Road Network
        self.show_ambulance = True   # Toggle 2: Ambulance Coverage
        self.show_heatmap   = False  # Toggle 3: Crime Risk Heatmap
        self.show_clusters  = False  # Toggle 4: K-Means Clusters

        self.show_costs     = False  # Show cost labels on edges

        # Simulation
        self.sim_active=False; self.sim_step=0; self.sim_timer=0
        self.SIM_INTERVAL=42
        self.next_flood_step=random.randint(8,14)
        self.ml_done=False; self.ga_done=False

        # Team animation
        self.team_px=self.team_py=None
        self.team_from_px=self.team_from_py=None
        self.team_to_px=self.team_to_py=None
        self.team_anim_t=0.0

        # Guard
        self._rebuilding=False

        # ── 20-Step Auto-Simulation state ─────────────────────────────────────
        self._auto_sim_active   = False   # master flag
        self._auto_sim_phase    = 0       # 0=idle,1=csp,2=mst,3=ml,4=ga,5=civs,6=run
        self._auto_sim_timer    = 0       # frame counter between phases
        self._auto_sim_steps    = 0       # steps executed in phase-6
        self._auto_sim_delay    = 90      # frames between setup phases (~1.5 s @60fps)
        self._auto_step_interval= 52      # frames between movement steps

        # Notify
        self._notify_msg=""; self._notify_tick=0

        # Panel dynamic rects
        self._type_btns=[]; self._pop_rects=[]
        self._del_r=self._start_r=self._end_r=None

        # Cluster state
        self._cluster_labels={}; self._cluster_counts={}

        self._calc_rects()
        self._build_top_buttons()
        self.regenerate()

    # ── Layout ────────────────────────────────────────────────────────────────
    def _calc_rects(self):
        sw,sh=self.sw,self.sh
        self.sky_rect=pygame.Rect(0,TOP_H,sw-PANEL_W,SKY_H)
        city_top=TOP_H+SKY_H
        self.city_rect=(0,city_top,sw-PANEL_W,sh-city_top-LOG_H)
        self.panel_rect=pygame.Rect(sw-PANEL_W,TOP_H,PANEL_W,sh-TOP_H)
        self.log_rect=pygame.Rect(0,sh-LOG_H,sw-PANEL_W,LOG_H)
        self._update_radius()
        self.renderer.set_city_rect(self.city_rect)

    def _update_radius(self):
        ax,ay,aw,ah=self.city_rect
        cw=aw/max(1,self.grid_cols); ch=ah/max(1,self.grid_rows)
        self.node_radius=max(10,min(28,int(min(cw,ch)*0.34)))

    # ── Top buttons ──────────────────────────────────────────────────────────
    def _build_top_buttons(self):
        pad,bh=5,34; by=(TOP_H-bh)//2
        defs=[
            ("Night",      (48,48,108),  True,  55),
            ("Grid M×N",   (65,88,158),  False, 76),
            ("Counts",     (55,128,75),  False, 65),
            ("BlockRoad",  (158,75,55),  True,  78),
            ("FloodRoad",  (28,75,158),  False, 78),
            ("Regen",      (108,55,168), False, 58),
            ("Run ML",     (78,48,138),  False, 64),
            ("Run GA",     (128,78,28),  False, 62),
            # ── 4 VIEW TOGGLES ──────────────────────────────────────────────
            ("Roads",      (40,100,180), True,  58),
            ("Ambulance",  (30,140,100), True,  82),
            ("Heatmap",    (98,38,98),   True,  70),
            ("Clusters",   (88,38,138),  True,  72),
            # ── SIMULATION ──────────────────────────────────────────────────
            ("PlaceCiv",   (0,108,88),   True,  68),
            ("SetTeam",    (0,88,128),   True,  66),
            ("Simulate",   (128,28,28),  True,  68),
            ("Step",       (80,80,30),   False, 50),
            ("20-Step Sim",(128,32,168), True,  88),
        ]
        self.top_btns=[]
        x=pad
        for label,col,toggle,w in defs:
            self.top_btns.append(Btn((x,by,w,bh),label,col,toggle=toggle))
            x+=w+pad
        (self.btn_night,self.btn_grid,self.btn_counts,
         self.btn_block,self.btn_flood,self.btn_regen,
         self.btn_ml,self.btn_ga,
         self.btn_roads,self.btn_ambulance,self.btn_heatmap,self.btn_clusters,
         self.btn_civ,self.btn_team,self.btn_sim,self.btn_step,
         self.btn_auto_sim)=self.top_btns
        # Sync toggle states
        self.btn_roads.active=self.show_roads
        self.btn_ambulance.active=self.show_ambulance

    def _build_type_btns(self):
        self._type_btns=[]; pr=self.panel_rect
        bw=pr.w-20; bh=24; px=pr.x+10; py=pr.y+95
        for t in TYPES:
            b=Btn((px,py,bw,bh),f"{TYPE_ICONS[t]}  {TYPE_LABELS[t]}",
                  color=TYPE_COLORS[t],tc=(28,28,28))
            b._ntype=t; self._type_btns.append(b); py+=bh+3

    # ── Core pipeline ─────────────────────────────────────────────────────────
    def regenerate(self):
        try:
            self.violations=self.csp.solve(
                self.grid_cols,self.grid_rows,
                self.type_counts,self.city_rect,self._log)
            self.mst.build(self._log)
        except Exception as e:
            self._log(f"⚠ Regen: {e}"); self.violations=[]

        self.selected_node=None; self.hovered_node=None; self.block_mode=False
        for btn in [self.btn_block,self.btn_sim,self.btn_civ,self.btn_team]:
            if hasattr(self,btn.label.lower().replace(' ','_')+'_active',): pass
            btn.active=False
        self.router.reset(); self.ga.placements=[]
        self.ml_done=False; self.ga_done=False
        self.sim_active=False; self.sim_step=0; self.sim_timer=0
        self.team_px=self.team_py=None
        self.next_flood_step=random.randint(8,14)
        self._cluster_labels={}; self._cluster_counts={}
        self._sync_type_counts(); self._update_radius()
        self.renderer.set_city_rect(self.city_rect)
        self._build_type_btns()

    def _run_algos(self, pinned=None):
        if self._rebuilding: return
        self._rebuilding=True
        try:
            self.violations=self.csp.heal(pinned or set(),self._log)
            self._sync_type_counts()
            self.mst.rebuild_after_change(self._log)
            if self.ml_done:
                try: self.ml.rerun(self._log)
                except Exception as e: self._log(f"⚠ ML: {e}")
            if self.ga_done:
                try: self.ga.rerun(self._log)
                except Exception as e: self._log(f"⚠ GA: {e}")
            if self.router.active: self.router._recalc(self._log)
            if self.ml_done: self._refresh_clusters()
            self._build_type_btns()
        except Exception as e: self._log(f"⚠ algos: {e}")
        finally: self._rebuilding=False

    def _sync_type_counts(self):
        c={t:0 for t in TYPES}
        for n in self.graph.nodes.values():
            if n.type in c: c[n.type]+=1
        self.type_counts=c

    def _refresh_clusters(self):
        try:
            km=self.ml.kmeans
            if not km.labels: return
            self._cluster_labels={nid:ci for nid,ci in km.labels.items()
                                   if nid in self.graph.nodes}
            cnt={}
            for ci in self._cluster_labels.values(): cnt[ci]=cnt.get(ci,0)+1
            self._cluster_counts=cnt
        except Exception: self._cluster_labels={}; self._cluster_counts={}

    def _notify(self,msg,log_too=True):
        self._notify_msg=msg; self._notify_tick=175
        if log_too: self._log(msg)

    def _flood_event(self, target_edge=None):
        if self._rebuilding: return
        g=self.graph
        victim=target_edge
        if victim is None:
            opts=[e for e in g.edges.values()
                  if not e.blocked and (e.in_mst or e.redundant)]
            if not opts: self._log("⚠ No roads to flood!"); return
            victim=random.choice(opts)
        victim.blocked=True; victim.flooded=True
        na=g.nodes.get(victim.a); nb=g.nodes.get(victim.b)
        self._log(f"🌊 FLOOD: #{victim.a}({na.type[:3] if na else '?'})↔"
                  f"#{victim.b}({nb.type[:3] if nb else '?'})")
        self._rebuilding=True
        try:
            self.mst.rebuild_after_change(self._log)
            self.violations=self.csp.validate_current()
        finally: self._rebuilding=False
        if self.router.active:
            self.router.on_edge_blocked(victim,self.sim_step,self._log)

    def _log(self,msg):
        self.log.append(str(msg))
        if len(self.log)>60: self.log=self.log[-60:]

    # ── Pre-validation ────────────────────────────────────────────────────────
    def _validate_type_change(self,node,new_type):
        g=self.graph; cols=g.grid_cols; rows=g.grid_rows
        asgn_wo={(n.gx,n.gy):n.type for n in g.nodes.values() if n.id!=node.id}
        pos=(node.gx,node.gy)
        if new_type=="industrial":
            for nb in _nb4(node.gx,node.gy,cols,rows):
                t=asgn_wo.get(nb)
                if t in ("school","hospital"):
                    return False,f"Cannot place Industrial next to {TYPE_LABELS[t]} at {nb}"
        if new_type in ("school","hospital"):
            for nb in _nb4(node.gx,node.gy,cols,rows):
                if asgn_wo.get(nb)=="industrial":
                    return False,f"Cannot place {TYPE_LABELS[new_type]} next to Industrial"
        asgn_w=dict(asgn_wo); asgn_w[pos]=new_type
        pos_set=set(asgn_w)
        if new_type=="residential":
            hosp={p for p,t in asgn_w.items() if t=="hospital"}
            if not hosp: return False,"No hospital in city (residential needs hospital ≤3 hops)"
            best=min((_bfs_dist(pos_set,h,pos) for h in hosp),default=999)
            if best>3: return False,f"Nearest hospital is {best} hops (max 3)"
        if new_type!="hospital" and node.type=="hospital":
            hosp_rem={p for p,t in asgn_w.items() if t=="hospital"}
            for rpos in {p for p,t in asgn_w.items() if t=="residential"}:
                if not hosp_rem: return False,"Removing last hospital strands residents"
                best=min((_bfs_dist(pos_set,h,rpos) for h in hosp_rem),default=999)
                if best>3: return False,f"Removing hospital strands Residential at {rpos}"
        if new_type=="powerplant":
            ind={p for p,t in asgn_w.items() if t=="industrial"}
            if not ind: return False,"No industrial zone (powerplant needs industrial ≤2 hops)"
            best=min((_bfs_dist(pos_set,i,pos) for i in ind),default=999)
            if best>2: return False,f"Nearest industrial is {best} hops (max 2)"
        if new_type!="industrial" and node.type=="industrial":
            ind_rem={p for p,t in asgn_w.items() if t=="industrial"}
            for ppos in {p for p,t in asgn_w.items() if t=="powerplant"}:
                if not ind_rem: return False,"Removing last industrial disconnects PowerPlant"
                best=min((_bfs_dist(pos_set,i,ppos) for i in ind_rem),default=999)
                if best>2: return False,f"Removing industrial disconnects PowerPlant at {ppos}"
        return True,""

    # ── Events ────────────────────────────────────────────────────────────────
    def handle_event(self,event):
        try: self._handle_safe(event)
        except Exception as e: self._log(f"⚠ Event: {e}")

    def _handle_safe(self,event):
        if self.popup: self._popup_event(event); return
        if self.ctx_menu and self._ctx_event(event): return

        for btn in self.top_btns:
            if btn.hit(event): self._btn(btn)

        if event.type==pygame.MOUSEMOTION: self._hover(*event.pos)
        if event.type==pygame.MOUSEBUTTONDOWN:
            mx,my=event.pos
            ax,ay,aw,ah=self.city_rect
            in_city=ax<=mx<ax+aw and ay<=my<ay+ah
            in_panel=mx>=self.sw-PANEL_W
            if event.button==1:
                if in_city: self._city_click(mx,my)
                elif in_panel: self._panel_click(mx,my)
            elif event.button==3 and in_city:
                self._open_ctx(mx,my)

    def _btn(self,btn):
        if btn is self.btn_night:
            self.renderer.dark_mode=btn.active
            self._log("🌙 Night" if btn.active else "☀ Day")

        elif btn is self.btn_grid:
            self.popup={"type":"grid","cols":self.grid_cols,"rows":self.grid_rows}

        elif btn is self.btn_counts:
            self.popup={"type":"counts","counts":dict(self.type_counts)}

        elif btn is self.btn_block:
            self.block_mode=btn.active
            self._log("🚧 Block mode ON" if btn.active else "Block mode OFF")

        elif btn is self.btn_flood:
            self._flood_event()

        elif btn is self.btn_regen:
            self._log("🔄 Regenerating…"); self.regenerate()

        elif btn is self.btn_ml:
            self._log("[C5] Running ML…")
            try:
                self.ml.run(self._log); self.ml_done=True
                self._refresh_clusters()
                self._rebuilding=True
                try:
                    self.mst.rebuild_after_change(self._log)
                    # Auto-run GA after ML
                    self._log("[C3] Auto-running GA after ML…")
                    self.ga.run(self._log)
                    self.ga_done=bool(self.ga.placements)
                finally: self._rebuilding=False
            except Exception as e: self._log(f"⚠ ML: {e}")

        elif btn is self.btn_ga:
            self._log("[C3] Running GA…")
            try:
                self.ga.run(self._log); self.ga_done=bool(self.ga.placements)
            except Exception as e: self._log(f"⚠ GA: {e}")

        # ── 4 VIEW TOGGLES ────────────────────────────────────────────────────
        elif btn is self.btn_roads:
            self.show_roads=btn.active

        elif btn is self.btn_ambulance:
            self.show_ambulance=btn.active

        elif btn is self.btn_heatmap:
            self.show_heatmap=btn.active

        elif btn is self.btn_clusters:
            self.show_clusters=btn.active
            if btn.active and not self.ml_done:
                self._log("⚠ Run ML first for clusters"); btn.active=False; self.show_clusters=False

        # ── SIMULATION ────────────────────────────────────────────────────────
        elif btn is self.btn_civ:
            self.civ_mode=btn.active; self.team_mode=False; self.btn_team.active=False
            if btn.active: self._log("[C4] Click nodes to place civilians")

        elif btn is self.btn_team:
            self.team_mode=btn.active; self.civ_mode=False; self.btn_civ.active=False
            if btn.active: self._log("[C4] Click a node to set team START")

        elif btn is self.btn_sim:
            if btn.active:
                if not self.router.team_pos:
                    self._notify("⚠ Set team START first"); btn.active=False; return
                if not self.router.civilians:
                    self._notify("⚠ Place at least 1 civilian first"); btn.active=False; return
                ok=self.router.start_mission(self._log)
                if ok:
                    self.sim_active=True
                    self._log(f"▶ Simulation started — {len(self.router.civilians)} civilians")
                    n=self.graph.nodes.get(self.router.team_pos)
                    if n: self.team_px=n.px; self.team_py=n.py
                else:
                    btn.active=False; self.sim_active=False
            else:
                self.sim_active=False; self._log("[C4] Simulation paused")

        elif btn is self.btn_step:
            # Single step (works even when paused)
            if self.router.team_pos and self.router.civilians:
                if not self.router.active:
                    self.router.start_mission(self._log)
                self._do_sim_step()

        elif btn is self.btn_auto_sim:
            if btn.active:
                self._start_auto_sim()
            else:
                self._stop_auto_sim()

    # ── 20-Step Auto-Simulation ───────────────────────────────────────────────

    def _start_auto_sim(self):
        """Called when user clicks '20-Step Sim' button to ON."""
        self._log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self._log("[Step 0] 🚀 20-STEP AUTO SIMULATION starting…")
        # Validate / set defaults
        if self.grid_cols < 3 or self.grid_rows < 3:
            self.grid_cols = 6; self.grid_rows = 6
        cap = self.grid_cols * self.grid_rows
        total = sum(self.type_counts.values())
        if total < 4 or total > cap:
            self.type_counts = dict(DEFAULT_COUNTS)
        self._log(f"[Step 0] Grid:{self.grid_cols}×{self.grid_rows} | "
                  f"Res={self.type_counts.get('residential',5)} "
                  f"Hosp={self.type_counts.get('hospital',2)} "
                  f"Ind={self.type_counts.get('industrial',1)} "
                  f"PP={self.type_counts.get('powerplant',1)} "
                  f"Depot={self.type_counts.get('depot',1)}")
        # Stop any running sim
        self.sim_active = False; self.btn_sim.active = False
        self._auto_sim_active = True
        self._auto_sim_phase  = 1      # start at CSP phase
        self._auto_sim_timer  = 0
        self._auto_sim_steps  = 0
        self.show_heatmap     = True;  self.btn_heatmap.active  = True
        self.show_ambulance   = True;  self.btn_ambulance.active= True
        self.show_roads       = True;  self.btn_roads.active    = True

    def _stop_auto_sim(self):
        """Called when user toggles button OFF, or sim completes."""
        self._auto_sim_active = False
        self._auto_sim_phase  = 0
        self.btn_auto_sim.active = False
        self._log("⏹ 20-Step Auto-Sim stopped.")

    def _tick_auto_sim(self):
        """
        Called every draw frame while _auto_sim_active is True.
        Phases:
          1 → CSP regen
          2 → MST  (already done in regen, just log delay)
          3 → ML
          4 → GA
          5 → place civilians + team start + initial A*
          6 → step the simulation (20 steps or until complete)
        """
        self._auto_sim_timer += 1

        # ── Phase 1: CSP ──────────────────────────────────────────────────────
        if self._auto_sim_phase == 1:
            if self._auto_sim_timer < 5:
                return                      # one-frame delay so screen renders
            self._log("[Step 1] ⚙ Running CSP city layout…")
            try:
                self.violations = self.csp.solve(
                    self.grid_cols, self.grid_rows,
                    self.type_counts, self.city_rect, self._log)
                self.mst.build(self._log)
            except Exception as e:
                self._log(f"⚠ CSP failed: {e}"); self._stop_auto_sim(); return
            # Reset routing & algos
            self.router.reset(); self.ga.placements = []
            self.ml_done = False; self.ga_done = False
            self.sim_active = False; self.sim_step = 0; self.sim_timer = 0
            self.team_px = self.team_py = None
            self._cluster_labels = {}; self._cluster_counts = {}
            self._sync_type_counts(); self._update_radius()
            self.renderer.set_city_rect(self.city_rect)
            self._build_type_btns()
            nv = len(self.violations)
            self._log(f"[Step 1] ✓ CSP: {len(self.graph.nodes)} nodes placed — "
                      f"{'all constraints satisfied' if nv==0 else f'{nv} violation(s)'}")
            mc = getattr(self.mst, 'mst_cost', 0)
            p2 = getattr(self.mst, 'path2', None)
            self._log(f"[Step 2] ✓ MST: {len([e for e in self.graph.edges.values() if e.in_mst])} edges, "
                      f"cost={mc:.2f} | Independent paths: {'YES' if p2 else 'NO'}")
            self._auto_sim_phase = 3
            self._auto_sim_timer = 0

        # ── Phase 3: ML ───────────────────────────────────────────────────────
        elif self._auto_sim_phase == 3:
            if self._auto_sim_timer < self._auto_sim_delay:
                return
            self._log("[Step 3] ⚙ Running ML (K-Means + Decision Tree)…")
            try:
                self.ml.run(self._log)
                self.ml_done = True
                self._refresh_clusters()
                self._rebuilding = True
                try:
                    self.mst.rebuild_after_change(self._log)
                finally:
                    self._rebuilding = False
                rc = self.ml.risk_counts
                self._log(f"[Step 3] ✓ K-Means: 3 clusters formed")
                self._log(f"[Step 3] ✓ Decision Tree trained on synthetic crime data")
                self._log(f"[Step 3] ✓ Risk: High={rc.get('high',0)} "
                          f"Med={rc.get('medium',0)} Low={rc.get('low',0)} "
                          f"| Police: {len(self.ml.officers)} deployed")
            except Exception as e:
                self._log(f"⚠ ML error: {e}")
            self._auto_sim_phase = 4
            self._auto_sim_timer = 0

        # ── Phase 4: GA ───────────────────────────────────────────────────────
        elif self._auto_sim_phase == 4:
            if self._auto_sim_timer < self._auto_sim_delay:
                return
            self._log("[Step 4] ⚙ Running Genetic Algorithm for ambulances…")
            try:
                self.ga.run(self._log)
                self.ga_done = bool(self.ga.placements)
                wd = getattr(self.ga, 'worst_dist', 0)
                pts = [f"#{p}" for p in self.ga.placements]
                self._log(f"[Step 4] ✓ GA: {len(self.ga.placements)} ambulances at depots "
                          f"{', '.join(pts)} | worst dist={wd:.2f}")
            except Exception as e:
                self._log(f"⚠ GA error: {e}")
            self._auto_sim_phase = 5
            self._auto_sim_timer = 0

        # ── Phase 5: place civilians + team start ─────────────────────────────
        elif self._auto_sim_phase == 5:
            if self._auto_sim_timer < self._auto_sim_delay:
                return
            g = self.graph
            # Pick residential nodes for civilians (3-5)
            res_nodes = [n for n in g.nodes.values() if n.type == "residential"]
            random.shuffle(res_nodes)
            num_civs = min(len(res_nodes), random.randint(3, 5))
            if num_civs == 0:
                # Fall back to any non-depot node
                res_nodes = [n for n in g.nodes.values() if n.type != "depot"]
                random.shuffle(res_nodes)
                num_civs = min(len(res_nodes), 3)
            civ_nodes = res_nodes[:num_civs]
            # Pick a depot node for team start (prefer depot, else any node)
            depot_nodes = [n for n in g.nodes.values() if n.type == "depot"]
            if not depot_nodes:
                depot_nodes = [n for n in g.nodes.values()
                               if n not in civ_nodes]
            random.shuffle(depot_nodes)
            start_node = depot_nodes[0] if depot_nodes else None
            if not start_node:
                self._log("⚠ No valid start node — stopping auto-sim")
                self._stop_auto_sim(); return
            # Apply
            self.router.reset()
            g.start_node_id = start_node.id
            self.router.set_start(start_node.id, self._log)
            self.team_px = start_node.px; self.team_py = start_node.py
            for cn in civ_nodes:
                self.router.add_civilian(cn.id, self._log)
            civ_ids = [f"#{cn.id}" for cn in civ_nodes]
            self._log(f"[Step 5] ✓ Civilians: {', '.join(civ_ids)} | "
                      f"Team start: depot #{start_node.id}")
            # Start the mission
            ok = self.router.start_mission(self._log)
            if not ok:
                self._log("⚠ Mission failed to start — stopping"); self._stop_auto_sim(); return
            self.sim_active = False       # we drive stepping ourselves in phase 6
            self.sim_step   = 0
            self.sim_timer  = 0
            self.next_flood_step = random.randint(8, 14)
            self._auto_sim_phase = 6
            self._auto_sim_timer = 0
            self._auto_sim_steps = 0

        # ── Phase 6: run 20 simulation steps ─────────────────────────────────
        elif self._auto_sim_phase == 6:
            if self._auto_sim_timer < self._auto_step_interval:
                # Animate team position between steps
                if (self.team_to_px is not None and
                        self.team_from_px is not None):
                    t2 = min(1.0, self._auto_sim_timer /
                             max(1, self._auto_step_interval))
                    self.team_px = int(self.team_from_px +
                                       (self.team_to_px - self.team_from_px) * t2)
                    self.team_py = int(self.team_from_py +
                                       (self.team_to_py - self.team_from_py) * t2)
                return

            self._auto_sim_timer = 0
            self._auto_sim_steps += 1

            # Check if mission already complete
            if not self.router.active:
                self._finish_auto_sim(); return

            # Max 40 steps guard (20 movement steps + extras)
            if self._auto_sim_steps > 40:
                self._log(f"[Step {self.sim_step}] ⏱ Step limit reached")
                self._finish_auto_sim(); return

            # 8% random flood chance
            if random.random() < 0.08:
                try: self._flood_event()
                except Exception: pass

            # Do one simulation step
            old_pos = self.router.team_pos
            old_n   = self.graph.nodes.get(old_pos) if old_pos else None
            try:
                self.router.step(self._log)
            except Exception as e:
                self._log(f"⚠ step: {e}")
            new_pos = self.router.team_pos
            new_n   = self.graph.nodes.get(new_pos) if new_pos else None
            if old_n and new_n and old_pos != new_pos:
                self.team_from_px = old_n.px; self.team_from_py = old_n.py
                self.team_to_px   = new_n.px; self.team_to_py   = new_n.py
                self.team_px      = old_n.px; self.team_py       = old_n.py
            elif new_n:
                self.team_px = new_n.px; self.team_py = new_n.py
                self.team_from_px = self.team_to_px = None
            self.sim_step += 1

            # Police redeploy every 5 steps
            if self.ml_done and self.sim_step % 5 == 0:
                try: self.ml.redeploy_during_simulation(self.sim_step, self._log)
                except Exception: pass

            # Mission complete?
            if not self.router.active:
                self._finish_auto_sim()

    def _finish_auto_sim(self):
        """Called when all civilians rescued or step limit hit."""
        rescued  = len(self.router.visited)
        total    = len(self.router.civilians)
        mc       = getattr(self.mst, 'mst_cost', 0)
        rc       = self.ml.risk_counts if self.ml_done else {}
        self._log(f"[Step {self.sim_step}] 🎉 MISSION COMPLETE! "
                  f"{rescued}/{total} civilians rescued in {self.sim_step} steps")
        self._log(f"[Step {self.sim_step}] 📊 Final stats: Steps={self.sim_step}, "
                  f"MST cost={mc:.2f}, "
                  f"Risk H={rc.get('high',0)} M={rc.get('medium',0)} L={rc.get('low',0)}")
        self._log("━━━ 🏁 20-STEP SIMULATION COMPLETE ━━━")
        self._stop_auto_sim()

    def _hover(self,mx,my):
        if self.hovered_node: self.hovered_node.hover=False; self.hovered_node=None
        self.hovered_edge=None
        n,d=self.graph.nearest_node_px(mx,my,self.node_radius*2)
        if n: n.hover=True; self.hovered_node=n; return
        e,_=self.graph.nearest_edge_px(mx,my,18)
        if e: self.hovered_edge=e

    def _city_click(self,mx,my):
        g=self.graph
        if self.civ_mode:
            n,_=g.nearest_node_px(mx,my,self.node_radius*2)
            if n: self.router.add_civilian(n.id,self._log)
            return
        if self.team_mode:
            n,_=g.nearest_node_px(mx,my,self.node_radius*2)
            if n:
                g.start_node_id=n.id; self.router.set_start(n.id,self._log)
                self.team_px=n.px; self.team_py=n.py; self.team_from_px=self.team_to_px=None
            return
        if self.block_mode:
            e,_=g.nearest_edge_px(mx,my,28)
            if e:
                e.blocked=not e.blocked
                if not e.blocked and hasattr(e,'flooded'): e.flooded=False
                st="🔴 Blocked" if e.blocked else "✅ Unblocked"
                self._log(f"{st} road #{e.a}↔#{e.b}")
                self._rebuilding=True
                try:
                    self.mst.rebuild_after_change(self._log)
                    self.violations=self.csp.validate_current()
                finally: self._rebuilding=False
                if e.blocked: self.router.on_edge_blocked(e,self.sim_step,self._log)
                self._build_type_btns()
            return
        # Node selection
        n,_=g.nearest_node_px(mx,my,self.node_radius*1.8)
        if n:
            if self.selected_node and self.selected_node.id==n.id:
                self.selected_node.selected=False; self.selected_node=None
            else:
                if self.selected_node: self.selected_node.selected=False
                self.selected_node=n; n.selected=True
                self._log(f"Selected {TYPE_LABELS.get(n.type,'?')} #{n.id} ({n.gx},{n.gy})")
        else:
            if self.selected_node: self.selected_node.selected=False; self.selected_node=None

    def _panel_click(self,mx,my):
        n=self.selected_node
        if not n: return
        # Type buttons
        for btn in self._type_btns:
            if btn.rect.collidepoint(mx,my) and btn._ntype!=n.type:
                ok,reason=self._validate_type_change(n,btn._ntype)
                if not ok: self._notify(f"⚠ {reason}"); return
                old=n.type; n.type=btn._ntype
                self._log(f"✏ #{n.id}: {TYPE_LABELS[old]}→{TYPE_LABELS[n.type]}")
                self._sync_type_counts()
                self._run_algos(pinned={(n.gx,n.gy)}); return
        # Population
        for r,pv in self._pop_rects:
            if r.collidepoint(mx,my):
                n.population_density=pv
                lbl={30:"Low",65:"Medium",90:"High"}.get(pv,str(pv))
                self._log(f"Pop #{n.id}→{lbl}")
                # density → ML → edges → MST → GA chain
                if self.ml_done:
                    try:
                        self.ml.rerun(self._log)
                        self.mst.rebuild_after_change(self._log)
                        if self.ga_done: self.ga.rerun(self._log)
                        self._refresh_clusters()
                    except Exception as e: self._log(f"⚠ risk upd: {e}")
                return
        # START / REMOVE
        if self._start_r and self._start_r.collidepoint(mx,my):
            self.graph.start_node_id=n.id
            self.router.set_start(n.id,self._log)
            nn=self.graph.nodes.get(n.id)
            if nn: self.team_px=nn.px; self.team_py=nn.py
        elif self._del_r and self._del_r.collidepoint(mx,my):
            self._remove_node(n.id)

    def _remove_node(self,nid):
        n=self.graph.nodes.get(nid)
        if not n: return
        self._log(f"🗑 Removed {TYPE_LABELS.get(n.type,'?')} #{nid}")
        if self.selected_node and self.selected_node.id==nid: self.selected_node=None
        self.router.remove_civilian(nid,self._log)
        if self.router.team_pos==nid:
            self.router.team_pos=None; self.team_px=self.team_py=None
        self.graph.remove_node(nid); self._sync_type_counts()
        self._run_algos(pinned=set())

    def _open_ctx(self,mx,my):
        g=self.graph
        n,_=g.nearest_node_px(mx,my,self.node_radius*2)
        e=None
        if not n:
            e,_=g.nearest_edge_px(mx,my,20)
        if n:
            items=[("✏ Select/Edit","select"),("▶ Set Team START","team_start"),
                   ("👤 Add Civilian","add_civ"),("🗑 Remove Node","remove")]
            self.ctx_menu=(mx,my,n,None,items)
        elif e:
            items=[("🌊 Flood Road","flood"),("🚧 Block/Unblock","block")]
            self.ctx_menu=(mx,my,None,e,items)

    def _ctx_event(self,event):
        if not self.ctx_menu: return False
        mx,my,node,edge,items=self.ctx_menu
        if event.type==pygame.MOUSEBUTTONDOWN:
            for i,(_,action) in enumerate(items):
                if pygame.Rect(mx,my+i*30,192,30).collidepoint(event.pos):
                    self._ctx_do(node,edge,action); self.ctx_menu=None; return True
            self.ctx_menu=None
        return False

    def _ctx_do(self,node,edge,action):
        if action=="select" and node:
            if self.selected_node: self.selected_node.selected=False
            self.selected_node=node; node.selected=True
        elif action=="team_start" and node:
            self.graph.start_node_id=node.id
            self.router.set_start(node.id,self._log)
            self.team_px=node.px; self.team_py=node.py
        elif action=="add_civ" and node:
            self.router.add_civilian(node.id,self._log)
        elif action=="remove" and node:
            self._remove_node(node.id)
        elif action=="flood" and edge:
            self._flood_event(target_edge=edge)
        elif action=="block" and edge:
            edge.blocked=not edge.blocked
            if not edge.blocked and hasattr(edge,'flooded'): edge.flooded=False
            self._log(f"{'🔴 Blocked' if edge.blocked else '✅ Unblocked'} #{edge.a}↔#{edge.b}")
            self._rebuilding=True
            try:
                self.mst.rebuild_after_change(self._log)
                self.violations=self.csp.validate_current()
            finally: self._rebuilding=False
            if edge.blocked: self.router.on_edge_blocked(edge,self.sim_step,self._log)

    # ── Popups ────────────────────────────────────────────────────────────────
    def _popup_event(self,event):
        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_ESCAPE: self.popup=None
            if event.key==pygame.K_RETURN: self._popup_apply()

    def _popup_apply(self):
        p=self.popup
        if not p: return
        if p["type"]=="grid":
            nc=max(3,min(16,p["cols"])); nr=max(3,min(14,p["rows"]))
            self.grid_cols,self.grid_rows=nc,nr
            cap=nc*nr
            while sum(self.type_counts.values())>cap:
                self.type_counts["residential"]=max(0,self.type_counts["residential"]-1)
            self._log(f"Grid → {nc}×{nr}"); self.popup=None; self.regenerate()
        elif p["type"]=="counts":
            new=p["counts"]; total=sum(new.values()); cap=self.grid_cols*self.grid_rows
            if total<4: self._log("⚠ Need ≥4 nodes"); return
            if total>cap: self._log(f"⚠ {total} > capacity {cap}"); return
            self.type_counts=dict(new); self._log(f"Counts → total {total}")
            self.popup=None; self.regenerate()

    # ── Simulation step ───────────────────────────────────────────────────────
    def _do_sim_step(self):
        old_pos=self.router.team_pos
        old_n=self.graph.nodes.get(old_pos) if old_pos else None
        try: self.router.step(self._log)
        except Exception as e: self._log(f"⚠ step: {e}")
        new_pos=self.router.team_pos
        new_n=self.graph.nodes.get(new_pos) if new_pos else None
        if old_n and new_n and old_pos!=new_pos:
            self.team_from_px=old_n.px; self.team_from_py=old_n.py
            self.team_to_px=new_n.px; self.team_to_py=new_n.py
            self.team_anim_t=0.0; self.team_px=old_n.px; self.team_py=old_n.py
        elif new_n:
            self.team_px=new_n.px; self.team_py=new_n.py
            self.team_from_px=self.team_to_px=None
        self.sim_step+=1
        # Auto-flood every 8-14 steps
        if self.sim_step>=self.next_flood_step:
            try: self._flood_event()
            except Exception: pass
            self.next_flood_step=self.sim_step+random.randint(8,14)
        # Police redeploy every 5 steps
        if self.ml_done and self.sim_step%5==0:
            try: self.ml.redeploy_during_simulation(self.sim_step,self._log)
            except Exception: pass
        # GA rerun every 10 steps
        if self.ga_done and self.ml_done and self.sim_step%10==0:
            try: self.ga.rerun(self._log)
            except Exception: pass

    # ── Main draw ─────────────────────────────────────────────────────────────
    def draw(self):
        # 20-Step Auto-Simulation tick (takes priority over manual sim)
        if self._auto_sim_active:
            self._tick_auto_sim()

        # Simulation auto-step (manual mode)
        if self.sim_active:
            if self.router.active:
                self.sim_timer+=1
                # Interpolate animation
                if self.team_to_px is not None and self.team_from_px is not None:
                    t2=min(1.0,self.sim_timer/max(1,self.SIM_INTERVAL))
                    self.team_px=int(self.team_from_px+(self.team_to_px-self.team_from_px)*t2)
                    self.team_py=int(self.team_from_py+(self.team_to_py-self.team_from_py)*t2)
                if self.sim_timer>=self.SIM_INTERVAL:
                    self.sim_timer=0
                    self._do_sim_step()
            else:
                self.sim_active=False; self.btn_sim.active=False

        try:
            self.screen.fill((8,8,18))
            rdr=self.renderer; p=rdr.palette()

            rdr.draw_sky_strip(self.sky_rect)
            rdr.draw_city_area(self.city_rect)

            # Toggle 3: Heatmap
            if self.show_heatmap and self.ml_done:
                rdr.draw_risk_heatmap()

            # Toggle 4: Clusters
            if self.show_clusters and self._cluster_labels:
                rdr.draw_cluster_view(self._cluster_labels,self.node_radius)

            # Toggle 1: Road Network
            if self.show_roads:
                rdr.draw_edges(self.city_rect, self.show_costs, self.hovered_edge)

            # Independent paths overlay
            if self.show_roads:
                p1=getattr(self.mst,'path1',None); p2=getattr(self.mst,'path2',None)
                if p1 or p2: rdr.draw_independent_paths(p1,p2)

            # Toggle 2: Ambulance coverage
            if self.show_ambulance and self.ga_done and self.ga.placements:
                rdr.draw_ambulance_coverage(self.ga.placements,self.node_radius)

            # A* path
            if self.router.current_path and len(self.router.current_path)>=2:
                rdr.draw_active_path(self.router.current_path,self.router.current_cost)

            rdr.draw_nodes(self.city_rect,self.node_radius)

            # Police
            if self.ml_done and self.ml.police_nodes:
                rdr.draw_police(self.ml.police_nodes,self.node_radius,
                                officers=self.ml.officers,node_scores=self.ml.node_scores)

            # Civilians
            if self.router.civilians:
                rdr.draw_civilians(self.router.civilians,self.router.visited,self.node_radius)

            # Team
            if self.router.team_pos:
                if self.team_px is not None:
                    rdr.draw_team_at_pixel(self.team_px,self.team_py,self.node_radius)
                else:
                    rdr.draw_team(self.router.team_pos,self.node_radius)

            rdr.draw_start_end_extended(self.node_radius)
            self._draw_topbar(p)
            self._draw_panel(p)
            self._draw_log(p)

            if self.ctx_menu:  self._draw_ctx(p)
            if self.popup:     self._draw_popup(p)
            if self._notify_tick>0:
                self._draw_notify(p); self._notify_tick-=1

            rdr.update_tick()
        except Exception as e:
            self._log(f"⚠ Render: {e}")

    # ── Drawing helpers ───────────────────────────────────────────────────────
    def _draw_notify(self,p):
        s=self.renderer.f16.render(self._notify_msg,True,(255,218,75))
        bg=pygame.Surface((s.get_width()+18,s.get_height()+10),pygame.SRCALPHA)
        bg.fill((18,8,8,min(222,self._notify_tick*3)))
        bx=(self.sw-bg.get_width())//2; by=self.sh-LOG_H-bg.get_height()-10
        self.screen.blit(bg,(bx,by)); self.screen.blit(s,(bx+9,by+5))

    def _draw_topbar(self,p):
        bar=pygame.Surface((self.sw,TOP_H),pygame.SRCALPHA)
        bar.fill((0,0,0,188)); self.screen.blit(bar,(0,0))
        for btn in self.top_btns: btn.draw(self.screen,self.renderer.f11)
        g=self.graph
        title=self.renderer.f22.render("🏙 CityMind",True,p["accent"])
        self.screen.blit(title,(self.sw-PANEL_W-title.get_width()-14,(TOP_H-title.get_height())//2))
        stats=(f"Grid:{g.grid_cols}×{g.grid_rows}  N:{len(g.nodes)}  E:{len(g.edges)}  Step:{self.sim_step}")
        st=self.renderer.f11.render(stats,True,p["text_dim"])
        self.screen.blit(st,(self.sw-PANEL_W-st.get_width()-14,(TOP_H+title.get_height())//2+2))

    def _draw_panel(self,p):
        pr=self.panel_rect; rdr=self.renderer
        surf=pygame.Surface((pr.w,pr.h),pygame.SRCALPHA)
        surf.fill((*[int(c) for c in p["panel"]],232))
        self.screen.blit(surf,pr.topleft)
        pygame.draw.rect(self.screen,p["accent"],pr,2)

        bw=pr.w-20; bh=25; px,py=pr.x+10,pr.y+10

        def sep():
            nonlocal py
            pygame.draw.line(self.screen,p["accent"],(px,py),(pr.right-10,py),1); py+=8

        if self.selected_node:
            n=self.selected_node
            col=TYPE_COLORS.get(n.type,(180,180,180))
            pygame.draw.circle(self.screen,col,(px+11,py+13),11)
            hdr=rdr.f16.render(f"#{n.id} · {TYPE_LABELS.get(n.type,'?')}",True,p["text"])
            self.screen.blit(hdr,(px+28,py)); py+=30
            info=rdr.f11.render(f"Grid ({n.gx},{n.gy})  Pop:{n.population_density}  Risk:{n.risk_level}",
                                  True,p["text_dim"])
            self.screen.blit(info,(px,py)); py+=17; sep()

            # Change Type
            self.screen.blit(rdr.f13.render("Change Type:",True,p["text"]),(px,py)); py+=20
            for btn in self._type_btns:
                btn.rect=pygame.Rect(px,py,bw,bh)
                btn.active=(btn._ntype==n.type)
                btn.draw(self.screen,rdr.f12)
                if btn.active:
                    chk=rdr.f12.render("✓",True,(255,255,255))
                    self.screen.blit(chk,(btn.rect.right-18,btn.rect.y+5))
                py+=bh+2
            sep()

            # Population
            self.screen.blit(rdr.f13.render("Population:",True,p["text"]),(px,py)); py+=20
            pop_opts=[("Low",30,(65,135,65)),("Med",65,(145,125,45)),("High",90,(175,55,55))]
            self._pop_rects=[]
            w3=(bw-8)//3
            for idx,(lbl,val,col) in enumerate(pop_opts):
                r=pygame.Rect(px+idx*(w3+4),py,w3,bh)
                active=abs(n.population_density-val)<20
                bcol=col if active else tuple(max(0,c-52) for c in col)
                pygame.draw.rect(self.screen,bcol,r,border_radius=5)
                pygame.draw.rect(self.screen,(255,255,255) if active else (75,75,75),r,2,border_radius=5)
                ls=rdr.f11.render(lbl+("✓" if active else ""),True,(255,255,255))
                self.screen.blit(ls,ls.get_rect(center=r.center))
                self._pop_rects.append((r,val))
            py+=bh+6; sep()

            # START / REMOVE
            half=(bw-4)//2
            self._start_r=pygame.Rect(px,py,half,bh)
            self._del_r=pygame.Rect(px+half+4,py,half,bh)
            self._end_r=None
            is_s=self.graph.start_node_id==n.id
            pygame.draw.rect(self.screen,(38,175,58) if is_s else (22,68,32),self._start_r,border_radius=5)
            pygame.draw.rect(self.screen,(158,25,25),self._del_r,border_radius=5)
            ss=rdr.f12.render("▶START"+("✓" if is_s else ""),True,(255,255,255))
            ds=rdr.f12.render("🗑 Remove",True,(255,215,215))
            self.screen.blit(ss,ss.get_rect(center=self._start_r.center))
            self.screen.blit(ds,ds.get_rect(center=self._del_r.center))

        else:
            # Status panel
            py=self._draw_status(px,py,bw,p,rdr)
            if self.show_clusters and self.ml_done and self._cluster_labels and hasattr(self.ml,'kmeans'):
                rdr.draw_cluster_legend(px,py,self.ml.kmeans.centroids,self._cluster_counts)
            else:
                rdr.draw_legend(px,py)

        self._draw_violations(p,pr)

    def _draw_status(self,px,py,bw,p,rdr):
        def line(txt,col=None):
            nonlocal py
            self.screen.blit(rdr.f11.render(str(txt)[:38],True,col or p["text"]),(px,py)); py+=15

        self.screen.blit(rdr.f14.render("System Status",True,p["accent"]),(px,py)); py+=22
        vc=len(self.violations)
        line(f"C1 CSP: {'✅ OK' if vc==0 else f'⚠ {vc} violation(s)'}",
             (95,215,95) if vc==0 else (255,95,95))
        mc=getattr(self.mst,'mst_cost',0)
        bc=sum(1 for e in self.graph.edges.values() if e.blocked)
        line(f"C2 MST cost: {mc:.2f}"+(f"  ({bc} blocked)" if bc else ""),(95,215,255))
        p2=getattr(self.mst,'path2',None)
        line(f"C2 H↔D backup: {'✅' if p2 else '⚠ none'}",
             (95,215,95) if p2 else (255,175,55))
        if self.ga_done and self.ga.placements:
            line(f"C3 Ambs: {len(self.ga.placements)} placed",(95,195,255))
            line(f"C3 Worst dist: {self.ga.worst_dist:.2f}",p["text"])
        else:
            line("C3: press Run GA or Run ML (auto)",p["text_dim"])
        r=self.router
        rescued=len(r.visited); total=len(r.civilians)
        line(f"C4: {rescued}/{total} civs rescued",
             (95,215,95) if rescued==total and total>0 else (215,195,55))
        if r.status: line(f"  {r.status[:36]}",p["text_dim"])
        if self.sim_active: line(f"SIM running — step {self.sim_step}",(255,155,55))
        elif self.sim_step>0: line(f"SIM done — {self.sim_step} steps",p["text_dim"])
        if self.ml_done:
            rc=self.ml.risk_counts
            line(f"C5 Risk: H={rc.get('high',0)} M={rc.get('medium',0)} L={rc.get('low',0)}",
                 (195,115,195))
            offs=self.ml.officers
            if offs:
                hc=sum(1 for o in offs if o.risk_level=="high")
                mc2=sum(1 for o in offs if o.risk_level=="medium")
                line(f"C5 Police: H={hc} M={mc2} deployed",(95,175,255))
        else:
            line("C5: press Run ML",p["text_dim"])
        py+=4
        # Toggle status summary
        self.screen.blit(rdr.f13.render("View Toggles:",True,p["accent"]),(px,py)); py+=18
        toggles=[
            ("🛣 Roads",      self.show_roads),
            ("🚑 Ambulance",  self.show_ambulance),
            ("🌡 Heatmap",    self.show_heatmap),
            ("🔵 Clusters",   self.show_clusters),
        ]
        for lbl,on in toggles:
            col=(95,215,95) if on else (130,140,160)
            self.screen.blit(rdr.f11.render(f"  {lbl}: {'ON' if on else 'off'}",True,col),(px,py)); py+=13
        py+=4; return py

    def _draw_violations(self,p,pr):
        viols=self.violations
        if not viols:
            vy=pr.bottom-36
            s=pygame.Surface((pr.w-4,32),pygame.SRCALPHA); s.fill((22,128,50,218))
            self.screen.blit(s,(pr.x+2,vy))
            pygame.draw.rect(self.screen,(55,215,95),(pr.x+2,vy,pr.w-4,32),1)
            t=self.renderer.f12.render("✅  All CSP Constraints OK",True,(195,255,195))
            self.screen.blit(t,t.get_rect(center=(pr.centerx,vy+16)))
        else:
            lh=14; h=24+min(len(viols),4)*lh+4; vy=pr.bottom-h-4
            s=pygame.Surface((pr.w-4,h),pygame.SRCALPHA); s.fill((155,22,22,218))
            self.screen.blit(s,(pr.x+2,vy))
            pygame.draw.rect(self.screen,(255,65,65),(pr.x+2,vy,pr.w-4,h),1)
            self.screen.blit(
                self.renderer.f12.render(f"⚠ {len(viols)} Violation(s)",True,(255,215,75)),
                (pr.x+6,vy+4))
            for i,v in enumerate(viols[:4]):
                if isinstance(v,dict):
                    c=v.get("constraint","?"); txt=f"{c}: {v.get('detail',str(v))}"
                    col=(255,125,125) if c=="C1" else (255,195,95) if c=="C2" else (175,195,255)
                else:
                    txt=str(v); col=(255,165,165)
                short=txt[:37]+".." if len(txt)>39 else txt
                self.screen.blit(self.renderer.f10.render(short,True,col),(pr.x+6,vy+22+i*lh))

    def _draw_log(self,p):
        lr=self.log_rect; rdr=self.renderer
        s=pygame.Surface((lr.w,lr.h),pygame.SRCALPHA); s.fill((0,0,0,172))
        self.screen.blit(s,lr.topleft)
        pygame.draw.rect(self.screen,p["accent"],lr,1)
        vc=len(self.violations)
        ttxt=(f"Event Log  |  ⚠ {vc} violation(s)" if vc else "Event Log  |  ✅ CSP OK")
        tcol=(255,155,55) if vc else p["accent"]
        self.screen.blit(rdr.f13.render(ttxt,True,tcol),(lr.x+7,lr.y+4))
        msgs=self.log[-9:]
        for i,msg in enumerate(msgs):
            if any(msg.startswith(x) for x in ("⚠","C1:","C2:","C3:")):   col=(255,125,75)
            elif msg.startswith("✓"):                                       col=(95,215,115)
            elif msg.startswith("↻"):                                       col=(125,195,255)
            elif msg.startswith("🌊"):                                      col=(95,175,255)
            elif msg.startswith("🚑"):                                      col=(95,255,145)
            elif msg.startswith("[Step"):                                    col=(215,215,175)
            elif i==len(msgs)-1:                                             col=p["text"]
            else:                                                            col=p["text_dim"]
            self.screen.blit(rdr.f11.render(msg[:80],True,col),(lr.x+7,lr.y+20+i*12))

    def _draw_ctx(self,p):
        mx,my,node,edge,items=self.ctx_menu
        iw,ih=194,30; mmx,mmy=pygame.mouse.get_pos(); rdr=self.renderer
        for i,(lbl,_) in enumerate(items):
            r=pygame.Rect(mx,my+i*ih,iw,ih)
            hov=r.collidepoint(mmx,mmy)
            col=(75,105,195) if hov else (32,42,82)
            if not rdr.dark_mode: col=(172,198,238) if hov else (212,220,242)
            pygame.draw.rect(self.screen,col,r)
            pygame.draw.rect(self.screen,p["accent"],r,1)
            self.screen.blit(rdr.f13.render(lbl,True,p["text"]),(r.x+8,r.y+7))

    def _draw_popup(self,p):
        sw,sh=self.sw,self.sh; rdr=self.renderer
        ov=pygame.Surface((sw,sh),pygame.SRCALPHA); ov.fill((0,0,0,148))
        self.screen.blit(ov,(0,0))
        pop=self.popup
        if pop["type"]=="grid":   self._draw_popup_grid(p,pop,rdr)
        elif pop["type"]=="counts": self._draw_popup_counts(p,pop,rdr)

    def _popup_box(self,bx,by,bw,bh,p):
        pygame.draw.rect(self.screen,p["panel"],(bx,by,bw,bh),border_radius=13)
        pygame.draw.rect(self.screen,p["accent"],(bx,by,bw,bh),2,border_radius=13)

    def _popup_ok_cancel(self,bx,by,bh,rdr,p):
        ok_r=pygame.Rect(bx+55,by+bh-35,130,27); can_r=pygame.Rect(bx+215,by+bh-35,130,27)
        pygame.draw.rect(self.screen,(50,135,50),ok_r,border_radius=7)
        pygame.draw.rect(self.screen,(125,38,38),can_r,border_radius=7)
        s1=rdr.f12.render("✔ Apply [Enter]",True,(255,255,255))
        s2=rdr.f12.render("✘ Cancel [Esc]",True,(255,255,255))
        self.screen.blit(s1,s1.get_rect(center=ok_r.center))
        self.screen.blit(s2,s2.get_rect(center=can_r.center))
        mb=pygame.mouse.get_pressed(); mmx,mmy=pygame.mouse.get_pos()
        if mb[0]:
            if ok_r.collidepoint(mmx,mmy):  self._popup_apply(); pygame.time.delay(148)
            if can_r.collidepoint(mmx,mmy): self.popup=None;     pygame.time.delay(148)

    def _draw_popup_grid(self,p,pop,rdr):
        sw,sh=self.sw,self.sh; bw,bh=420,228; bx,by=(sw-bw)//2,(sh-bh)//2
        self._popup_box(bx,by,bw,bh,p)
        self.screen.blit(rdr.f22.render("Grid Size  (M × N)",True,p["text"]),(bx+20,by+14))
        cap=pop["cols"]*pop["rows"]; total=sum(self.type_counts.values())
        warn=(f"⚠ Node count ({total}) > capacity ({cap})!" if total>cap
              else f"Capacity: {cap}  |  Current nodes: {total}")
        self.screen.blit(rdr.f12.render(warn,True,(255,115,55) if total>cap else p["text_dim"]),(bx+20,by+48))
        mb=pygame.mouse.get_pressed(); mmx,mmy=pygame.mouse.get_pos()
        def spinner(label,key,vmin,vmax,sx,sy,width=128):
            val=pop[key]
            mr=pygame.Rect(sx,sy,35,35); vr=pygame.Rect(sx+38,sy,width-76,35); pr2=pygame.Rect(sx+width-35,sy,35,35)
            self.screen.blit(rdr.f13.render(label,True,p["text"]),(sx,sy-19))
            for r2,sym,col in [(mr,"−",(175,55,55)),(pr2,"+",(55,145,55))]:
                pygame.draw.rect(self.screen,col,r2,border_radius=7)
                ss=rdr.f18.render(sym,True,(255,255,255))
                self.screen.blit(ss,ss.get_rect(center=r2.center))
                if mb[0] and r2.collidepoint(mmx,mmy):
                    pop[key]=max(vmin,min(vmax,val+(-1 if sym=="−" else 1))); pygame.time.delay(108)
            pygame.draw.rect(self.screen,p["panel2"],vr,border_radius=5)
            vs=rdr.f18.render(str(val),True,p["text"]); self.screen.blit(vs,vs.get_rect(center=vr.center))
        spinner("Columns (M)","cols",3,16,bx+38,by+108)
        spinner("Rows    (N)","rows",3,14,bx+228,by+108)
        self._popup_ok_cancel(bx,by,bh,rdr,p)

    def _draw_popup_counts(self,p,pop,rdr):
        sw,sh=self.sw,self.sh; bw,bh=498,378; bx,by=(sw-bw)//2,(sh-bh)//2
        self._popup_box(bx,by,bw,bh,p)
        self.screen.blit(rdr.f22.render("Node Type Counts",True,p["text"]),(bx+20,by+14))
        cap=self.grid_cols*self.grid_rows; total=sum(pop["counts"].values())
        ok=4<=total<=cap
        info=f"Total: {total}  |  Cap: {cap}"+(" ✅" if ok else f"  ← need 4–{cap}")
        self.screen.blit(rdr.f12.render(info,True,(95,215,115) if ok else (255,115,55)),(bx+20,by+48))
        self.screen.blit(rdr.f11.render("Min: 1 Hospital, 1 Depot, 1 Industrial  (CSP auto-enforces)",
                                         True,p["text_dim"]),(bx+20,by+66))
        mb=pygame.mouse.get_pressed(); mmx,mmy=pygame.mouse.get_pos(); cx=bx+20
        for i,t in enumerate(TYPES):
            row=by+95+i*44; col=TYPE_COLORS[t]
            pygame.draw.circle(self.screen,col,(cx+13,row+17),12)
            pygame.draw.circle(self.screen,tuple(max(0,c-38) for c in col),(cx+13,row+17),12,2)
            self.screen.blit(rdr.f12.render(f"{TYPE_ICONS[t]}  {TYPE_LABELS[t]}",True,p["text"]),(cx+30,row+7))
            val=pop["counts"].get(t,0)
            mr=pygame.Rect(cx+218,row+4,33,26); vr=pygame.Rect(cx+255,row+4,58,26); pr2=pygame.Rect(cx+318,row+4,33,26)
            maxv=max(1,cap-sum(pop["counts"].values())+val) if t!="residential" else cap
            hint={"industrial":"(min 1)","hospital":"(min 1)","depot":"(min 1)","powerplant":"(≤2h ind)"}.get(t,"")
            if hint: self.screen.blit(rdr.f10.render(hint,True,p["text_dim"]),(cx+358,row+9))
            for r2,sym,bcol in [(mr,"−",(155,50,50)),(pr2,"+",(48,135,50))]:
                pygame.draw.rect(self.screen,bcol,r2,border_radius=5)
                ss=rdr.f16.render(sym,True,(255,255,255)); self.screen.blit(ss,ss.get_rect(center=r2.center))
                if mb[0] and r2.collidepoint(mmx,mmy):
                    minv=1 if t in ("hospital","depot","industrial") else 0
                    pop["counts"][t]=max(minv,min(maxv,val+(-1 if sym=="−" else 1))); pygame.time.delay(108)
            pygame.draw.rect(self.screen,p["panel2"],vr,border_radius=5)
            vs=rdr.f16.render(str(val),True,p["text"]); self.screen.blit(vs,vs.get_rect(center=vr.center))
        self._popup_ok_cancel(bx,by,bh,rdr,p)


# ─────────────────────────────────────────────────────────────────────────────
# Module helpers
# ─────────────────────────────────────────────────────────────────────────────
def _nb4(gx,gy,cols,rows):
    for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
        nx,ny=gx+dx,gy+dy
        if 0<=nx<cols and 0<=ny<rows: yield (nx,ny)

def _bfs_dist(pos_set,start,target):
    if start==target: return 0
    from collections import deque
    dist={start:0}; q=deque([start])
    while q:
        cur=q.popleft()
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
            nb=(cur[0]+dx,cur[1]+dy)
            if nb in pos_set and nb not in dist:
                dist[nb]=dist[cur]+1
                if nb==target: return dist[nb]
                q.append(nb)
    return 999
