"""
CityMind – Renderer  (clean straight-line roads, 4 toggles, no popups)
"""
import pygame, math, random, os
from graph import CityGraph, TYPE_COLORS, TYPE_LABELS

# ── Cluster palette ───────────────────────────────────────────────────────────
CLUSTER_COLS  = [(255,70,70), (255,200,40), (60,220,100)]
CLUSTER_NAMES = ["High-Risk", "Medium-Risk", "Low-Risk"]


class CityRenderer:
    def __init__(self, screen, graph: CityGraph):
        self.screen    = screen
        self.graph     = graph
        self.dark_mode = False
        self.tick      = 0
        self._city_rect = (0, 192, 800, 480)
        self.stars  = [(random.randint(0,1920), random.randint(0,300),
                        random.random()) for _ in range(160)]
        self.clouds = [(random.randint(0,1920), random.randint(20,120),
                        random.uniform(0.2,0.6)) for _ in range(5)]
        self.sprites: dict[str, pygame.Surface] = {}
        self._bg_buildings = self._gen_bg_buildings()
        self._init_fonts()

    # ── Fonts ─────────────────────────────────────────────────────────────────
    def _init_fonts(self):
        pygame.font.init()
        self.f10 = pygame.font.SysFont("segoeui", 10)
        self.f11 = pygame.font.SysFont("segoeui", 11)
        self.f12 = pygame.font.SysFont("segoeui", 12)
        self.f13 = pygame.font.SysFont("segoeui", 13)
        self.f14 = pygame.font.SysFont("segoeui", 14, bold=True)
        self.f16 = pygame.font.SysFont("segoeui", 16, bold=True)
        self.f18 = pygame.font.SysFont("segoeui", 18, bold=True)
        self.f22 = pygame.font.SysFont("segoeui", 22, bold=True)
        self.f28 = pygame.font.SysFont("segoeui", 28, bold=True)

    def set_city_rect(self, r): self._city_rect = r

    # ── Sprites ───────────────────────────────────────────────────────────────
    def load_sprites(self, asset_dir):
        mapping = {
            "residential":"house.png","hospital":"hospital.png",
            "school":"school.png","industrial":"industry.png",
            "powerplant":"powerplant.png","depot":"depot.png",
            "civilian":"civilian.png","ambulance":"ambulance.png",
            "police":"police.png",
        }
        for key, fname in mapping.items():
            path = os.path.join(asset_dir, fname)
            if os.path.exists(path):
                try:
                    img = pygame.image.load(path).convert()
                    img.set_colorkey((0,0,0))
                    self.sprites[key] = img
                except Exception: pass

    # ── Palette ───────────────────────────────────────────────────────────────
    def palette(self):
        if self.dark_mode:
            return dict(
                sky_top=(10,10,35), sky_bot=(25,25,70),
                ground=(30,40,30), city_bg=(22,30,48),
                city_grid=(35,45,62),
                road_mst=(70,160,255), road_red=(255,50,50),
                road_orange=(255,160,40), road_gray=(80,90,110),
                text=(220,230,255), text_dim=(130,145,175),
                panel=(16,20,38), panel2=(26,33,58),
                accent=(80,160,255),
                building=(38,48,78), building2=(52,62,95), win=(55,75,115),
            )
        else:
            return dict(
                sky_top=(80,160,255), sky_bot=(180,220,255),
                ground=(110,170,70), city_bg=(228,233,218),
                city_grid=(198,208,193),
                road_mst=(60,110,210), road_red=(210,35,35),
                road_orange=(210,130,20), road_gray=(150,160,170),
                text=(28,30,50), text_dim=(95,108,128),
                panel=(238,241,250), panel2=(218,223,240),
                accent=(55,115,225),
                building=(175,178,198), building2=(195,198,215), win=(135,175,215),
            )

    # ── Background buildings ──────────────────────────────────────────────────
    def _gen_bg_buildings(self):
        blds = []
        x = 0
        while x < 1920:
            w = random.randint(28,75)
            h = random.randint(35,170)
            blds.append((x,w,h))
            x += w + random.randint(2,8)
        return blds

    # ── Sky strip ─────────────────────────────────────────────────────────────
    def draw_sky_strip(self, sky_rect):
        p  = self.palette()
        sx, sy, sw, sh = sky_rect.x, sky_rect.y, sky_rect.w, sky_rect.h
        surf = pygame.Surface((sw, sh))
        band = max(1, sh//24)
        for i in range(0, sh, band):
            t  = i/max(sh-1,1)
            cr = int(p["sky_top"][0]+(p["sky_bot"][0]-p["sky_top"][0])*t)
            cg = int(p["sky_top"][1]+(p["sky_bot"][1]-p["sky_top"][1])*t)
            cb = int(p["sky_top"][2]+(p["sky_bot"][2]-p["sky_top"][2])*t)
            surf.fill((cr,cg,cb),(0,i,sw,band))
        self.screen.blit(surf,(sx,sy))
        self.screen.set_clip(sky_rect)
        if self.dark_mode:
            self._draw_stars(sx,sy,sw,sh)
            self._draw_moon(sx,sy,sw,sh)
        else:
            self._draw_clouds(sx,sy,sw,sh)
            self._draw_sun(sx,sy,sw,sh)
        self._draw_skyline(sx,sy,sw,sh,p)
        self.screen.set_clip(None)

    def _draw_stars(self,sx,sy,sw,sh):
        t=self.tick
        for ox,oy,ph in self.stars:
            a=int(130+80*math.sin(t*0.05+ph*6.28))
            pygame.draw.circle(self.screen,(a,a,min(255,a+20)),
                               (sx+int(ox%sw), sy+int(oy%sh)),1)

    def _draw_moon(self,sx,sy,sw,sh):
        cx,cy = sx+sw-90, sy+sh//2
        r=min(36,sh//3)
        for gr in range(45,0,-9):
            a=int(11*(45-gr)/45)
            pygame.draw.circle(self.screen,(100+a,100+a,55+a),(cx,cy),r+gr)
        pygame.draw.circle(self.screen,(238,238,198),(cx,cy),r)
        pygame.draw.circle(self.screen,(18,18,52),(cx+r//3,cy-3),r-5)
        for pos,rad in [((cx-r//3,cy+r//4),4),((cx-r//6,cy-r//3),3),((cx+r//4,cy+r//3),3)]:
            pygame.draw.circle(self.screen,(208,208,172),pos,rad)
        lbl=self.f10.render("NIGHT",True,(170,170,215))
        self.screen.blit(lbl,(cx-lbl.get_width()//2,cy+r+5))

    def _draw_sun(self,sx,sy,sw,sh):
        cx,cy = sx+sw-100, sy+sh//2
        r=min(32,sh//3)
        t=self.tick*0.02
        for i in range(12):
            ang=math.radians(i*30+t*40)
            r1,r2=r+7,r+18+int(5*math.sin(t+i))
            pygame.draw.line(self.screen,(255,215,55),
                (cx+int(math.cos(ang)*r1),cy+int(math.sin(ang)*r1)),
                (cx+int(math.cos(ang)*r2),cy+int(math.sin(ang)*r2)),2)
        for gr in range(24,0,-4):
            ratio=gr/24
            pygame.draw.circle(self.screen,(255,int(200*ratio+155*(1-ratio)),55),(cx,cy),r+gr)
        pygame.draw.circle(self.screen,(255,238,75),(cx,cy),r)
        pygame.draw.circle(self.screen,(255,255,170),(cx,cy),r-8)
        lbl=self.f10.render("DAY",True,(170,115,15))
        self.screen.blit(lbl,(cx-lbl.get_width()//2,cy+r+5))

    def _draw_clouds(self,sx,sy,sw,sh):
        for i,(ocx,_,spd) in enumerate(self.clouds):
            dx=sx+int((ocx+self.tick*spd*0.35)%(sw+180))-90
            dy=sy+sh//2-sh//4+i*(sh//max(1,len(self.clouds)))
            dy=max(sy+8,min(sy+sh-28,dy))
            for ddx,ddy,dr in [(-40,0,38),(0,-18,42),(40,0,38),(0,14,28)]:
                pygame.draw.circle(self.screen,(238,244,255),(dx+ddx,dy+ddy),dr)

    def _draw_skyline(self,sx,sy,sw,sh,p):
        gy=sy+sh-6
        pygame.draw.rect(self.screen,p["ground"],(sx,gy,sw,6))
        for bx_off,bw,bh in self._bg_buildings:
            bx=sx+bx_off%sw; by=gy-bh
            pygame.draw.rect(self.screen,p["building"],(bx,by,bw,bh))
            pygame.draw.rect(self.screen,p["building2"],(bx,by,bw,3))
            for wy in range(by+5,by+bh-4,11):
                for wx in range(bx+4,bx+bw-4,8):
                    wc=p["win"] if (not self.dark_mode or
                        math.sin(self.tick*0.03+wx*0.4+wy*0.3)>0.3) else p["building"]
                    pygame.draw.rect(self.screen,wc,(wx,wy,4,6))

    # ── City grid ─────────────────────────────────────────────────────────────
    def draw_city_area(self, city_rect):
        ax,ay,aw,ah = city_rect
        p=self.palette(); g=self.graph
        cols,rows=g.grid_cols,g.grid_rows
        pygame.draw.rect(self.screen,p["city_bg"],(ax,ay,aw,ah))
        cw,ch=aw/cols,ah/rows
        for gx in range(cols+1):
            x=int(ax+gx*cw)
            pygame.draw.line(self.screen,p["city_grid"],(x,ay),(x,ay+ah),1)
        for gy in range(rows+1):
            y=int(ay+gy*ch)
            pygame.draw.line(self.screen,p["city_grid"],(ax,y),(ax+aw,y),1)

    # ── EDGES – straight lines, ONLY MST + redundant + blocked ───────────────
    def draw_edges(self, city_rect, show_costs=False, hovered_edge=None):
        """
        Draw ONLY MST edges (in_mst=True), redundant edges, and blocked edges.
        Backup non-MST edges are intentionally HIDDEN to prevent congestion.
        Uses straight lines (not bezier) for clarity.
        """
        g=self.graph; p=self.palette()

        def cost_col(ec):
            if ec<=0.85: return (55,210,75)
            if ec<=1.05: return (70,150,255)
            if ec<=1.25: return (255,165,35)
            return (255,50,50)

        for e in g.edges.values():
            na=g.nodes.get(e.a); nb=g.nodes.get(e.b)
            if na is None or nb is None: continue

            x1,y1=na.px,na.py; x2,y2=nb.px,nb.py

            # Blocked edges always shown
            if e.blocked:
                pygame.draw.line(self.screen,(180,20,20),(x1,y1),(x2,y2),3)
                self._block_marker((x1+x2)//2,(y1+y2)//2,
                                   flooded=getattr(e,'flooded',False))
                continue

            # ONLY draw in_mst or redundant — skip backup edges
            if not e.in_mst and not e.redundant:
                continue

            if show_costs:
                col = cost_col(e.effective_cost)
            elif e.redundant:
                col = p["road_orange"]
            else:
                col = p["road_mst"]

            w = 4 if e.in_mst else 3

            # Shadow
            pygame.draw.line(self.screen,(0,0,0,60),(x1+1,y1+2),(x2+1,y2+2),w+1)

            if e.redundant:
                # dashed straight line
                dx,dy=x2-x1,y2-y1; L=max(1,math.hypot(dx,dy))
                step=12; t=0
                while t<1:
                    t2=min(1,t+step/L)
                    px1=int(x1+dx*t); py1=int(y1+dy*t)
                    px2=int(x1+dx*t2); py2=int(y1+dy*t2)
                    pygame.draw.line(self.screen,col,(px1,py1),(px2,py2),w)
                    t=t2+step/L
            else:
                pygame.draw.line(self.screen,col,(x1,y1),(x2,y2),w)
                # centre highlight
                hi=tuple(min(255,c+55) for c in col)
                pygame.draw.line(self.screen,hi,(x1,y1),(x2,y2),1)

            # Cost label on MST edges when show_costs ON
            if show_costs and e.in_mst:
                mx,my=(x1+x2)//2,(y1+y2)//2
                ts=self.f10.render(f"{e.effective_cost:.2f}",True,(255,255,255))
                bg=pygame.Surface((ts.get_width()+4,ts.get_height()+2),pygame.SRCALPHA)
                bg.fill((0,0,0,160))
                self.screen.blit(bg,(mx-ts.get_width()//2-2,my-8))
                self.screen.blit(ts,(mx-ts.get_width()//2,my-7))

        # Hover tooltip
        if hovered_edge and (hovered_edge.in_mst or hovered_edge.redundant or hovered_edge.blocked):
            self._edge_tooltip(hovered_edge,g,p)

    def _block_marker(self,mx,my,flooded=False):
        if flooded:
            t=self.tick
            pulse=int(2*math.sin(t*0.15))
            pygame.draw.circle(self.screen,(15,55,200),(mx,my),11+pulse)
            pygame.draw.circle(self.screen,(55,135,255),(mx,my),9+pulse)
            wc=(195,230,255)
            for off,base in [(-4,0),(2,0)]:
                pts=[(mx-6+s*2, my+off+int(2*math.sin(s*1.2+t*0.1+base)))
                     for s in range(7)]
                if len(pts)>1: pygame.draw.lines(self.screen,wc,False,pts,2)
            lbl=self.f10.render("FLOOD",True,(195,230,255))
            self.screen.blit(lbl,(mx-lbl.get_width()//2,my+13))
        else:
            pygame.draw.circle(self.screen,(215,25,25),(mx,my),11)
            pygame.draw.circle(self.screen,(255,75,75),(mx,my),9)
            pygame.draw.line(self.screen,(255,255,255),(mx-6,my-6),(mx+6,my+6),2)
            pygame.draw.line(self.screen,(255,255,255),(mx+6,my-6),(mx-6,my+6),2)

    def _edge_tooltip(self,e,g,p):
        na=g.nodes.get(e.a); nb=g.nodes.get(e.b)
        if not na or not nb: return
        mx,my=(na.px+nb.px)//2,(na.py+nb.py)//2-22
        lines=[
            f"Edge #{e.a}↔#{e.b}",
            f"Base: {e.base_cost:.2f}  Eff: {e.effective_cost:.2f}",
            "Residential (×0.8)" if e.base_cost<0.9 else "Standard (×1.0)",
            "MST Primary" if e.in_mst else ("Redundancy" if e.redundant else "Backup"),
        ]
        tw=max(self.f11.size(l)[0] for l in lines)+14
        th=len(lines)*14+8
        bx,by=mx-tw//2,my-th
        bg=pygame.Surface((tw,th),pygame.SRCALPHA); bg.fill((8,8,28,220))
        self.screen.blit(bg,(bx,by))
        pygame.draw.rect(self.screen,(90,155,255),(bx,by,tw,th),1,border_radius=3)
        for i,l in enumerate(lines):
            col=(200,255,200) if "MST" in l else \
                (255,220,100) if "Eff:" in l else (205,215,255)
            self.screen.blit(self.f11.render(l,True,col),(bx+7,by+4+i*14))

    # ── Nodes ─────────────────────────────────────────────────────────────────
    def draw_nodes(self, city_rect, node_radius):
        g=self.graph; p=self.palette(); t=self.tick
        for n in g.nodes.values():
            bc=TYPE_COLORS.get(n.type,(180,180,180))
            r=node_radius
            dr=int(r*1.1) if n.hover else r

            # Selected glow
            if n.selected:
                pulse=0.5+0.5*math.sin(t*0.1)
                gr=int(dr+7+5*pulse)
                gc=tuple(min(255,c+55) for c in bc)
                s=pygame.Surface((gr*2+2,gr*2+2),pygame.SRCALPHA)
                pygame.draw.circle(s,(*gc,80),(gr+1,gr+1),gr)
                self.screen.blit(s,(n.px-gr-1,n.py-gr-1))

            # Shadow
            pygame.draw.circle(self.screen,(0,0,0),(n.px+2,n.py+3),dr)
            # Body
            pygame.draw.circle(self.screen,bc,(n.px,n.py),dr)
            # Highlight
            hi=tuple(min(255,c+65) for c in bc)
            pygame.draw.circle(self.screen,hi,(n.px-dr//4,n.py-dr//4),dr//3)
            # Border
            bdr=(255,255,255) if n.selected else tuple(max(0,c-55) for c in bc)
            pygame.draw.circle(self.screen,bdr,(n.px,n.py),dr,3 if n.selected else 2)

            # Sprite
            spr=self.sprites.get(n.type)
            if spr and dr>=13:
                sz=int(dr*1.55)
                sc=pygame.transform.smoothscale(spr,(sz,sz))
                self.screen.blit(sc,(n.px-sz//2,n.py-sz//2))

            # Label
            if dr>=11:
                lbl=TYPE_LABELS.get(n.type,n.type)[:7]
                ls=self.f10.render(lbl,True,p["text"])
                lr=ls.get_rect(center=(n.px,n.py+dr+7))
                bg=pygame.Surface((lr.w+5,lr.h+1),pygame.SRCALPHA)
                bg.fill((0,0,0,75))
                self.screen.blit(bg,(lr.x-2,lr.y))
                self.screen.blit(ls,lr)

    # ── Risk Heatmap ──────────────────────────────────────────────────────────
    def draw_risk_heatmap(self):
        g=self.graph
        if not hasattr(self,"_city_rect"): return
        ax,ay,aw,ah=self._city_rect
        cw,ch=aw/g.grid_cols,ah/g.grid_rows
        COLS={"high":(220,35,35,65),"medium":(220,135,25,40),"low":(45,195,75,22)}
        for n in g.nodes.values():
            col=COLS.get(n.risk_level,(100,100,100,18))
            sx,sy=int(ax+n.gx*cw),int(ay+n.gy*ch)
            s=pygame.Surface((int(cw),int(ch)),pygame.SRCALPHA)
            s.fill(col)
            self.screen.blit(s,(sx,sy))

    # ── Cluster view ──────────────────────────────────────────────────────────
    def draw_cluster_view(self, cluster_labels: dict, node_radius: int):
        g=self.graph
        for n in g.nodes.values():
            ci=cluster_labels.get(n.id,0)
            col=CLUSTER_COLS[ci%len(CLUSTER_COLS)]
            r=node_radius+4
            s=pygame.Surface((r*2+4,r*2+4),pygame.SRCALPHA)
            pygame.draw.circle(s,(*col,100),(r+2,r+2),r)
            pygame.draw.circle(s,(*col,210),(r+2,r+2),r,3)
            self.screen.blit(s,(n.px-r-2,n.py-r-2))
            badge=self.f10.render(f"C{ci}",True,(255,255,255))
            bw,bh=badge.get_width()+5,badge.get_height()+2
            bg=pygame.Surface((bw,bh),pygame.SRCALPHA)
            bg.fill((*col,205))
            self.screen.blit(bg,(n.px-bw//2,n.py-node_radius-bh-3))
            self.screen.blit(badge,(n.px-bw//2+2,n.py-node_radius-bh-2))

    def draw_cluster_legend(self, x,y, centroids, counts):
        p=self.palette()
        self.screen.blit(self.f14.render("K-Means Clusters (C5)",True,p["text"]),(x,y)); y+=22
        for ci,col in enumerate(CLUSTER_COLS):
            pygame.draw.rect(self.screen,col,(x,y,11,11),border_radius=2)
            cnt=counts.get(ci,0)
            cen=centroids[ci] if ci<len(centroids) else [0,0]
            pop_n=cen[0] if cen else 0
            ind_n=cen[1] if len(cen)>1 else 0
            lbl=CLUSTER_NAMES[ci]
            self.screen.blit(self.f12.render(f"Cluster {ci} [{lbl}]: {cnt} nodes",
                                              True,col),(x+16,y))
            y+=15
            self.screen.blit(self.f11.render(f"  pop={pop_n:.2f}  ind_prox={ind_n:.2f}",
                                              True,p["text_dim"]),(x+16,y))
            y+=17
        return y

    # ── Ambulance coverage ────────────────────────────────────────────────────
    def draw_ambulance_coverage(self, placement, node_radius):
        g=self.graph; amb=self.sprites.get("ambulance")
        for nid in placement:
            n=g.nodes.get(nid)
            if not n: continue
            r=node_radius*3
            s=pygame.Surface((r*2+4,r*2+4),pygame.SRCALPHA)
            pygame.draw.circle(s,(75,155,255,22),(r+2,r+2),r)
            self.screen.blit(s,(n.px-r-2,n.py-r-2))
            s2=pygame.Surface((r*2+4,r*2+4),pygame.SRCALPHA)
            pygame.draw.circle(s2,(95,175,255,95),(r+2,r+2),r,2)
            self.screen.blit(s2,(n.px-r-2,n.py-r-2))
            sz=int(node_radius*1.75)
            if amb and sz>=10:
                sc=pygame.transform.smoothscale(amb,(sz,sz))
                self.screen.blit(sc,(n.px-sz//2,n.py-node_radius-sz-2))

    # ── Police ────────────────────────────────────────────────────────────────
    def draw_police(self, police_ids, node_radius,
                    officers=None, node_scores=None):
        g=self.graph; spr=self.sprites.get("police")
        risk_cols={"high":(255,55,55),"medium":(255,165,35),"low":(55,215,75)}
        om={} if not officers else {o.node_id:o for o in officers}
        for nid in police_ids:
            n=g.nodes.get(nid)
            if not n: continue
            o=om.get(nid); risk=o.risk_level if o else "low"
            rc=risk_cols.get(risk,(100,100,255))
            badge=o.officer_id if o else "?"
            pulse=0.6+0.4*math.sin(self.tick*0.12+(badge if isinstance(badge,int) else 0)*0.7)
            rp=int((node_radius+5)*pulse)
            rs=pygame.Surface((rp*2+4,rp*2+4),pygame.SRCALPHA)
            pygame.draw.circle(rs,(*rc,125),(rp+2,rp+2),rp,3)
            self.screen.blit(rs,(n.px-rp-2,n.py-rp-2))
            sz=int(node_radius*1.95); sz=max(15,sz)
            spy=n.py-node_radius-sz-3
            if spr and sz>=12:
                sc=pygame.transform.smoothscale(spr,(sz,sz))
                self.screen.blit(sc,(n.px-sz//2,spy))
            else:
                self._police_fallback(n.px,spy,sz,rc)
            bt=f"#{badge}"
            bgs=pygame.Surface((24,13),pygame.SRCALPHA); bgs.fill((*rc,195))
            br=pygame.Rect(n.px-12,spy-14,24,13)
            self.screen.blit(bgs,br.topleft)
            pygame.draw.rect(self.screen,(255,255,255),br,1,border_radius=2)
            bs=self.f10.render(bt,True,(255,255,255))
            self.screen.blit(bs,bs.get_rect(center=br.center))
            lbl=self.f10.render("POLICE",True,rc)
            self.screen.blit(lbl,(n.px-lbl.get_width()//2,n.py+node_radius+3))
            if node_scores and nid in node_scores:
                sc2=node_scores[nid]; bw=int(node_radius*1.75); bh=4
                bx=n.px-bw//2; by=n.py+node_radius+14
                pygame.draw.rect(self.screen,(55,55,55),(bx,by,bw,bh),border_radius=2)
                pygame.draw.rect(self.screen,rc,(bx,by,max(2,int(bw*sc2)),bh),border_radius=2)

    def _police_fallback(self,cx,cy,sz,col):
        r=max(4,sz//5)
        pygame.draw.circle(self.screen,(250,205,165),(cx,cy+r),r)
        pygame.draw.rect(self.screen,col,(cx-r,cy,r*2,r))
        pygame.draw.rect(self.screen,col,(cx-r+2,cy+r*2,r*2-4,r*2))
        pygame.draw.circle(self.screen,(255,215,0),(cx,cy+r*2+r//2),max(2,r//3))

    # ── Civilians ─────────────────────────────────────────────────────────────
    def draw_civilians(self, civilians, visited, node_radius):
        g=self.graph; spr=self.sprites.get("civilian")
        for nid in civilians:
            n=g.nodes.get(nid)
            if not n: continue
            done=nid in visited
            col=(100,100,100) if done else (255,215,0)
            pygame.draw.circle(self.screen,col,(n.px,n.py),node_radius+5,3)
            sz=int(node_radius*1.9)
            if spr and sz>=10:
                sc=pygame.transform.smoothscale(spr,(sz,sz))
                sc.set_alpha(55 if done else 230)
                self.screen.blit(sc,(n.px-sz//2,n.py-node_radius-sz-2))
            lbl=self.f10.render("DONE" if done else "CIV",True,col)
            self.screen.blit(lbl,(n.px-lbl.get_width()//2,n.py-node_radius-11))

    # ── Team ──────────────────────────────────────────────────────────────────
    def draw_team_at_pixel(self, px,py, node_radius):
        pulse=0.6+0.4*math.sin(self.tick*0.15)
        rp=int((node_radius+7)*pulse)
        pygame.draw.circle(self.screen,(45,215,45),(px,py),rp,3)
        amb=self.sprites.get("ambulance"); sz=int(node_radius*2.1)
        if amb and sz>=10:
            sc=pygame.transform.smoothscale(amb,(sz,sz))
            self.screen.blit(sc,(px-sz//2,py-rp-sz-2))
        lbl=self.f12.render("TEAM",True,(45,255,75))
        self.screen.blit(lbl,(px-lbl.get_width()//2,py+rp+3))

    def draw_team(self, nid, node_radius):
        n=self.graph.nodes.get(nid)
        if n: self.draw_team_at_pixel(n.px,n.py,node_radius)

    # ── Active A* path ────────────────────────────────────────────────────────
    def draw_active_path(self, path, cost, color=(0,225,175)):
        if not path or len(path)<2: return
        g=self.graph
        pulse=0.7+0.3*math.sin(self.tick*0.10)
        col=tuple(int(c*pulse) for c in color)
        hi=tuple(min(255,c+75) for c in col)
        for i in range(len(path)-1):
            na=g.nodes.get(path[i]); nb=g.nodes.get(path[i+1])
            if na and nb:
                pygame.draw.line(self.screen,col,(na.px,na.py),(nb.px,nb.py),6)
                pygame.draw.line(self.screen,hi,(na.px,na.py),(nb.px,nb.py),2)
        mn=g.nodes.get(path[len(path)//2])
        if mn:
            txt=self.f12.render(f"cost {cost:.2f}",True,(255,255,255))
            bg=pygame.Surface((txt.get_width()+8,txt.get_height()+4),pygame.SRCALPHA)
            bg.fill((0,0,0,155))
            self.screen.blit(bg,(mn.px-txt.get_width()//2-4,mn.py-24))
            self.screen.blit(txt,(mn.px-txt.get_width()//2,mn.py-22))

    # ── Independent H↔D paths ─────────────────────────────────────────────────
    def draw_independent_paths(self,path1,path2):
        if path1: self.draw_active_path(path1,0,color=(0,215,185))
        if path2:
            g=self.graph; col=(255,215,0)
            for i in range(len(path2)-1):
                na=g.nodes.get(path2[i]); nb=g.nodes.get(path2[i+1])
                if na and nb:
                    # dashed yellow
                    dx,dy=nb.px-na.px,nb.py-na.py; L=max(1,math.hypot(dx,dy))
                    step=10; t=0
                    while t<1:
                        t2=min(1,t+step/L)
                        pygame.draw.line(self.screen,col,
                            (int(na.px+dx*t),int(na.py+dy*t)),
                            (int(na.px+dx*t2),int(na.py+dy*t2)),3)
                        t=t2+step/L

    # ── Legend ────────────────────────────────────────────────────────────────
    def draw_legend(self, x, y):
        from graph import TYPE_ICONS
        p=self.palette()
        self.screen.blit(self.f14.render("Node Types",True,p["text"]),(x,y)); y+=22
        for ntype,col in TYPE_COLORS.items():
            pygame.draw.circle(self.screen,col,(x+8,y+8),8)
            pygame.draw.circle(self.screen,tuple(max(0,c-38) for c in col),(x+8,y+8),8,2)
            lbl=self.f12.render(TYPE_LABELS[ntype],True,p["text"])
            self.screen.blit(lbl,(x+21,y+2)); y+=20
        y+=6
        self.screen.blit(self.f14.render("Roads",True,p["text"]),(x,y)); y+=20
        for col,lbl in [(p["road_mst"],"MST (primary)"),(p["road_orange"],"Redundancy"),
                        (p["road_red"],"Blocked/Flooded")]:
            pygame.draw.line(self.screen,col,(x,y+7),(x+17,y+7),3)
            self.screen.blit(self.f12.render(lbl,True,p["text"]),(x+22,y+2)); y+=18
        return y

    def draw_start_end_extended(self, node_radius):
        g=self.graph
        for nid,col,lbl in [(g.start_node_id,(45,215,45),"START"),
                            (g.end_node_id,(255,75,75),"END")]:
            if nid and nid in g.nodes:
                n=g.nodes[nid]; rp=node_radius+6
                pulse=0.5+0.5*math.sin(self.tick*0.10)
                rp2=rp+int(4*pulse)
                pygame.draw.circle(self.screen,col,(n.px,n.py),rp2,4)
                ls=self.f12.render(lbl,True,col)
                self.screen.blit(ls,ls.get_rect(center=(n.px,n.py-rp2-9)))

    def update_tick(self): self.tick+=1
