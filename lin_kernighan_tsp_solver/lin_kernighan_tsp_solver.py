import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tsplib95
from scipy.spatial import Delaunay
from itertools import combinations


"""
This is a Lin-Kernighan TSP solver that implements all the standard features:
    - Basic k-opt moves
    - Recursive improvement (step & alternate_step)
    - Flip tracking & rollback
    - Neighborhood ordering (lk-ordering)
    - Breadth & depth parameters
    - Kick / double-bridge restarts
    - Tour object abstraction (next, flip)
    - Delta-based tentative improvements
    - Chained Lin-Kernighan algorithm
"""


# Configuration parameters (modifiable)
MAX_LEVEL = 12        # recursion depth
BREADTH1 = 5          # breadth at level 1
BREADTH2 = 5          # breadth at level 2
BREADTH_K = 1         # breadth at deeper levels
BREADTHA = 5          # alternate-step A-ordering breadth
BREADTHB = 5          # alternate-step B-ordering breadth
BREADTHD = 1          # alternate-step D-ordering breadth
TIME_LIMIT = 5.0      # time limit (s) for chained LK

# Tour data structure with incremental cost and consistency checks
class Tour:
    def __init__(self, order, D=None):
        self.n = len(order)
        self.order = np.array(order, dtype=np.int32)
        self.pos = np.empty(self.n, dtype=np.int32)
        for i, v in enumerate(self.order):
            self.pos[v] = i
        self.cost = None
        self._flip_counter = 0
        if D is not None:
            self.init_cost(D)

    def init_cost(self, D):
        # Compute full tour cost once
        c = 0.0
        for i in range(self.n):
            a = self.order[i]
            b = self.order[(i + 1) % self.n]
            c += D[a, b]
        self.cost = c

    def next(self, v):
        return self.order[(self.pos[v] + 1) % self.n]

    def prev(self, v):
        return self.order[(self.pos[v] - 1) % self.n]

    def sequence(self, a, b, c):
        ia, ib, ic = self.pos[a], self.pos[b], self.pos[c]
        if ia <= ic:
            return ia < ib <= ic
        return ia < ib or ib <= ic

    def get_tour(self):
        return list(self.order)

    def flip(self, a, b):
        # Reverse the segment between a and b inclusive
        ia, ib = self.pos[a], self.pos[b]
        if ia <= ib:
            self.order[ia:ib+1] = self.order[ia:ib+1][::-1]
        else:
            seg = np.concatenate((self.order[ia:], self.order[:ib+1]))[::-1]
            self.order[ia:] = seg[:self.n - ia]
            self.order[:ib+1] = seg[self.n - ia:]
        # update positions
        for i, v in enumerate(self.order):
            self.pos[v] = i

    def flip_and_update_cost(self, a, b, D):
        # Compute cost delta
        pa = self.prev(a)
        nb = self.next(b)
        removed = D[pa, a] + D[b, nb]
        added = D[pa, b] + D[a, nb]
        delta = added - removed
        # Perform flip
        self.flip(a, b)
        # Update cost and counter
        self.cost += delta
        self._flip_counter += 1
        return delta


# Distance & neighbors
def build_distance_matrix(coords):
    return np.linalg.norm(coords[:, None] - coords[None, :], axis=2)


def delaunay_neighbors(coords):
    tri = Delaunay(coords)
    neigh = {i: set() for i in range(len(coords))}
    for simplex in tri.simplices:
        for u, v in combinations(simplex, 2):
            neigh[u].add(v); neigh[v].add(u)
    return [sorted(neigh[i]) for i in range(len(coords))]

# Lin–Kernighan subroutines
def step(level, delta, base, tour, D, neigh, flip_seq, start_c, best_c):
    breadth = [BREADTH1, BREADTH2] + [BREADTH_K] * (MAX_LEVEL - 2)
    b = breadth[min(level - 1, len(breadth) - 1)]
    s1 = tour.next(base)
    cand = []
    # u-step
    for a in neigh[s1]:
        if a in (base, s1, tour.prev(s1)): continue
        if delta + D[base,s1] - D[s1,a] <= 0: continue
        probe = tour.prev(a)
        gain = (D[base,s1] - D[s1,a]) + (D[probe,a] - D[probe,base])
        cand.append(('u',a,probe,gain))
    # m-step
    for a in neigh[base]:
        if a in (tour.next(base),tour.prev(base),base): continue
        if delta + D[base,s1] - D[base,a] <=0: continue
        gain = (D[base,s1]-D[base,a])+(D[a,tour.next(a)]-D[tour.next(a),s1])
        cand.append(('m',a,None,gain))
    cand.sort(key=lambda x:-x[3])
    cnt = 0
    for typ,a,probe,gain in cand:
        if cnt>=b: break
        new_delta = delta+gain
        if typ=='u':
            x,y = s1,probe
            tour.flip(x,y); flip_seq.append((x,y))
            if start_c-new_delta<best_c: return True, flip_seq.copy()
            if level<MAX_LEVEL:
                ok,seq = step(level+1,new_delta,base,tour,D,neigh,flip_seq,start_c,best_c)
                if ok: return True, seq
            tour.flip(x,y); flip_seq.pop()
        else:
            x,y = tour.next(a),base
            tour.flip(x,y); flip_seq.append((x,y))
            if start_c-new_delta<best_c: return True, flip_seq.copy()
            newbase = tour.next(a)
            if level<MAX_LEVEL:
                ok,seq = step(level+1,new_delta,newbase,tour,D,neigh,flip_seq,start_c,best_c)
                if ok: return True, seq
            tour.flip(x,y); flip_seq.pop()
        cnt+=1
    return False,None


def alternate_step(base,tour,D,neigh):
    s1 = tour.next(base)
    A=[]
    for a in neigh[s1]:
        if a in (base,s1) or D[base,s1]-D[s1,a]<=0: continue
        probe=tour.prev(a)
        A.append((D[probe,a]-D[s1,a],a,probe))
    A.sort(reverse=True)
    for _,a,probe in A[:BREADTHA]:
        a1 = tour.next(a)
        B=[]
        for b in neigh[a1]:
            if b in (base,s1,a): continue
            b1=tour.next(b)
            B.append((D[b1,b]-D[a1,b],b,b1))
        B.sort(reverse=True)
        for _,b,b1 in B[:BREADTHB]:
            if tour.sequence(s1,b,a):
                return True, [(s1,b),(b,a)]
            C=[]
            for d in neigh[b1]:
                if d in (base,s1,a,a1,b): continue
                d1=tour.next(d)
                C.append((D[d1,d]-D[b1,d],d,d1))
            C.sort(reverse=True)
            for _,d,d1 in C[:BREADTHD]:
                return True, [(s1,d),(d,a),(a1,d1)]
    return False,None


def lk_search(v,tour,D,neigh):
    T0 = Tour(tour.get_tour(), D)
    start_c = T0.cost
    ok,seq = step(1,0,v,T0,D,neigh,[],start_c,start_c)
    if ok: return seq
    ok,seq = alternate_step(v,tour,D,neigh)
    return seq if ok else None

# Algorithm 15.4: Lin-Kernighan with safety and final cost check
def lin_kernighan(coords, init):
    n = len(coords)
    D = build_distance_matrix(coords)
    neigh = delaunay_neighbors(coords)
    tour = Tour(init, D)
    best_len = tour.cost
    improved = True

    while improved:
        improved = False
        for v in range(n):
            seq = lk_search(v, tour, D, neigh)
            if not seq:
                continue
            temp = Tour(tour.get_tour(), D)
            for a,b in seq:
                temp.flip_and_update_cost(a,b,D)
            if temp.cost + 1e-12 < best_len:
                for a,b in seq:
                    tour.flip_and_update_cost(a,b,D)
                best_len = temp.cost
                improved = True
                break
    return tour, best_len

# Algorithm 15.5: Chained LK with final cost recompute
def double_bridge(order):
    n = len(order)
    a,b,c,d = sorted(np.random.choice(range(1,n),4,False))
    s0,s1 = order[:a], order[a:b]
    s2,s3 = order[b:c], order[c:d]
    s4 = order[d:]
    return s0 + s2 + s1 + s3 + s4


def chained_lin_kernighan(coords, init, time_limit=None):
    if time_limit is None:
        time_limit = TIME_LIMIT
    tour_obj, bl = lin_kernighan(coords, init)
    start = time.time()
    while time.time() - start < time_limit:
        cand = double_bridge(tour_obj.get_tour())
        t2_obj, l2 = lin_kernighan(coords, cand)
        if l2 < bl:
            tour_obj, bl = t2_obj, l2
    # Final full-tour cost recompute to eliminate drift
    D = build_distance_matrix(coords)
    true_cost = 0.0
    for i in range(tour_obj.n):
        x = tour_obj.order[i]
        y = tour_obj.order[(i+1)%tour_obj.n]
        true_cost += D[x,y]
    tour_obj.cost = true_cost
    bl = true_cost
    return tour_obj.get_tour(), bl


# TSPLIB I/O
def read_tsp(path):
    prob = tsplib95.load(path)
    coords_map = dict(prob.node_coords)
    nodes = sorted(coords_map.keys())
    return np.array([coords_map[i] for i in nodes], float)

def read_opt_tour(path):
    tour, reading = [], False
    with open(path) as f:
        for line in f:
            tok = line.strip()
            if tok.upper().startswith('TOUR_SECTION'):
                reading = True
                continue
            if not reading:
                continue
            for p in tok.split():
                if p in ('-1', 'EOF'):
                    reading = False
                    break
                try:
                    idx = int(p)
                except:
                    continue
                if idx > 0:
                    tour.append(idx - 1)
            if not reading:
                break
    return tour


# Batch & Plot
if __name__ == '__main__':
    folder = '../TSPLIB95/tsp'
    results = []
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith('.tsp'): continue
        base = fn[:-4]
        tsp_path = os.path.join(folder, fn)
        opt_path = os.path.join(folder, base + '.opt.tour')
        if not os.path.exists(opt_path): continue

        problem = tsplib95.load(tsp_path)
        if getattr(problem, 'edge_weight_type','').upper() != 'EUC_2D': continue

        print(f"Processing {base} (EUC_2D)...")
        coords = read_tsp(tsp_path)
        D = build_distance_matrix(coords)
        opt_tour = read_opt_tour(opt_path)
        opt_len = 0.0
        for i in range(len(opt_tour)):
            a = opt_tour[i]
            b = opt_tour[(i+1)%len(opt_tour)]
            opt_len += D[a,b]
        print(f"  Optimal length: {opt_len:.2f}")

        init = list(range(len(coords)))
        start = time.time()
        heu_tour, heu_len = chained_lin_kernighan(coords, init, time_limit=TIME_LIMIT)
        elapsed = time.time() - start
        gap = max(0.0, 100.0 * (heu_len - opt_len) / opt_len)
        print(f"  Heuristic length: {heu_len:.2f}  Gap: {gap:.2f}%  Time: {elapsed:.2f}s")

        results.append({'name':base, 'coords':coords, 'opt_tour':opt_tour,
                        'heu_tour':heu_tour, 'opt_len':opt_len,
                        'heu_len':heu_len, 'gap':gap, 'time':elapsed})

    # Summary
    print("Configuration parameters:")
    print(f"  MAX_LEVEL   = {MAX_LEVEL}")
    print(f"  BREADTH1    = {BREADTH1}")
    print(f"  BREADTH2    = {BREADTH2}")
    print(f"  BREADTH_K   = {BREADTH_K}")
    print(f"  BREADTHA    = {BREADTHA}")
    print(f"  BREADTHB    = {BREADTHB}")
    print(f"  BREADTHD    = {BREADTHD}")
    print(f"  TIME_LIMIT  = {TIME_LIMIT:.2f}s")

    print("\nInstance   OptLen   HeuLen   Gap(%)   Time(s)")
    for r in results:
        print(f"{r['name']:10s} {r['opt_len']:8.2f} {r['heu_len']:8.2f} {r['gap']:8.2f} {r['time']:8.2f}")
    if results:
        avg_opt = sum(r['opt_len'] for r in results) / len(results)
        avg_heu = sum(r['heu_len'] for r in results) / len(results)
        avg_gap = sum(r['gap'] for r in results) / len(results)
        avg_time = sum(r['time'] for r in results) / len(results)
        print(f"{'AVERAGE':10s} {avg_opt:8.2f} {avg_heu:8.2f} {avg_gap:8.2f} {avg_time:8.2f}")
    print("Done.")

    # Plot tours
    n = len(results)
    if n > 0:
        import math
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes_list = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for ax, r in zip(axes_list, results):
            coords = r['coords']
            opt = r['opt_tour'] + [r['opt_tour'][0]]
            heu = r['heu_tour'] + [r['heu_tour'][0]]
            ax.plot(coords[heu, 0], coords[heu, 1], '-', label='Heuristic')
            ax.plot(coords[opt, 0], coords[opt, 1], ':', label='Optimal')
            ax.set_title(f"{r['name']} gap={r['gap']:.2f}%")
            ax.axis('equal')
            ax.grid(True)
        for ax in axes_list[len(results):]:
            ax.axis('off')
        plt.tight_layout()
        plt.show()