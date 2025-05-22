import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tsplib95
from scipy.spatial import Delaunay
from itertools import combinations

# --- Algorithm Parameters (from book) ---
MAX_LEVEL = 12        # recursion depth (maximum k)
BREADTH = [5, 5] + [1] * 20  # breadth(k): breadth(1)=5, breadth(2)=5, breadth(k)=1 for k>2
BREADTH_A = 5  # breadthA (alternate first step)
BREADTH_B = 5  # breadthB (alternate first step)
BREADTH_D = 1  # breadthD (alternate first step)
TIME_LIMIT = 20.0

# --- Tour Data Structure (per section 15.2) ---
class Tour:
    def __init__(self, order, D=None):
        self.n = len(order)
        self.order = np.array(order, dtype=np.int32)
        self.pos = np.empty(self.n, dtype=np.int32)
        for i, v in enumerate(self.order):
            self.pos[v] = i
        self.cost = None
        if D is not None:
            self.init_cost(D)
    def init_cost(self, D):
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
    def flip(self, a, b):
        ia, ib = self.pos[a], self.pos[b]
        indices = []
        i = ia
        while True:
            indices.append(i)
            if i == ib:
                break
            i = (i + 1) % self.n
        segment = [self.order[i] for i in indices][::-1]
        for i, idx in enumerate(indices):
            self.order[idx] = segment[i]
        for i, v in enumerate(self.order):
            self.pos[v] = i
    def get_tour(self):
        zero_pos = self.pos[0]
        if zero_pos == 0:
            return list(self.order)
        else:
            return list(np.concatenate((self.order[zero_pos:], self.order[:zero_pos])))
    def flip_and_update_cost(self, a, b, D):
        pa = self.prev(a)
        nb = self.next(b)
        removed = D[pa, a] + D[b, nb]
        added = D[pa, b] + D[a, nb]
        delta = added - removed
        self.flip(a, b)
        self.cost += delta
        return delta

# --- Utility routines ---
def build_distance_matrix(coords):
    return np.linalg.norm(coords[:, None] - coords[None, :], axis=2)

def delaunay_neighbors(coords):
    tri = Delaunay(coords)
    neigh = {i: set() for i in range(len(coords))}
    for simplex in tri.simplices:
        for u, v in combinations(simplex, 2):
            neigh[u].add(v)
            neigh[v].add(u)
    return [sorted(neigh[i]) for i in range(len(coords))]

# --- Algorithm 15.1: step(level, delta) ---
def step(level, delta, base, tour, D, neigh, flip_seq, start_cost, best_cost, deadline):
    if time.time() >= deadline:
        return False, None
    b = BREADTH[min(level - 1, len(BREADTH) - 1)]
    s1 = tour.next(base)
    candidates = []

    # Usual flip: flip(s1, probe)
    for a in neigh[s1]:
        if a in (base, s1, tour.prev(s1)): continue
        # probe = prev(a)
        probe = tour.prev(a)
        g = (D[base, s1] - D[s1, a]) + (D[probe, a] - D[probe, base])
        if delta + D[base, s1] - D[s1, a] > 0:
            candidates.append(('flip', a, probe, g))
    # Mak-Morton flip: flip(next(a), base)
    for a in neigh[base]:
        if a in (tour.next(base), tour.prev(base), base): continue
        g = (D[base, s1] - D[base, a]) + (D[a, tour.next(a)] - D[tour.next(a), s1])
        if delta + D[base, s1] - D[base, a] > 0:
            candidates.append(('makmorton', a, None, g))
    # Sort by gain descending
    candidates.sort(key=lambda x: -x[3])
    count = 0
    for typ, a, probe, g in candidates:
        if time.time() >= deadline or count >= b:
            break
        new_delta = delta + g
        if typ == 'flip':
            x, y = s1, probe
            tour.flip(x, y)
            flip_seq.append((x, y))
            if start_cost - new_delta < best_cost:
                return True, flip_seq.copy()
            if level < MAX_LEVEL:
                ok, seq = step(level + 1, new_delta, base, tour, D, neigh, flip_seq, start_cost, best_cost, deadline)
                if ok: return True, seq
            tour.flip(y, x)
            flip_seq.pop()
        else:
            # Mak-Morton: flip(next(a), base)
            x, y = tour.next(a), base
            tour.flip(x, y)
            flip_seq.append((x, y))
            if start_cost - new_delta < best_cost:
                return True, flip_seq.copy()
            new_base = tour.next(a)
            if level < MAX_LEVEL:
                ok, seq = step(level + 1, new_delta, new_base, tour, D, neigh, flip_seq, start_cost, best_cost, deadline)
                if ok: return True, seq
            tour.flip(y, x)
            flip_seq.pop()
        count += 1
    return False, None

# --- Algorithm 15.2: alternate_step(base, ...) ---
def alternate_step(base, tour, D, neigh, deadline):
    if time.time() >= deadline:
        return False, None
    s1 = tour.next(base)
    # A-ordering
    A = []
    for a in neigh[s1]:
        if a in (base, s1) or D[base, s1] - D[s1, a] <= 0: continue
        probe = tour.prev(a)
        A.append((D[probe, a] - D[s1, a], a, probe))
    A.sort(reverse=True)
    for _, a, probe in A[:BREADTH_A]:
        if time.time() >= deadline: return False, None
        a1 = tour.next(a)
        # B-ordering
        B = []
        for b in neigh[a1]:
            if b in (base, s1, a): continue
            b1 = tour.next(b)
            B.append((D[b1, b] - D[a1, b], b, b1))
        B.sort(reverse=True)
        for _, b, b1 in B[:BREADTH_B]:
            if time.time() >= deadline: return False, None
            if tour.sequence(s1, b, a):
                return True, [(s1, b), (b, a)]
            # D-ordering
            C = []
            for d in neigh[b1]:
                if d in (base, s1, a, a1, b): continue
                d1 = tour.next(d)
                C.append((D[d1, d] - D[b1, d], d, d1))
            C.sort(reverse=True)
            for _, d, d1 in C[:BREADTH_D]:
                if time.time() >= deadline: return False, None
                return True, [(s1, d), (d, a), (a1, d1)]
    return False, None

# --- Algorithm 15.3: lk_search(v, T) ---
def lk_search(v, tour, D, neigh, deadline):
    if time.time() >= deadline:
        return None
    temp_tour = Tour(tour.get_tour(), D)
    start_cost = temp_tour.cost
    ok, seq = step(1, 0, v, temp_tour, D, neigh, [], start_cost, start_cost, deadline)
    if ok:
        return seq
    if time.time() >= deadline:
        return None
    ok, seq = alternate_step(v, tour, D, neigh, deadline)
    return seq if ok else None

# --- Algorithm 15.4: lin_kernighan(T) ---
def lin_kernighan(coords, init, D, neigh, deadline):
    n = len(coords)
    tour = Tour(init, D)
    best_cost = tour.cost
    marked = set(range(n))
    while marked:
        v = marked.pop()
        seq = lk_search(v, tour, D, neigh, deadline)
        if not seq:
            continue
        temp = Tour(tour.get_tour(), D)
        for x, y in seq:
            temp.flip_and_update_cost(x, y, D)
        if temp.cost + 1e-12 < best_cost:
            for x, y in seq:
                tour.flip_and_update_cost(x, y, D)
                marked.add(x)
                marked.add(y)
            best_cost = tour.cost
            marked = set(range(n))  # re-mark all as per original LK
    return tour, best_cost

# --- Algorithm 15.5: chained_lin_kernighan(S) ---
def double_bridge(order):
    n = len(order)
    if n <= 4:
        return list(order)
    a, b, c, d = sorted(np.random.choice(range(1, n), 4, replace=False))
    s0, s1 = order[:a], order[a:b]
    s2, s3 = order[b:c], order[c:d]
    s4 = order[d:]
    return s0 + s2 + s1 + s3 + s4

def chained_lin_kernighan(coords, init, opt_len=None, time_limit=None):
    if time_limit is None:
        time_limit = TIME_LIMIT
    t_start = time.time()
    deadline = t_start + time_limit
    D = build_distance_matrix(coords)
    neigh = delaunay_neighbors(coords)
    tour_obj, best_cost = lin_kernighan(coords, init, D, neigh, deadline)
    while time.time() < deadline:
        cand = double_bridge(tour_obj.get_tour())
        t2_obj, l2 = lin_kernighan(coords, cand, D, neigh, deadline)
        if l2 < best_cost:
            tour_obj, best_cost = t2_obj, l2
            if opt_len is not None and abs(best_cost - opt_len) < 1e-8:
                # Found optimal tour, exit early!
                break
    # Final cost recompute
    true_cost = 0.0
    for i in range(tour_obj.n):
        x = tour_obj.order[i]
        y = tour_obj.order[(i + 1) % tour_obj.n]
        true_cost += D[x, y]
    tour_obj.cost = true_cost
    best_cost = true_cost
    return tour_obj.get_tour(), best_cost

# --- TSPLIB Utilities ---
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

# --- Batch & Plot ---
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
        if getattr(problem, 'edge_weight_type', '').upper() != 'EUC_2D': continue
        print(f"Processing {base} (EUC_2D)...")
        coords = read_tsp(tsp_path)
        D = build_distance_matrix(coords)
        opt_tour = read_opt_tour(opt_path)
        opt_len = 0.0
        for i in range(len(opt_tour)):
            a = opt_tour[i]
            b = opt_tour[(i + 1) % len(opt_tour)]
            opt_len += D[a, b]
        print(f"  Optimal length: {opt_len:.2f}")
        init = list(range(len(coords)))
        start = time.time()
        heu_tour, heu_len = chained_lin_kernighan(coords, init, opt_len=opt_len, time_limit=TIME_LIMIT)
        elapsed = time.time() - start
        gap = max(0.0, 100.0 * (heu_len - opt_len) / opt_len)
        print(f"  Heuristic length: {heu_len:.2f}  Gap: {gap:.2f}%  Time: {elapsed:.2f}s")
        results.append({'name': base, 'coords': coords, 'opt_tour': opt_tour,
                        'heu_tour': heu_tour, 'opt_len': opt_len,
                        'heu_len': heu_len, 'gap': gap, 'time': elapsed})

    print("Configuration parameters:")
    print(f"  MAX_LEVEL   = {MAX_LEVEL}")
    print(f"  BREADTH     = {BREADTH}")
    print(f"  BREADTH_A   = {BREADTH_A}")
    print(f"  BREADTH_B   = {BREADTH_B}")
    print(f"  BREADTH_D   = {BREADTH_D}")
    print(f"  TIME_LIMIT  = {TIME_LIMIT:.2f}s\n")

    print("Instance   OptLen   HeuLen   Gap(%)   Time(s)")
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
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
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