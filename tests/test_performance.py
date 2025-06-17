import timeit

# Setup code that runs once before the timing loop
setup_code_old = """
from lin_kernighan_tsp_solver.lk_algorithm import Tour as OldTourClass # Assuming you want to test an 'old' version
from lin_kernighan_tsp_solver.lk_algorithm import build_distance_matrix
import numpy as np

# --- IMPORTANT: How to get the OLD Tour class? ---
# Option 1: If you have the old Tour class code available, define it directly here:
class OldTour:
    # Paste the ENTIRE old Tour class definition here, including its __init__ and the old flip method
    def __init__(self, initial_order_nodes, D):
        self.n = len(initial_order_nodes)
        self.order = np.array(initial_order_nodes, dtype=np.intc)
        self.pos = np.empty(np.max(initial_order_nodes) + 1 if initial_order_nodes else 0, dtype=np.intc)
        if self.n > 0:
            for i, node in enumerate(self.order):
                self.pos[node] = i
        self.cost = None # Will be calculated by init_cost if D is provided
        if D is not None and self.n > 0:
            self.init_cost(D)
        elif self.n == 0:
            self.cost = 0.0

    def init_cost(self, D):
        self.cost = 0.0
        if self.n > 1:
            for i in range(self.n):
                self.cost += D[self.order[i], self.order[(i + 1) % self.n]]
        elif self.n == 1:
            self.cost = 0.0 # Or some other defined cost for a single node tour

    def get_tour(self):
        if self.n == 0:
            return []
        start_node_val = 0
        if start_node_val not in self.order: # If 0 is not in tour, start with the smallest node
            start_node_val = self.order[0]

        start_idx = self.pos[start_node_val]
        return list(np.roll(self.order, -start_idx))

    def next(self, node_val):
        if self.n == 0:
            raise IndexError("next() called on an empty tour")
        if node_val not in self.pos or self.pos[node_val] >= self.n : # Check if node_val is valid and in current tour
             raise IndexError(f"Node {node_val} not in current tour or pos array too small")
        current_idx = self.pos[node_val]
        return self.order[(current_idx + 1) % self.n]

    def prev(self, node_val):
        if self.n == 0:
            raise IndexError("prev() called on an empty tour")
        if node_val not in self.pos or self.pos[node_val] >= self.n: # Check if node_val is valid and in current tour
             raise IndexError(f"Node {node_val} not in current tour or pos array too small")
        current_idx = self.pos[node_val]
        return self.order[(current_idx - 1 + self.n) % self.n]

    def sequence(self, node_a, node_b, node_c):
        if self.n == 0:
            return False
        # Check if nodes are in tour
        for node_val in [node_a, node_b, node_c]:
            if node_val not in self.pos or self.pos[node_val] >= len(self.order):
                # Node not in tour or pos array outdated/incorrect
                return False

        pos_a, pos_b, pos_c = self.pos[node_a], self.pos[node_b], self.pos[node_c]
        # Simplified: a->b->c or c->b->a (forward or backward)
        # This assumes a, b, c are distinct for a meaningful sequence.
        # Check a -> b -> c
        if (pos_a + 1) % self.n == pos_b and (pos_b + 1) % self.n == pos_c:
            return True
        # Check c -> b -> a (reverse sequence means b is between a and c in forward)
        if (pos_c + 1) % self.n == pos_b and (pos_b + 1) % self.n == pos_a:
            return True
        return False # Default to False if not a direct sequence

    # --- PASTE THE OLD FLIP METHOD HERE ---
    def flip(self, segment_start_node: int, segment_end_node: int) -> None:
        idx_a, idx_b = self.pos[segment_start_node], self.pos[segment_end_node]
        segment_indices_in_order = []
        current_idx = idx_a
        while True:
            segment_indices_in_order.append(current_idx)
            if current_idx == idx_b:
                break
            current_idx = (current_idx + 1) % self.n
        segment_nodes = self.order[segment_indices_in_order]
        reversed_segment_nodes = segment_nodes[::-1]
        for i, order_idx in enumerate(segment_indices_in_order):
            node_val_to_place = reversed_segment_nodes[i]
            self.order[order_idx] = node_val_to_place
            self.pos[node_val_to_place] = order_idx
    # --- END OF OLD FLIP METHOD ---

    def flip_and_update_cost(self, node_a, node_b, D):
        # ... (rest of the old Tour class if needed for flip) ...
        prev_node_of_a = self.prev(node_a)
        next_node_of_b = self.next(node_b)

        delta_cost = (D[prev_node_of_a, node_b] + D[node_a, next_node_of_b]) - \
                     (D[prev_node_of_a, node_a] + D[node_b, next_node_of_b])

        self.flip(node_a, node_b)

        if self.cost is not None:
            self.cost += delta_cost
        else:
            # If cost was None, it means it wasn't initialized.
            # Recalculate from scratch after the flip.
            self.init_cost(D)
        return delta_cost


coords = np.array([[0,0],[1,0],[1,1],[0,1],[0.5,0.5]])
dist_matrix = build_distance_matrix(coords)
tour_obj_old = OldTour([0,1,2,3,4], dist_matrix) # Use the OldTour class defined above
"""
# Code snippet to time (old version)
stmt_code_old = "tour_obj_old.flip(1, 3)"  # Assuming old flip is available on tour_obj_old

setup_code_new = """
from lin_kernighan_tsp_solver.lk_algorithm import Tour, build_distance_matrix
import numpy as np
coords = np.array([[0,0],[1,0],[1,1],[0,1],[0.5,0.5]])
dist_matrix = build_distance_matrix(coords)
tour_obj_new = Tour([0,1,2,3,4], dist_matrix)
"""
# Code snippet to time (new version)
stmt_code_new = "tour_obj_new.flip(1, 3)"

# Number of times to execute the statement in each loop
number_of_executions = 10000
# Number of times to repeat the timing loop
repeat_count = 5

time_old = timeit.repeat(stmt_code_old, setup=setup_code_old, number=number_of_executions, repeat=repeat_count)
time_new = timeit.repeat(stmt_code_new, setup=setup_code_new, number=number_of_executions, repeat=repeat_count)

print(f"Old flip average time: {min(time_old) / number_of_executions:.6e} seconds")  # min is often used to get best case
print(f"New flip average time: {min(time_new) / number_of_executions:.6e} seconds")
print(f"Improvement: {(min(time_old) - min(time_new)) / min(time_old) * 100:.2f}%")
