import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math

# --------------------- Data Structures ---------------------
@dataclass
class Node:
    id: int
    x: float
    y: float
    constraint: List[Optional[float]] = field(default_factory=lambda: [None, None, None])
    # constraint: [ux, uy, theta], None=free, 0=fixed, other=specified displacement
    skew_constraint: Optional[Tuple[float, float, float, str]] = None
    # (a, b, value, direction) -- direction: 'tangent' or 'normal'

@dataclass
class Element:
    id: int
    start: int
    end: int
    E: float
    A: float
    I: float

# ----------------- Element Stiffness and Assembly -------------------
def element_stiffness_2d_frame(E, A, I, L):
    k = np.zeros((6, 6))
    k[0,0] = k[3,3] = E*A/L
    k[0,3] = k[3,0] = -E*A/L
    k[1,1] = k[4,4] = 12*E*I/L**3
    k[1,4] = k[4,1] = -12*E*I/L**3
    k[1,2] = k[2,1] = 6*E*I/L**2
    k[1,5] = k[5,1] = 6*E*I/L**2
    k[4,2] = k[2,4] = -6*E*I/L**2
    k[4,5] = k[5,4] = -6*E*I/L**2
    k[2,2] = k[5,5] = 4*E*I/L
    k[2,5] = k[5,2] = 2*E*I/L
    return k

def transformation_matrix(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)
    c = dx / L
    s = dy / L
    T = np.zeros((6, 6))
    T[0,0] = T[3,3] = c
    T[0,1] = T[3,4] = s
    T[1,0] = T[4,3] = -s
    T[1,1] = T[4,4] = c
    T[2,2] = T[5,5] = 1
    return T

def assemble_global_stiffness(nodes, elements):
    ndof = len(nodes) * 3
    K = np.zeros((ndof, ndof))
    for elem in elements:
        n1, n2 = elem.start, elem.end
        node1, node2 = nodes[n1], nodes[n2]
        L = np.hypot(node2.x - node1.x, node2.y - node1.y)
        k_local = element_stiffness_2d_frame(elem.E, elem.A, elem.I, L)
        T = transformation_matrix(node1.x, node1.y, node2.x, node2.y)
        k_global = T.T @ k_local @ T
        dof_map = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
        for i in range(6):
            for j in range(6):
                K[dof_map[i], dof_map[j]] += k_global[i, j]
    return K

# ---------------- Boundary Conditions / Skewed Supports ----------------
def apply_boundary_conditions(K, F, nodes, penalty=1e20):
    ndof = len(nodes) * 3
    K_mod = K.copy()
    F_mod = F.copy()
    for i, node in enumerate(nodes):
        # Conventional constraints (ux, uy, theta)
        for j in range(3):
            if node.constraint[j] is not None:
                idx = i*3 + j
                K_mod[idx, :] = 0
                K_mod[:, idx] = 0
                K_mod[idx, idx] = 1
                F_mod[idx] = node.constraint[j]
        # Skewed supports
        if node.skew_constraint is not None:
            a, b, value, direction = node.skew_constraint
            if direction == 'tangent':
                # Skewed pin: a*ux + b*uy = value
                A, B = a, b
            elif direction == 'normal':
                # Skewed roller: -b*ux + a*uy = value
                A, B = -b, a
            else:
                raise ValueError('direction must be tangent or normal')
            idx_ux = i*3 + 0
            idx_uy = i*3 + 1
            K_mod[idx_ux, idx_ux] += penalty * A**2
            K_mod[idx_ux, idx_uy] += penalty * A * B
            K_mod[idx_uy, idx_ux] += penalty * B * A
            K_mod[idx_uy, idx_uy] += penalty * B**2
            F_mod[idx_ux] += penalty * A * value
            F_mod[idx_uy] += penalty * B * value
    return K_mod, F_mod

# ---------------------- Main Program ----------------------
def main():
    # Material and element properties
    E = 2e11      # Pa
    A = 0.003     # m^2
    I = 1.6e-5    # m^4

    # Node settings
    cos30 = math.cos(math.pi/6)
    sin30 = math.sin(math.pi/6)
    nodes = [
        Node(0, 0.0, 0.0, constraint=[0, 0, 0]),  # Node 0: fully fixed
        Node(1, 4.0, 0.0, skew_constraint=(cos30, sin30, 0, 'tangent')),  # Node 1: skewed pin support, 30 degrees
        # Node(1, 4.0, 0.0, skew_constraint=(cos30, sin30, 0, 'normal')),  # Node 1: skewed roller support, 30 degrees (uncomment to switch type)
        Node(2, 4.0, 3.0, constraint=[None, None, None]),  # Node 2: free
    ]

    # Elements (0-1, 1-2)
    elements = [
        Element(0, 0, 1, E, A, I),
        Element(1, 1, 2, E, A, I),
    ]

    ndof = len(nodes) * 3
    F = np.zeros(ndof)
    F[2*3+1] = -20e3  # Node 2: vertical force -20kN

    K = assemble_global_stiffness(nodes, elements)
    K_mod, F_mod = apply_boundary_conditions(K, F, nodes)
    U = np.linalg.solve(K_mod, F_mod)

    print("Nodal displacements (ux, uy, M):")
    for i in range(len(nodes)):
        print(f"Node {i}: ux={U[i*3]:.6e} m, uy={U[i*3+1]:.6e} m, M={U[i*3+2]:.6e} rad")

if __name__ == '__main__':
    main()
