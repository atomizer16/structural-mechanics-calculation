import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Node:
    id: int
    x: float
    y: float
    constraint: List[Optional[float]] = field(default_factory=lambda: [None, None, None])  # [ux, uy, theta], None=free, 0=fixed

@dataclass
class Element:
    id: int
    start: int
    end: int
    E: float
    A: float
    I: float
    hinge: Tuple[bool, bool] = (False, False)
    rigid: Tuple[float, float] = (0.0, 0.0)
    ignore_axial: bool = False

def beam_uniform_load_local(q, L):
    f_local = np.zeros(6)
    f_local[1] = q * L / 2
    f_local[2] = q * L ** 2 / 12
    f_local[4] = q * L / 2
    f_local[5] = -q * L ** 2 / 12
    return f_local

def beam_point_load_local(P, a, L):
    f_local = np.zeros(6)
    b = L - a
    f_local[1] = P * b ** 2 * (L + 2 * a) / L ** 3
    f_local[2] = -P * a * b ** 2 / L ** 2
    f_local[4] = P * a ** 2 * (3 * L - 2 * a) / L ** 3
    f_local[5] = P * a ** 2 * b / L ** 2
    return f_local

def assemble_element_loads(F, nodes, elements, element_loads):
    for idx, elem in enumerate(elements):
        n1, n2 = elem.start, elem.end
        node1, node2 = nodes[n1], nodes[n2]
        L = np.hypot(node2.x - node1.x, node2.y - node1.y)
        T = transformation_matrix(node1.x, node1.y, node2.x, node2.y)
        dof_map = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
        # --- uniform load ---
        if "uniform" in element_loads[idx] and element_loads[idx]["uniform"]:
            q = element_loads[idx]["uniform"]
            f_local_uni = beam_uniform_load_local(q, L)
            f_global_uni = T.T @ f_local_uni
            for i in range(6):
                F[dof_map[i]] += f_global_uni[i]
        # --- point loads ---
        if "points" in element_loads[idx]:
            for P, a in element_loads[idx]["points"]:
                f_local_p = beam_point_load_local(P, a, L)
                f_global_p = T.T @ f_local_p
                for i in range(6):
                    F[dof_map[i]] += f_global_p[i]
    return F

def element_stiffness_2d_frame(E, A, I, L, ignore_axial=False, start_hinge=False, end_hinge=False):
    k = np.zeros((6, 6))
    if not ignore_axial:
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

    if start_hinge:
        k[2,:] = 0
        k[:,2] = 0
        k[2,2] = 1e-10
    if end_hinge:
        k[5,:] = 0
        k[:,5] = 0
        k[5,5] = 1e-10

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
        n1 = elem.start
        n2 = elem.end
        node1 = nodes[n1]
        node2 = nodes[n2]
        L = np.hypot(node2.x - node1.x, node2.y - node1.y)
        k_local = element_stiffness_2d_frame(elem.E, elem.A, elem.I, L,
                                             ignore_axial=elem.ignore_axial,
                                             start_hinge=elem.hinge[0], end_hinge=elem.hinge[1])
        T = transformation_matrix(node1.x, node1.y, node2.x, node2.y)
        k_global = T.T @ k_local @ T

        dof_map = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
        for i in range(6):
            for j in range(6):
                K[dof_map[i], dof_map[j]] += k_global[i, j]
    return K

def apply_boundary_conditions(K, F, nodes):
    ndof = len(nodes) * 3
    K_mod = K.copy()
    F_mod = F.copy()
    for i, node in enumerate(nodes):
        for j in range(3):
            if node.constraint[j] is not None:
                idx = i*3 + j
                K_mod[idx, :] = 0
                K_mod[:, idx] = 0
                K_mod[idx, idx] = 1
                F_mod[idx] = node.constraint[j]
    return K_mod, F_mod

def main():
    nodes = [
        Node(0, 0.0, 10.0, constraint=[None, 0, None]),
        Node(1, 10.0, 0.0, constraint=[0, 0, 0]),
        Node(2, 10.0, 10.0, constraint=[None, None, None]),
        Node(3, 20.0, 10.0, constraint=[None, 0, None]),
    ]
    elements = [
        Element(0, 0, 2, E=1e11, A=0.02, I=2e-3, ignore_axial=False),
        Element(1, 1, 2, E=1e11, A=0.02, I=2e-3, ignore_axial=False),
        Element(2, 3, 2, E=1e11, A=0.02, I=2e-3, ignore_axial=False)
    ]

    ndof = len(nodes) * 3
    F = np.zeros(ndof)

    # === Smart element load table ===
    element_loads = [
        {"uniform": 12e3},  # Element 0: uniform load
        {},  # Element 1: uniform + two point loads
        {}                 # Element 2: no loads
    ]

    # --- Assemble element (distributed/point) loads ---
    F = assemble_element_loads(F, nodes, elements, element_loads)

    # --- Add additional nodal loads (such as moment at node 2) ---

    K = assemble_global_stiffness(nodes, elements)
    print("Global stiffness matrix:\n", K)
    K_mod, F_mod = apply_boundary_conditions(K, F, nodes)
    print("Modified stiffness matrix:\n", K_mod)
    print("Modified load vector:\n", F_mod)
    U = np.linalg.solve(K_mod, F_mod)
    print("Displacement result:\n", U)
    print("\nElement local end forces (N, N, Nm, N, N, Nm):")
    for elem in elements:
        n1, n2 = elem.start, elem.end
        node1, node2 = nodes[n1], nodes[n2]
        L = np.hypot(node2.x - node1.x, node2.y - node1.y)
        dof_map = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
        Ue_global = np.array([U[d] for d in dof_map])
        T = transformation_matrix(node1.x, node1.y, node2.x, node2.y)
        Ue_local = T @ Ue_global
        k_local = element_stiffness_2d_frame(elem.E, elem.A, elem.I, L,
                                             ignore_axial=elem.ignore_axial,
                                             start_hinge=elem.hinge[0], end_hinge=elem.hinge[1])
        # If you used distributed or point loads, subtract the equivalent nodal force (in local system)
        # Get equivalent local nodal force for this element from loading table
        f_eqv = np.zeros(6)
        if len(element_loads) > elem.id:
            # Uniform load
            if "uniform" in element_loads[elem.id] and element_loads[elem.id].get("uniform", 0):
                f_eqv += beam_uniform_load_local(element_loads[elem.id]["uniform"], L)
            # Point loads
            if "points" in element_loads[elem.id]:
                for P, a in element_loads[elem.id]["points"]:
                    f_eqv += beam_point_load_local(P, a, L)
        # Final local end force: k_local @ u_local - f_eqv
        Fe_local = k_local @ Ue_local - f_eqv
        print(f"Element {elem.id}: {Fe_local}")

if __name__ == '__main__':
    main()