import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import math

@dataclass
class Node:
    id: int
    x: float
    y: float
    constraint: List[Optional[float]] = field(default_factory=lambda: [None, None, None])

@dataclass
class Element:
    id: int
    start: int
    end: int
    E: float
    A: float
    I: float
    delta_T: float = 0.0     # Uniform temperature change shang xia wen du bian hua yi zhi
    alpha: float = 0.0       # Expansion coefficient
    grad_T: float = 0.0      # Temp. difference top-bottom shang xia wen du bian hua bu yi zhi
    h: float = 0.0           # Section height (m)

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
        # Uniform distributed load
        if "uniform" in element_loads[idx] and element_loads[idx]["uniform"]:
            q = element_loads[idx]["uniform"]
            f_local_uni = beam_uniform_load_local(q, L)
            f_global_uni = T.T @ f_local_uni
            for i in range(6):
                F[dof_map[i]] += f_global_uni[i]
        # Point loads
        if "points" in element_loads[idx]:
            for P, a in element_loads[idx]["points"]:
                f_local_p = beam_point_load_local(P, a, L)
                f_global_p = T.T @ f_local_p
                for i in range(6):
                    F[dof_map[i]] += f_global_p[i]
    return F

def thermal_equivalent_nodal_force(E, A, I, L, alpha, delta_T, grad_T, h):
    F_local = np.zeros(6)
    if delta_T != 0.0 and alpha != 0.0:
        N = E * A * alpha * delta_T
        F_local[0] = -N
        F_local[3] = N
    if grad_T != 0.0 and alpha != 0.0 and h != 0.0:
        M = E * I * alpha * grad_T / h
        F_local[2] = -M
        F_local[5] = M
    return F_local

def assemble_global_temperature_force(nodes, elements):
    ndof = len(nodes) * 3
    F_temp = np.zeros(ndof)
    for elem in elements:
        if (elem.delta_T == 0.0 and elem.grad_T == 0.0) or elem.alpha == 0.0:
            continue
        n1, n2 = elem.start, elem.end
        node1, node2 = nodes[n1], nodes[n2]
        L = np.hypot(node2.x - node1.x, node2.y - node1.y)
        F_local = thermal_equivalent_nodal_force(elem.E, elem.A, elem.I, L, elem.alpha, elem.delta_T, elem.grad_T, elem.h)
        T = transformation_matrix(node1.x, node1.y, node2.x, node2.y)
        F_global = T.T @ F_local
        dof_map = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
        for i in range(6):
            F_temp[dof_map[i]] += F_global[i]
    return F_temp

def apply_boundary_conditions(K, F, nodes, penalty=1e20):
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
    E = 2e11      # Pa
    A = 0.003     # m^2
    I = 1.6e-5    # m^4
    h = 0.4       # m, section height
    alpha = 1.2e-5  # 1/degC

    nodes = [
        Node(0, 0.0, 0.0, constraint=[0, 0, 0]),
        Node(1, 4.0, 0.0, constraint=[None, 0, None]),
        Node(2, 4.0, 3.0, constraint=[None, None, None]),
    ]

    elements = [
        Element(0, 0, 1, E, A, I, delta_T=0.0, alpha=alpha, grad_T=50.0, h=h),
        Element(1, 1, 2, E, A, I, delta_T=0.0, alpha=alpha, grad_T=0.0, h=h),
    ]

    ndof = len(nodes) * 3
    F = np.zeros(ndof)
    F[2*3+1] = -20e3  # Node 2: vertical force -20kN

    # Element-wise loads: uniform = distributed load (N/m), points = list of (P[N], a[m] from node 1)
    element_loads = [
        {"uniform": 18e3, "points": [(10e3, 1.2)]},                # Element 0: distributed + one point load
        {"uniform": 16e3, "points": [(15e3, 0.8), (12e3, 2.1)]}    # Element 1: distributed + two point loads
    ]

    # Assemble temperature effects
    F += assemble_global_temperature_force(nodes, elements)
    # Assemble element distributed and point loads
    F = assemble_element_loads(F, nodes, elements, element_loads)

    K = assemble_global_stiffness(nodes, elements)
    print("\nGlobal stiffness matrix K:\n", K)

    K_mod, F_mod = apply_boundary_conditions(K, F, nodes)
    print("\nModified global stiffness matrix K_mod:\n", K_mod)
    print("\nModified load vector F_mod:\n", F_mod)

    U = np.linalg.solve(K_mod, F_mod)

    print("\nNodal displacements (ux, uy, M):")
    for i in range(len(nodes)):
        print(f"Node {i}: ux={U[i*3]:.6e} m, uy={U[i*3+1]:.6e} m, M={U[i*3+2]:.6e} rad")

    # ---- Calculate and print element local end forces ----
    print("\nElement local end forces (N, N, Nm, N, N, Nm):")
    for idx, elem in enumerate(elements):
        n1, n2 = elem.start, elem.end
        node1, node2 = nodes[n1], nodes[n2]
        L = np.hypot(node2.x - node1.x, node2.y - node1.y)
        dof_map = [n1*3, n1*3+1, n1*3+2, n2*3, n2*3+1, n2*3+2]
        Ue_global = np.array([U[d] for d in dof_map])
        T = transformation_matrix(node1.x, node1.y, node2.x, node2.y)
        Ue_local = T @ Ue_global
        k_local = element_stiffness_2d_frame(elem.E, elem.A, elem.I, L)
        # Subtract the equivalent nodal force (local) for this element
        f_eqv = np.zeros(6)
        # Thermal effects
        f_eqv += thermal_equivalent_nodal_force(elem.E, elem.A, elem.I, L, elem.alpha, elem.delta_T, elem.grad_T, elem.h)
        # Uniform load
        if "uniform" in element_loads[idx] and element_loads[idx]["uniform"]:
            f_eqv += beam_uniform_load_local(element_loads[idx]["uniform"], L)
        # Point loads
        if "points" in element_loads[idx]:
            for P, a in element_loads[idx]["points"]:
                f_eqv += beam_point_load_local(P, a, L)
        # Local element end force: k_local @ u_local - f_eqv
        Fe_local = k_local @ Ue_local - f_eqv
        print(f"Element {elem.id}: {Fe_local}")

if __name__ == '__main__':
    main()
