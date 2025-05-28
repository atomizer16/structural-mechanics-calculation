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
        Node(0, 0.0, 0.0, constraint=[0, 0, 0]),  #fixed
        Node(1, 4.0, 0.0, constraint=[None, 0, None]),   #constrained y (x,y,M)
        Node(2, 4.0, 3.0, constraint=[None, None, None]),   #free
    ]
    elements = [
        Element(0, 0, 1, E=2e11, A=0.003, I=1.6e-5),   # Element 0 from node 0 to node 1
        Element(1, 1, 2, E=2e11, A=0.003, I=1.6e-5)
    ]
    ndof = len(nodes) * 3  # def degrees of freedom (3 per node)
    F = np.zeros(ndof)  # def global force vector
    F[2*3+1] = -20e3  # Apply -20kN vertical force at node 2

    K = assemble_global_stiffness(nodes, elements)
    print(K)
    K_mod, F_mod = apply_boundary_conditions(K, F, nodes)
    print(K_mod)
    print(F_mod)
    U = np.linalg.solve(K_mod, F_mod)
    print(U)

if __name__ == '__main__':
    main()
