"""
Seismic zone scenario implementation.
Goal: Ensure network resilience in seismic areas.
"""

from typing import List, Tuple, Optional, Dict
from src.models.network import WirelessNetwork
# from src.algorithms.kruskal import KruskalMST
# from src.algorithms.prim import PrimMST
from src.algorithm.cost_functions import SeismicCostFunction
from src.algorithm.kruskal import KruskalMST
from src.algorithm.prim import PrimMST

def solve_seismic_scenario(network: WirelessNetwork,
                          algorithm: str = 'kruskal',
                          constraints: Dict = None) -> Tuple[List[Tuple[int, int]],bool, List[Tuple[int,int,float]]]:
    """
    Find optimal MST considering seismic vulnerability.
    
    Args:
        network: The wireless network
        algorithm: MST algorithm to use ('kruskal' or 'prim')
        constraints: Dictionary of constraints including:
                    - redundancy_factor: Required connection redundancy
                    - max_vulnerability: Maximum allowed vulnerability score
                    
    Returns:
        Optional[List[Tuple[int, int]]]: MST edges if found
    """
    # TODO: Student Implementation
    
    # 1. Apply vulnerability-based cost adjustments
    # - Consider node vulnerability scores
    # - Account for terrain stability
    # - Factor in redundancy requirements
    cost_func = SeismicCostFunction(network=network)
    cost_func.apply_to_network()


    # 2. Initialize MST algorithm
    # if algorithm == 'kruskal':
    #     mst_solver = KruskalMST(network)
    # else:
    #     mst_solver = PrimMST(network)
    
    if algorithm == 'kruskal':
        mst_solver = KruskalMST(network=network)
    else:
        mst_solver = PrimMST(network=network)

    # 3. Consider:
    # - Minimize vulnerability scores
    # - Ensure redundant paths where needed
    # - Account for seismic zone characteristics
    # - Balance redundancy with cost
    
    # 4. Find and validate MST solution
    # mst_edges = mst_solver.find_mst()
    # if validate_seismic_solution(network, mst_edges, constraints):
    #     return mst_edges

    mst_edges = mst_solver.find_mst()

    passed = False
    error_edges = []

    [passed,error_edges] = validate_seismic_solution(network,mst_edges,constraints)
        
    return [mst_edges,passed,error_edges]

def validate_seismic_solution(network: WirelessNetwork,
                            mst_edges: List[Tuple[int, int]],
                            constraints: Dict) -> Tuple[bool,List[Tuple[int,int,float]]]:
    """Validate MST solution for seismic scenario."""
    
    if not mst_edges:
        return False
        
    max_vulnerability = constraints.get('max_vulnerability', 0.7)
    redundancy_factor = constraints.get('redundancy_factor', 2.0)
    
    error_edges = [] 

    # Check vulnerability constraints
    for edge in mst_edges:
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]
        
        if node1.get_vulnerability_score(node2) > max_vulnerability:
            info = (node1.id,node2.id,node1.get_vulnerability_score(node2))
            error_edges.append(info)
            
    
    # Check redundancy requirements
    # TODO: Implement redundancy validation
    
    if len(error_edges)==0:
        return [True,error_edges]
    else:
        return [False,error_edges]
