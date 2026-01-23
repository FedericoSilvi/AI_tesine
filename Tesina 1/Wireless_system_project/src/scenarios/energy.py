"""
Energy optimization scenario implementation.
Goal: Minimize power consumption while maintaining connectivity.
"""

from typing import List, Tuple, Optional, Dict
from src.models.network import WirelessNetwork
# from src.algorithms.kruskal import KruskalMST
# from src.algorithms.prim import PrimMST
from src.algorithm.cost_functions import EnergyCostFunction
from src.algorithm.kruskal import KruskalMST
from src.algorithm.prim import PrimMST


def solve_energy_scenario(network: WirelessNetwork,
                         algorithm: str = 'kruskal',
                         constraints: Dict = None) -> Tuple[List[Tuple[int,int]],bool,List[Tuple[int,int,float]],bool]:
    """
    Find optimal MST considering power consumption.
    
    Args:
        network: The wireless network
        algorithm: MST algorithm to use ('kruskal' or 'prim')
        constraints: Dictionary of constraints including:
                    - max_power_per_node: Maximum power per node
                    - total_power_budget: Total network power budget
                    
    Returns:
        Optional[List[Tuple[int, int]]]: MST edges if found
    """
    # TODO: Student Implementation
    
    # 1. Apply power-based cost adjustments
    # - Consider power requirements for connections
    # - Account for node power capacities
    # - Factor in distance-based power needs

    cost_func = EnergyCostFunction(network=network)

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
    # - Minimize total power consumption
    # - Balance load across nodes
    # - Ensure sufficient power capacity
    # - Optimize transmission distances
    

    # 4. Find and validate MST solution
    # mst_edges = mst_solver.find_mst()
    # if validate_energy_solution(network, mst_edges, constraints):
    #     return mst_edges

    mst_edges = mst_solver.find_mst()
    passed, error_edges, power_passed = validate_energy_solution(network,mst_edges,constraints)

    return mst_edges,passed, error_edges, power_passed
    
    

def validate_energy_solution(network: WirelessNetwork,
                           mst_edges: List[Tuple[int, int]],
                           constraints: Dict) -> Tuple[bool,List[Tuple[int,int,float]],bool]:
    """Validate MST solution for energy optimization scenario."""
    if not mst_edges:
        return False
        
    error_edges = [] 
    tot_power_passed = True

    max_power_per_node = constraints.get('max_power_per_node', 100.0)
    total_power_budget = constraints.get('total_power_budget', float('inf'))
    
    # Calculate power requirements
    total_power = 0.0
    node_power = {node_id: 0.0 for node_id in network.nodes}
    
    for edge in mst_edges:
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]
        
        power_req = node1.get_power_requirement(node2)
        total_power += power_req
        
        # Add power requirements to both nodes
        node_power[edge[0]] += power_req / 2
        node_power[edge[1]] += power_req / 2

        node1.scale_power_capacity(node_power[edge[0]])
        node2.scale_power_capacity(node_power[edge[1]])
        
    # Check constraints
    
    if total_power > total_power_budget:
        tot_power_passed = False
        
        
    for power in node_power.values():
        
        if power > max_power_per_node:
            info = (node1, node2, power)
            error_edges.append(info)

            
    
    if len(error_edges)==0 and tot_power_passed == True :

        return True,error_edges,tot_power_passed
    
    else :

        return False,error_edges,tot_power_passed