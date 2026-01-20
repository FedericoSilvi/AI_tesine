"""
Smart City IoT scenario implementation.
Goal: Optimize network topology for smart city IoT devices.
"""

from typing import List, Tuple, Optional, Dict
from src.models.network import WirelessNetwork
# from src.algorithm.kruskal import KruskalMST
# from src.algorithm.prim import PrimMST
from src.algorithm.cost_functions import SmartCityCostFunction
from src.algorithm.kruskal import KruskalMST
from src.algorithm.prim import PrimMST

def solve_smartcity_scenario(network: WirelessNetwork,
                             algorithm: str = 'kruskal',
                             constraints: Dict = None) -> Optional[List[Tuple[int, int]]]:
    """
    Find optimal MST considering smart city IoT requirements.
    
    Args:
        network: The wireless network
        algorithm: MST algorithm to use ('kruskal' or 'prim')
        constraints: Dictionary of constraints including:
                    - max_latency: Maximum allowed latency per hop (ms)
                    - bandwidth_factor: Impact of bandwidth requirements
                    
    Returns:
        Optional[List[Tuple[int, int]]]: MST edges if found
    """
    # TODO: Student Implementation
    
    # 1. Apply latency-based cost adjustments
    # - Consider distance-based latency (latency increases with distance)
    # - Account for node priority levels (high priority nodes should be connected first)
    # - Factor in bandwidth requirements

    cost_func = SmartCityCostFunction(network=network)

    cost_func.apply_to_network()
    
    # 2. Initialize appropriate MST algorithm
    # if algorithm == 'kruskal':
    #     mst_solver = KruskalMST(network)
    # else:
    #     mst_solver = PrimMST(network)
    
    if algorithm == 'kruskal':
        mst_solver = KruskalMST(network=network)
    else:
        mst_solver = PrimMST(network=network)


    # 3. Consider:
    # - Minimize total latency across the network
    # - Prioritize connections to high-priority nodes (traffic sensors)
    # - Ensure sufficient bandwidth for data transmission
    # - Stay within latency constraints per hop
    
    # 4. Find and validate MST solution
    # mst_edges = mst_solver.find_mst()
    # if validate_smartcity_solution(network, mst_edges, constraints):
    #     return mst_edges
    
    mst_edges = mst_solver.find_mst()
    if validate_smartcity_solution(network,mst_edges,constraints):
        return mst_edges

    return None


def validate_smartcity_solution(network: WirelessNetwork,
                                mst_edges: List[Tuple[int, int]],
                                constraints: Dict) -> bool:
    """
    Validate MST solution for smart city IoT scenario.
    
    Args:
        network: The wireless network
        mst_edges: Proposed MST solution
        constraints: Scenario constraints
        
    Returns:
        bool: True if solution is valid
    """
    if not mst_edges:
        return False
        
    max_latency = constraints.get('max_latency', float('inf'))
    bandwidth_factor = constraints.get('bandwidth_factor', 1.0)
    
    for edge in mst_edges:
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]
        
        # Check latency constraint (distance-based estimation: ~0.1ms per unit distance)
        estimated_latency = node1.distance_to(node2) * 0.1
        print(f"Distance: {node1.distance_to(node2)}")
        print(f"Latency: {estimated_latency}",flush=True)
        if estimated_latency > max_latency:
            return False
            
        # Check bandwidth requirement doesn't exceed capacity
        # Higher priority nodes require more bandwidth
        priority_factor = (node1.terrain_difficulty + node2.terrain_difficulty) / 2
        print(f"Bandwidth: {priority_factor*bandwidth_factor}",flush=True)
        if priority_factor * bandwidth_factor > 3.0:  # Max acceptable load
            return False
    
    return True
