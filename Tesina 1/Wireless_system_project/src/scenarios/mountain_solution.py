# """
# Mountain scenario implementation.
# Goal: Optimize network topology in mountainous terrain.
# """

# from typing import List, Tuple, Optional, Dict
# from src.models.network import WirelessNetwork
# # from src.algorithms.kruskal import KruskalMST
# # from src.algorithms.prim import PrimMST

# def solve_mountain_scenario(network: WirelessNetwork,
#                           algorithm: str = 'kruskal',
#                           constraints: Dict = None) -> Optional[List[Tuple[int, int]]]:
#     """
#     Find optimal MST considering mountainous terrain.
    
#     Args:
#         network: The wireless network
#         algorithm: MST algorithm to use ('kruskal' or 'prim')
#         constraints: Dictionary of constraints including:
#                     - max_link_distance: Maximum allowed link distance
#                     - elevation_factor: Impact of elevation differences
                    
#     Returns:
#         Optional[List[Tuple[int, int]]]: MST edges if found
#     """
#     # TODO: Student Implementation
    
#     # 1. Apply elevation-based cost adjustments
#     # - Consider elevation differences between nodes
#     # - Adjust costs based on terrain difficulty
#     # - Account for weather impact at high altitudes
    
#     # 2. Initialize appropriate MST algorithm
#     # if algorithm == 'kruskal':
#     #     mst_solver = KruskalMST(network)
#     # else:
#     #     mst_solver = PrimMST(network)
    
#     # 3. Consider:
#     # - Minimize extreme elevation changes
#     # - Account for terrain difficulty
#     # - Ensure reliable connections
#     # - Stay within distance constraints
    
#     # 4. Find and validate MST solution
#     # mst_edges = mst_solver.find_mst()
#     # if validate_mountain_solution(network, mst_edges, constraints):
#     #     return mst_edges
    
#     return None

# def validate_mountain_solution(network: WirelessNetwork,
#                              mst_edges: List[Tuple[int, int]],
#                              constraints: Dict) -> bool:
#     """Validate MST solution for mountain scenario."""
#     if not mst_edges:
#         return False
        
#     max_link_distance = constraints.get('max_link_distance', float('inf'))
#     elevation_factor = constraints.get('elevation_factor', 1.5)
    
#     for edge in mst_edges:
#         # Check distance constraint
#         node1 = network.nodes[edge[0]]
#         node2 = network.nodes[edge[1]]
        
#         if node1.distance_to(node2) > max_link_distance:
#             return False
            
#         # Check elevation difference
#         if node1.elevation_difference(node2) * elevation_factor > max_link_distance:
#             return False
    
#     return True


"""
Mountain scenario implementation.
Goal: Optimize network topology in mountainous terrain.
"""

from typing import List, Tuple, Optional, Dict, Set
import networkx as nx
import numpy as np
from src.models.network import WirelessNetwork
from src.models.node import Node

class MountainMST:
    """MST implementation optimized for mountainous terrain."""
    
    def __init__(self, network: WirelessNetwork):
        """Initialize with network."""
        self.network = network
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}
        
    def _make_set(self, v: int) -> None:
        """Initialize disjoint set for vertex."""
        self.parent[v] = v
        self.rank[v] = 0
        
    def _find(self, v: int) -> int:
        """Find set representative with path compression."""
        if self.parent[v] != v:
            self.parent[v] = self._find(self.parent[v])
        return self.parent[v]
        
    def _union(self, v1: int, v2: int) -> None:
        """Union sets by rank."""
        root1 = self._find(v1)
        root2 = self._find(v2)
        
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                root1, root2 = root2, root1
            self.parent[root2] = root1
            if self.rank[root1] == self.rank[root2]:
                self.rank[root1] += 1

    def _calculate_edge_cost(self, node1: Node, node2: Node) -> float:
        """
        Calculate edge cost considering mountainous terrain.
        
        Factors:
        1. Base distance
        2. Elevation difference
        3. Terrain difficulty
        4. Weather impact (higher elevations = higher cost)
        """
        # Base distance cost
        distance_cost = node1.distance_to(node2)
        
        # Elevation difference factor (exponential penalty for large differences)
        elevation_diff = abs(node1.elevation - node2.elevation)
        elevation_factor = 1.0 + (elevation_diff / 100.0) ** 1.5
        
        # Weather impact factor (higher elevation = worse weather)
        avg_elevation = (node1.elevation + node2.elevation) / 2
        weather_factor = 1.0 + (avg_elevation / 1000.0) ** 2
        
        # Terrain difficulty
        terrain_factor = (node1.terrain_difficulty + node2.terrain_difficulty) / 2
        
        return distance_cost * elevation_factor * weather_factor * terrain_factor

    def find_mst(self) -> List[Tuple[int, int]]:
        """
        Find minimum spanning tree using Kruskal's algorithm.
        Adapted for mountainous terrain considerations.
        """
        edges = []
        
        # Calculate costs for all possible edges
        for node1_id in self.network.nodes:
            for node2_id in self.network.nodes:
                if node1_id < node2_id:  # Avoid duplicates
                    node1 = self.network.nodes[node1_id]
                    node2 = self.network.nodes[node2_id]
                    
                    cost = self._calculate_edge_cost(node1, node2)
                    edges.append((node1_id, node2_id, cost))
        
        # Sort edges by cost
        edges.sort(key=lambda x: x[2])
        
        # Initialize disjoint sets
        for v in self.network.nodes:
            self._make_set(v)
        
        # Build MST
        mst_edges = []
        for v1, v2, cost in edges:
            if self._find(v1) != self._find(v2):
                self._union(v1, v2)
                mst_edges.append((v1, v2))
                
        return mst_edges

def calculate_mst_metrics(network: WirelessNetwork, 
                         mst_edges: List[Tuple[int, int]]) -> Dict:
    """Calculate metrics for MST solution."""
    total_distance = 0.0
    max_elevation_diff = 0.0
    total_elevation_change = 0.0
    max_edge_cost = 0.0
    
    for edge in mst_edges:
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]
        
        # Distance
        distance = node1.distance_to(node2)
        total_distance += distance
        
        # Elevation
        elev_diff = abs(node1.elevation - node2.elevation)
        max_elevation_diff = max(max_elevation_diff, elev_diff)
        total_elevation_change += elev_diff
        
        # Cost
        edge_data = network.graph.get_edge_data(edge[0], edge[1])
        if edge_data:
            max_edge_cost = max(max_edge_cost, edge_data['weight'])
            
    return {
        'total_distance': total_distance,
        'max_elevation_diff': max_elevation_diff,
        'avg_elevation_change': total_elevation_change / len(mst_edges),
        'max_edge_cost': max_edge_cost
    }

def solve_mountain_scenario(network: WirelessNetwork,
                          algorithm: str = 'kruskal',
                          constraints: Dict = None) -> Optional[List[Tuple[int, int]]]:
    """
    Find optimal MST considering mountainous terrain.
    
    Args:
        network: The wireless network
        algorithm: MST algorithm to use ('kruskal' or 'prim')
        constraints: Dictionary of constraints including:
                    - max_link_distance: Maximum allowed link distance
                    - elevation_factor: Impact of elevation differences
                    
    Returns:
        Optional[List[Tuple[int, int]]]: MST edges if found
    """
    # Apply terrain-based cost adjustments
    for edge in network.graph.edges():
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]
        
        # Update edge weight with terrain-aware cost
        elevation_diff = abs(node1.elevation - node2.elevation)
        terrain_factor = (node1.terrain_difficulty + node2.terrain_difficulty) / 2
        weather_factor = 1.0 + (max(node1.elevation, node2.elevation) / 1000.0)
        
        base_weight = network.graph.edges[edge]['weight']
        adjusted_weight = base_weight * (1 + elevation_diff/100) * terrain_factor * weather_factor
        
        network.graph.edges[edge]['weight'] = adjusted_weight
    
    # Find MST
    mst = MountainMST(network)
    mst_edges = mst.find_mst()
    
    # Validate solution
    if mst_edges and validate_mountain_solution(network, mst_edges, constraints):
        # Calculate and log metrics
        metrics = calculate_mst_metrics(network, mst_edges)
        print(f"\nMountain MST Metrics:")
        print(f"Total Distance: {metrics['total_distance']:.2f}")
        print(f"Max Elevation Difference: {metrics['max_elevation_diff']:.2f}m")
        print(f"Average Elevation Change: {metrics['avg_elevation_change']:.2f}m")
        print(f"Maximum Edge Cost: {metrics['max_edge_cost']:.2f}")
        
        return mst_edges
        
    return None

def validate_mountain_solution(network: WirelessNetwork,
                             mst_edges: List[Tuple[int, int]],
                             constraints: Dict) -> bool:
    """
    Validate MST solution for mountain scenario.
    
    Args:
        network: The wireless network
        mst_edges: Proposed MST solution
        constraints: Scenario constraints
        
    Returns:
        bool: True if solution is valid
    """
    if not mst_edges:
        return False
        
    max_link_distance = constraints.get('max_link_distance', float('inf'))
    elevation_factor = constraints.get('elevation_factor', 1.5)
    max_elevation_diff = constraints.get('max_elevation_diff', float('inf'))
    
    for edge in mst_edges:
        node1 = network.nodes[edge[0]]
        node2 = network.nodes[edge[1]]
        
        # Check distance constraint
        if node1.distance_to(node2) > max_link_distance:
            return False
            
        # Check elevation constraints
        elevation_diff = abs(node1.elevation - node2.elevation)
        if elevation_diff > max_elevation_diff:
            return False
            
        if elevation_diff * elevation_factor > max_link_distance:
            return False
            
        # Check terrain difficulty
        terrain_factor = (node1.terrain_difficulty + node2.terrain_difficulty) / 2
        if terrain_factor * node1.distance_to(node2) > max_link_distance * 1.5:
            return False
    
    return True