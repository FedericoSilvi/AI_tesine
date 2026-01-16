"""
Wireless network representation for MST optimization.
"""
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from src.models.node import Node

class WirelessNetwork:
    """Represents the wireless communication network."""
    
    def __init__(self):
        """Initialize empty network."""
        self.graph = nx.Graph()
        self.nodes: Dict[int, Node] = {}
    
    def add_node(self, node: Node) -> None:
        """
        Add a node to the network.
        
        Args:
            node: Node instance to add
        """
        self.nodes[node.id] = node
        self.graph.add_node(node.id)
    
    def add_link(self, node1_id: int, node2_id: int, 
                terrain_factor: float = 1.0) -> None:
        """
        Add a link between nodes with calculated cost.
        
        Args:
            node1_id: ID of first node
            node2_id: ID of second node
            terrain_factor: Additional terrain difficulty multiplier
        """
        if node1_id not in self.nodes or node2_id not in self.nodes:
            raise ValueError("Both nodes must exist in the network")
            
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        
        # Calculate link cost
        cost = node1.get_link_cost(node2, terrain_factor)
        
        # Add edge with cost as weight
        self.graph.add_edge(node1_id, node2_id, weight=cost)
    
    @classmethod
    def create_fixed_network(cls) -> 'WirelessNetwork':
        """
        Create a fixed network with predefined nodes and properties.
        
        Returns:
            WirelessNetwork: A new network with fixed layout
        """
        network = cls()
        
        # Define fixed positions with elevations
        positions = {
            # Central area (0-3)
            0: (300, 300, 120),   # Central node
            1: (200, 300, 180),
            2: (400, 300, 140),
            3: (300, 200, 110),
            # North area (4-7) - Mountainous
            4: (200, 500, 950),   # Mountain node
            5: (300, 500, 900),
            6: (400, 500, 820),
            7: (300, 400, 650),
            # South area (8-11) - Valley
            8: (200, 100, 35),
            9: (300, 100, 25),    # Valley node
            10: (400, 100, 30),
            11: (300, 0, 15),
            # East area (12-15)
            12: (500, 200, 220),
            13: (500, 300, 280),
            14: (500, 400, 350),
            15: (600, 300, 320),
            # West area (16-19)
            16: (100, 200, 200),
            17: (100, 300, 250),
            18: (100, 400, 300),
            19: (0, 300, 270),
        }
        
        # Define node properties
        for node_id, (x, y, elevation) in positions.items():
            # Calculate terrain difficulty based on elevation
            terrain_difficulty = 1.0 + (elevation / 1000.0)
            
            # Calculate vulnerability (higher for certain areas)
            base_vulnerability = 0.35
            if elevation > 600:  # Mountain areas
                base_vulnerability = 0.85
            elif elevation < 40:  # Valley areas
                base_vulnerability = 0.55
                
            # Create and add node
            node = Node(
                id=node_id,
                position=(x, y),
                elevation=elevation,
                power_capacity=100.0,  # Default power capacity
                vulnerability=base_vulnerability,
                terrain_difficulty=terrain_difficulty
            )
            network.add_node(node)
        
        # Add all possible connections with terrain-based costs
        connections = [
            # Central connections
            (0, 1), (0, 2), (0, 3), (0, 7),
            # North area
            (4, 5), (5, 6), (7, 5),
            # South area
            (8, 9), (9, 10), (9, 11),
            # East area
            (12, 13), (13, 14), (13, 15),
            # West area
            (16, 17), (17, 18), (17, 19),
            # Cross connections
            (1, 17), (2, 13), (3, 9), (7, 14),
            (4, 18), (6, 14), (8, 16), (10, 12)
        ]
        
        for node1_id, node2_id in connections:
            network.add_link(node1_id, node2_id)
            
        return network
    
    def get_node_positions(self) -> Dict[int, Tuple[float, float]]:
        """Return positions of all nodes for visualization."""
        return {node_id: node.position for node_id, node in self.nodes.items()}
    
    def get_node_colors(self, scenario: str = 'mountain') -> List[str]:
        """
        Return colors for nodes based on their properties and scenario.
        
        Args:
            scenario: Type of scenario ('mountain', 'seismic', or 'energy')
        """
        colors = []
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            
            if scenario == 'mountain':
                # Color based on elevation
                if node.elevation > 500:
                    colors.append('red')  # High elevation
                elif node.elevation > 200:
                    colors.append('orange')  # Medium elevation
                else:
                    colors.append('lightblue')  # Low elevation
            
            elif scenario == 'seismic':
                # Color based on vulnerability
                if node.vulnerability > 0.7:
                    colors.append('red')  # High vulnerability
                elif node.vulnerability > 0.4:
                    colors.append('orange')  # Medium vulnerability
                else:
                    colors.append('lightblue')  # Low vulnerability
            
            elif scenario == 'energy':
                # Color based on power capacity
                if node.power_capacity < 50:
                    colors.append('red')  # Low capacity
                elif node.power_capacity < 80:
                    colors.append('orange')  # Medium capacity
                else:
                    colors.append('lightblue')  # High capacity
                    
        return colors
    
    def calculate_total_power(self, mst_edges: List[Tuple[int, int]]) -> float:
        """Calculate total power requirement for MST solution."""
        total_power = 0.0
        for n1, n2 in mst_edges:
            node1 = self.nodes[n1]
            node2 = self.nodes[n2]
            total_power += node1.get_power_requirement(node2)
        return total_power
    
    def calculate_vulnerability_score(self, mst_edges: List[Tuple[int, int]]) -> float:
        """Calculate overall vulnerability score for MST solution."""
        if not mst_edges:
            return float('inf')
            
        scores = []
        for n1, n2 in mst_edges:
            node1 = self.nodes[n1]
            node2 = self.nodes[n2]
            scores.append(node1.get_vulnerability_score(node2))
            
        return np.mean(scores)
    
    def validate_mst_constraints(self, 
                               mst_edges: List[Tuple[int, int]],
                               scenario: str,
                               constraints: Dict) -> bool:
        """
        Validate MST solution against scenario-specific constraints.
        
        Args:
            mst_edges: List of edges in MST solution
            scenario: Type of scenario
            constraints: Dictionary of constraint values
            
        Returns:
            bool: True if constraints are satisfied
        """
        if scenario == 'mountain':
            # Check max link distance and elevation constraints
            for n1, n2 in mst_edges:
                node1 = self.nodes[n1]
                node2 = self.nodes[n2]
                if node1.distance_to(node2) > constraints.get('max_link_distance', float('inf')):
                    return False
                if node1.elevation_difference(node2) > constraints.get('max_elevation_diff', float('inf')):
                    return False
                    
        elif scenario == 'seismic':
            # Check vulnerability constraints
            vulnerability_score = self.calculate_vulnerability_score(mst_edges)
            if vulnerability_score > constraints.get('max_vulnerability', 1.0):
                return False
                
        elif scenario == 'energy':
            # Check power constraints
            total_power = self.calculate_total_power(mst_edges)
            if total_power > constraints.get('total_power_budget', float('inf')):
                return False
            
        return True
    
    def get_edge_weights(self) -> Dict[Tuple[int, int], float]:
        """Return all edge weights."""
        return nx.get_edge_attributes(self.graph, 'weight')