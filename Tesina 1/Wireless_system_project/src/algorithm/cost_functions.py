"""
Cost functions for MST optimization in wireless networks.

This module provides different cost calculation strategies that can be
used to customize edge weights for different scenarios (Smart City, Seismic, Energy).
"""

from abc import ABC, abstractmethod
from typing import Dict
from src.models.network import WirelessNetwork
from src.models.node import Node


class BaseCostFunction(ABC):
    """
    Abstract base class for cost function implementations.
    
    Cost functions determine how edge weights are calculated or adjusted
    for different optimization scenarios. Each scenario may prioritize
    different factors (latency, resilience, power consumption, etc.).
    
    Students should extend this class to create scenario-specific
    cost calculations.
    """
    
    def __init__(self, network: WirelessNetwork):
        """
        Initialize cost function with network.
        
        Args:
            network: The wireless network
        """
        self.network = network
    
    @abstractmethod
    def calculate_cost(self, node1: Node, node2: Node) -> float:
        """
        Calculate cost between two nodes.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            float: Calculated cost for the edge
        """
        pass
    
    def apply_to_network(self) -> None:
        """
        Apply this cost function to all edges in the network.
        
        This method updates all edge weights using the calculate_cost method.
        """
        for edge in self.network.graph.edges():
            node1 = self.network.nodes[edge[0]]
            node2 = self.network.nodes[edge[1]]
            new_cost = self.calculate_cost(node1, node2)
            self.network.graph.edges[edge]['weight'] = new_cost


class SmartCityCostFunction(BaseCostFunction):
    """
    Cost function for Smart City IoT scenario.
    
    Prioritizes:
    - Low latency connections
    - High-priority sensor nodes
    - Bandwidth efficiency
    """
    
    def __init__(self, network: WirelessNetwork, constraints: Dict = None):
        """
        Initialize Smart City cost function.
        
        Args:
            network: The wireless network
            constraints: Optional constraints dict with max_latency, bandwidth_factor
        """
        super().__init__(network)
        self.constraints = constraints or {}
        self.max_latency = self.constraints.get('max_latency', 50)
        self.bandwidth_factor = self.constraints.get('bandwidth_factor', 1.5)
    
    def calculate_cost(self, node1: Node, node2: Node) -> float:
        """
        Calculate cost considering latency and priority.
        
        TODO: Student Implementation
        
        Consider:
        - Distance-based latency estimation
        - Node priority (use terrain_difficulty as priority indicator)
        - Bandwidth requirements
        
        Higher cost = less desirable connection
        """
        # Base implementation - students should improve this
        distance = node1.distance_to(node2)
        print(node1,flush=True)
        # Latency factor (longer distance = higher latency = higher cost)
        latency_cost = distance * 0.1  # 0.1ms per unit distance
        # Penalizzo se sfora il limite 
        if latency_cost>self.max_latency:
            latency_cost=3.0*latency_cost

        # Priority factor (lower priority = higher cost multiplier)
        # Using terrain_difficulty as a proxy for priority (1.0 = high priority)
        avg_priority = (node1.terrain_difficulty + node2.terrain_difficulty) / 2
        priority_multiplier = 1/avg_priority
        
        # Bandwidth efficiency 
        band_cost = avg_priority * self.bandwidth_factor
        # Penalizzo se sfora il limite di carica (visto in smartcity.py)
        if band_cost > 3.0:
            band_cost = 3.0 *band_cost

        return (latency_cost+band_cost) * priority_multiplier


class SeismicCostFunction(BaseCostFunction):
    """
    Cost function for Seismic Zone scenario.
    
    Prioritizes:
    - Low vulnerability connections
    - Stable terrain
    - Redundancy potential
    """
    
    def __init__(self, network: WirelessNetwork, constraints: Dict = None):
        """Initialize Seismic cost function."""
        super().__init__(network)
        self.constraints = constraints or {}
        self.max_vulnerability = self.constraints.get('max_vulnerability', 0.65)
        self.redundancy_factor = self.constraints.get('redundancy_factor', 2.5)
    
    def calculate_cost(self, node1: Node, node2: Node) -> float:
        """
        Calculate cost considering seismic vulnerability.
        
        TODO: Student Implementation
        
        Consider:
        - Node vulnerability scores
        - Connection stability
        - Redundancy requirements
        """
        vuln_score = node1.get_vulnerability_score(node2)
        # Penalizzo se sfora il limite 
        if vuln_score > self.max_vulnerability:
            vuln_score = 3.0 * vuln_score

        # Metrica ideata per la stabilità
        needed_power = node1.get_power_requirement(node2)
        actual_power = (node1.power_capacity + node2.power_capacity) / 2 
        stability_factor = actual_power/needed_power

        stability_penalty = 1/stability_factor 

        # Utilizzo del redundancy factor per accentuare il costo 
        return (vuln_score+stability_penalty)*self.redundancy_factor


class EnergyCostFunction(BaseCostFunction):
    """
    Cost function for Energy Optimization scenario.
    
    Prioritizes:
    - Minimal power consumption
    - Balanced load distribution
    - Efficient transmission paths
    """
    
    def __init__(self, network: WirelessNetwork, constraints: Dict = None):
        """Initialize Energy cost function."""
        super().__init__(network)
        self.constraints = constraints or {}
        self.max_power_per_node = self.constraints.get('max_power_per_node', 85)
        self.total_power_budget = self.constraints.get('total_power_budget', 1350)
    
    def calculate_cost(self, node1: Node, node2: Node) -> float:
        """
        Calculate cost considering power consumption.
        
        TODO: Student Implementation
        
        Consider:
        - Power required for transmission
        - Node power capacities
        - Distance-based power needs
        """
        required_power = node1.get_power_requirement(node2)

        power_per_node = required_power/2

        power_cost = required_power

        if power_per_node > self.max_power_per_node:
            power_cost = 3.0*power_cost
        
        node_load = power_per_node/node1.power_capacity

        if node_load > 0.8: # Se la potenza richiesta supera l'80% della capacità del nodo
            load_cost = node_load*3.0
        else:
            load_cost = node_load
        
        return power_cost + load_cost

        
