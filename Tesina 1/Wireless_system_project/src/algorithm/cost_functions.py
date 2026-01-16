"""
Heuristic functions for A* pathfinding in energy grid optimization.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from src.models.graph import EnergyGrid


class BaseHeuristic(ABC):
    """Base class for heuristic implementations."""
    
    def __init__(self, grid: EnergyGrid):
        """
        Initialize heuristic with grid.
        
        Args:
            grid: The energy distribution grid
        """
        self.grid = grid
    
    @abstractmethod
    def estimate(self, current: int, goal: int) -> float:
        """
        Estimate cost from current to goal node.
        
        Args:
            current: Current station ID
            goal: Goal station ID
            
        Returns:
            float: Estimated cost to goal
        """
        pass


class EmergencyHeuristic(BaseHeuristic):
    """Heuristic for emergency scenario."""
    
    def __init__(self, grid: EnergyGrid, critical_station: int):
        """Initialize emergency heuristic."""
        super().__init__(grid)
        self.critical_station = critical_station
        
    def estimate(self, current: int, goal: int) -> float:
        """
        Estimate cost considering emergency requirements.
        
        TODO: Student Implementation
        Consider:
        - Distance to critical station
        - Energy consumption impact
        - Path criticality
        """
        pass

class MaintenanceHeuristic(BaseHeuristic):
    """Heuristic for maintenance scenario."""
    
    def __init__(self, grid: EnergyGrid, stations_to_visit: List[int]):
        """Initialize maintenance heuristic."""
        super().__init__(grid)
        self.stations_to_visit = stations_to_visit
        
    def estimate(self, current: int, goal: int) -> float:
        """
        Estimate cost considering maintenance requirements.
        
        TODO: Student Implementation
        Consider:
        - Distance to next station
        - Remaining stations to visit
        - Overall path optimization
        """
        pass

class BalancingHeuristic(BaseHeuristic):
    """Heuristic for energy balancing scenario."""
    
    def __init__(self, grid: EnergyGrid, low_energy_stations: List[int]):
        """Initialize balancing heuristic."""
        super().__init__(grid)
        self.low_energy_stations = low_energy_stations
        
    def estimate(self, current: int, goal: int) -> float:
        """
        Estimate cost considering balancing requirements.
        
        TODO: Student Implementation
        Consider:
        - Distance to low energy stations
        - Energy levels
        - Optimal connection sequence
        """
        pass