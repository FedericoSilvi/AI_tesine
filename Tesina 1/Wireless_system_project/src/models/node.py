"""
Node class for wireless communication network representation.
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


@dataclass
class Node:
    """
    Represents a wireless communication station/node.
    
    Attributes:
        id: Unique identifier for the node
        position: (x, y) coordinates
        elevation: Height above sea level (meters)
        power_capacity: Maximum power handling capacity (Watts)
        vulnerability: Seismic vulnerability score (0-1)
        terrain_difficulty: Terrain difficulty factor (1.0 = normal)
    """
    id: int
    position: Tuple[float, float]
    elevation: float = 0.0
    power_capacity: float = 100.0
    vulnerability: float = 0.0
    terrain_difficulty: float = 1.0
    
    @property
    def x(self) -> float:
        """Get x coordinate."""
        return self.position[0]
    
    @property
    def y(self) -> float:
        """Get y coordinate."""
        return self.position[1]
    
    def distance_to(self, other: 'Node') -> float:
        """Calculate Euclidean distance to another node."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    
    def elevation_difference(self, other: 'Node') -> float:
        """Calculate elevation difference with another node."""
        return abs(self.elevation - other.elevation)
    
    def get_link_cost(self, other: 'Node', terrain_factor: float = 1.0) -> float:
        """
        Calculate the cost of establishing a link to another node.
        
        The cost considers:
        - Base distance (Euclidean)
        - Elevation difference
        - Terrain difficulty
        - Power requirements
        """
        base_distance = self.distance_to(other)
        elevation_diff = self.elevation_difference(other)
        
        # Increase cost based on elevation difference
        elevation_factor = 1.0 + (elevation_diff / 1000.0)  # 1000m difference = 2x cost
        
        # Combine terrain difficulties
        terrain_multiplier = (self.terrain_difficulty + other.terrain_difficulty) / 2
        
        return base_distance * elevation_factor * terrain_multiplier * terrain_factor
    
    def get_vulnerability_score(self, other: 'Node') -> float:
        """
        Calculate vulnerability score for a connection with another node.
        Higher score = more vulnerable to disruption.
        """
        # Average the vulnerability of both nodes
        mean_vulnerability = (self.vulnerability + other.vulnerability) / 2
        
        # Factor in elevation (higher elevation = more vulnerable)
        max_elevation = max(self.elevation, other.elevation)
        elevation_factor = 1.0 + (max_elevation / 1000.0)  # 1000m = 2x vulnerability
        
        return mean_vulnerability * elevation_factor
    
    def get_power_requirement(self, other: 'Node') -> float:
        """
        Calculate power required for link with another node.
        Considers distance and elevation difference.
        """
        distance = self.distance_to(other)
        elevation_diff = self.elevation_difference(other)
        
        # Basic power model: P = k * (d^2) * (1 + Δh/100)
        # where k is a constant, d is distance, Δh is elevation difference
        k = 0.001  # power coefficient
        return k * (distance ** 2) * (1 + elevation_diff/100)
    
    def to_dict(self) -> Dict:
        """Convert node attributes to dictionary."""
        return {
            'id': self.id,
            'position': self.position,
            'elevation': self.elevation,
            'power_capacity': self.power_capacity,
            'vulnerability': self.vulnerability,
            'terrain_difficulty': self.terrain_difficulty
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Node':
        """Create node from dictionary data."""
        return cls(
            id=data['id'],
            position=data['position'],
            elevation=data.get('elevation', 0.0),
            power_capacity=data.get('power_capacity', 100.0),
            vulnerability=data.get('vulnerability', 0.0),
            terrain_difficulty=data.get('terrain_difficulty', 1.0)
        )