"""
Visualization utilities for wireless network MST solutions.
"""
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


class NetworkVisualizer:
    """Utility class for visualizing the wireless network and MST solutions."""
    
    SCENARIO_CONFIGS = {
        'mountain': {
            'title': 'Mountain Scenario - Alpine Terrain Optimization',
            'path_color': 'red',
            'description': (
                'Objective: Optimize network in mountainous terrain\n'
                'Constraints:\n'
                '- Minimize elevation changes\n'
                '- Account for terrain difficulty\n'
                '- Maintain connection reliability'
            ),
            'legend_items': [
                ('lightblue', 'Low Elevation Node (<200m)'),
                ('orange', 'Medium Elevation Node (200-500m)'),
                ('red', 'High Elevation Node (>500m)')
            ]
        },
        'seismic': {
            'title': 'Seismic Zone Scenario - Network Resilience',
            'path_color': 'blue',
            'description': (
                'Objective: Ensure network stability in seismic zones\n'
                'Constraints:\n'
                '- Minimize vulnerability\n'
                '- Ensure redundancy\n'
                '- Consider terrain stability'
            ),
            'legend_items': [
                ('lightblue', 'Low Vulnerability Node (<0.4)'),
                ('orange', 'Medium Vulnerability Node (0.4-0.7)'),
                ('red', 'High Vulnerability Node (>0.7)')
            ]
        },
        'energy': {
            'title': 'Energy Optimization Scenario',
            'path_color': 'green',
            'description': (
                'Objective: Minimize power consumption\n'
                'Constraints:\n'
                '- Stay within power budget\n'
                '- Balance load distribution\n'
                '- Optimize transmission paths'
            ),
            'legend_items': [
                ('lightblue', 'High Power Capacity Node (>80%)'),
                ('orange', 'Medium Power Capacity Node (50-80%)'),
                ('red', 'Low Power Capacity Node (<50%)')
            ]
        }
    }
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """Initialize visualizer with figure size."""
        self.figsize = figsize
        plt.close('all')
    
    def display_initial_network(self, network) -> None:
        """
        Display the initial state of the network.
        
        Args:
            network: The wireless network instance
        """
        plt.close('all')
        fig = plt.figure(figsize=self.figsize)
        
        # Get positions and basic visualization data
        pos = network.get_node_positions()
        node_colors = network.get_node_colors()
        
        # Draw the basic network structure
        nx.draw_networkx_nodes(network.graph, pos, 
                             node_color=node_colors,
                             node_size=1000)
        nx.draw_networkx_edges(network.graph, pos, edge_color='gray')
        
        # Draw node labels with elevation information
        labels = {}
        for node_id, node in network.nodes.items():
            labels[node_id] = f"N{node_id}\n{node.elevation:.0f}m"
        
        pos_attrs = {}
        for node_id, coords in pos.items():
            pos_attrs[node_id] = (coords[0], coords[1])
        nx.draw_networkx_labels(network.graph, pos_attrs, labels, font_size=9)
        
        # Draw edge weights (distances/costs)
        edge_labels = {}
        for (u, v, data) in network.graph.edges(data=True):
            edge_labels[(u, v)] = f"{data['weight']:.1f}"
        nx.draw_networkx_edge_labels(network.graph, pos, 
                                   edge_labels=edge_labels,
                                   font_size=8)
        
        # Add legend for elevation-based coloring
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='lightblue', markersize=10,
                      label='Low Elevation Node (<200m)'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='orange', markersize=10,
                      label='Medium Elevation Node (200-500m)'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='red', markersize=10,
                      label='High Elevation Node (>500m)')
        ]
        
        plt.legend(handles=legend_elements, 
                  loc='center left',
                  bbox_to_anchor=(1, 0.5),
                  title='Node Types',
                  title_fontsize=12,
                  fontsize=10)
        
        # Add title and information
        plt.suptitle('Wireless Communication Network - Initial State', 
                    fontsize=14, y=0.95)
        plt.figtext(0.02, 0.02, 
                   'Node Format: Node ID\nElevation (m)\n'
                   'Edge weights represent connection costs',
                   fontsize=10, style='italic',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close('all')
    
    def plot_scenario(self,
                     network,
                     scenario_type: str,
                     mst_edges: List[Tuple[int, int]]) -> None:
        """
        Plot a specific scenario with MST solution.
        
        Args:
            network: The wireless network instance
            scenario_type: Type of scenario
            mst_edges: Edges in the MST solution
        """
        if scenario_type not in self.SCENARIO_CONFIGS:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
            
        config = self.SCENARIO_CONFIGS[scenario_type]
        
        plt.close('all')
        fig = plt.figure(figsize=self.figsize)
        
        # Get positions and colors for current scenario
        pos = network.get_node_positions()
        node_colors = network.get_node_colors(scenario_type)
        
        # Draw the basic network
        nx.draw_networkx_nodes(network.graph, pos, 
                             node_color=node_colors,
                             node_size=1000)
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(network.graph, pos, 
                             edge_color='lightgray',
                             style='dashed',
                             width=1)
        
        # Highlight MST edges
        if mst_edges:
            nx.draw_networkx_edges(network.graph,
                                 pos,
                                 edgelist=mst_edges,
                                 edge_color=config['path_color'],
                                 width=3)
        
        # Draw node labels
        if scenario_type == 'mountain':
            labels = {node_id: f"N{node_id}\n{node.elevation:.0f}m" 
                     for node_id, node in network.nodes.items()}
        elif scenario_type == 'seismic':
            labels = {node_id: f"N{node_id}\n{node.vulnerability:.2f}" 
                     for node_id, node in network.nodes.items()}
        else:  # energy
            labels = {node_id: f"N{node_id}\n{node.power_capacity:.0f}W" 
                     for node_id, node in network.nodes.items()}
        
        nx.draw_networkx_labels(network.graph, pos, labels, font_size=9)
        
        # Draw edge weights for MST edges
        edge_labels = {}
        for u, v in mst_edges:
            edge_data = network.graph.get_edge_data(u, v)
            if edge_data:
                edge_labels[(u, v)] = f"{edge_data['weight']:.1f}"
        
        nx.draw_networkx_edge_labels(network.graph, pos, 
                                   edge_labels=edge_labels,
                                   font_size=8)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=10,
                      label=label)
            for color, label in config['legend_items']
        ]
        
        legend_elements.append(
            plt.Line2D([0], [0], color=config['path_color'],
                      linewidth=3, label='MST Solution')
        )
        
        legend_elements.append(
            plt.Line2D([0], [0], color='lightgray',
                      linewidth=1, linestyle='--',
                      label='Possible Connections')
        )
        
        plt.legend(handles=legend_elements, 
                  loc='center left',
                  bbox_to_anchor=(1, 0.5),
                  title='Legend',
                  title_fontsize=12,
                  fontsize=10)
        
        # Add title and description
        plt.suptitle(config['title'], fontsize=14, y=0.95)
        plt.figtext(0.02, 0.02, config['description'], 
                   fontsize=10, style='italic',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Add edge weight explanation
        plt.figtext(0.02, 0.95, 
                   'Edge weights represent connection costs',
                   fontsize=10, style='italic')
        
        plt.axis('off')
        plt.tight_layout()
        
    def save_plot(self, filename: str) -> None:
        """Save current plot to file."""
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        
    def show_plot(self) -> None:
        """Display current plot."""
        plt.show()
        
    def plot_comparison(self, network, results: Dict) -> None:
        """
        Plot comparison of different MST solutions.
        
        Args:
            network: The wireless network instance
            results: Dictionary with scenario results
        """
        plt.close('all')
        fig, axes = plt.subplots(1, len(results), 
                                figsize=(20, 8),
                                subplot_kw={'aspect': 'equal'})
        
        for ax, (scenario, data) in zip(axes, results.items()):
            config = self.SCENARIO_CONFIGS[scenario]
            pos = network.get_node_positions()
            
            # Draw network on subplot
            nx.draw_networkx_nodes(network.graph, pos,
                                 node_color=network.get_node_colors(scenario),
                                 node_size=700,
                                 ax=ax)
            
            nx.draw_networkx_edges(network.graph, pos,
                                 edge_color='lightgray',
                                 style='dashed',
                                 width=1,
                                 ax=ax)
            
            # Draw MST solution
            nx.draw_networkx_edges(network.graph, pos,
                                 edgelist=data['mst_edges'],
                                 edge_color=config['path_color'],
                                 width=2,
                                 ax=ax)
            
            ax.set_title(f"{scenario.capitalize()}\nTotal Cost: {data['cost']:.1f}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()