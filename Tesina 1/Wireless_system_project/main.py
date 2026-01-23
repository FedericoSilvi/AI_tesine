import argparse
import logging
from typing import List, Dict, Tuple, Optional
import sys

sys.dont_write_bytecode = True

from src.models.network import WirelessNetwork
from src.utils.visualization import NetworkVisualizer
from src.scenarios.smartcity import solve_smartcity_scenario
from src.scenarios.seismic import solve_seismic_scenario
from src.scenarios.energy import solve_energy_scenario

def setup_logging() -> None:
    """Configure logging for the application."""
    class CustomFormatter(logging.Formatter):
        """Custom formatter with colors and symbols"""
        grey = "\x1b[38;20m"
        blue = "\x1b[34;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        def __init__(self):
            super().__init__()
            self.FORMATS = {
                logging.DEBUG: self.grey + "🔍 DEBUG: %(message)s" + self.reset,
                logging.INFO: self.blue + "ℹ️  %(message)s" + self.reset,
                logging.WARNING: self.yellow + "⚠️  WARNING: %(message)s" + self.reset,
                logging.ERROR: self.red + "❌ ERROR: %(message)s" + self.reset,
                logging.CRITICAL: self.bold_red + "🚨 CRITICAL: %(message)s" + self.reset
            }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.handlers = []
    logger.addHandler(ch)

def calculate_network_cost(network: WirelessNetwork, mst_edges: List[Tuple[int, int]]) -> float:
    """Calculate the total cost of the MST solution."""
    total_cost = 0.0
    if not mst_edges:
        return total_cost
        
    for edge in mst_edges:
        edge_data = network.graph.get_edge_data(edge[0], edge[1])
        if edge_data and 'weight' in edge_data:
            total_cost += edge_data['weight']
            
    return total_cost

def get_scenario_data(scenario_type: str) -> dict:
    """Get predefined data for each scenario."""
    scenarios = {
        'smartcity': {
            'description': 'Smart City IoT scenario - Optimizing connections for urban sensors',
            'constraints': {
                'max_latency': 50,
                'bandwidth_factor': 1.5
            }
        },
        'seismic': {
            'description': 'Seismic zone scenario - Ensuring network resilience',
            'constraints': {
                'redundancy_factor': 2.5,
                'max_vulnerability': 0.65
            }
        },
        'energy': {
            'description': 'Energy optimization scenario - Minimizing power consumption',
            'constraints': {
                'max_power_per_node': 85,
                'total_power_budget': 1350
            }
        }
    }
    return scenarios.get(scenario_type, None)

def validate_mst(network: WirelessNetwork, mst_edges: List[Tuple[int, int]], 
                constraints: Dict) -> bool:
    """Validate if MST meets scenario constraints."""
    if not mst_edges:
        return False
        
    # Check basic MST properties
    if len(mst_edges) != len(network.graph.nodes) - 1:
        return False
        
    # Check scenario-specific constraints (to be implemented)

    

    return True

def main():
    parser = argparse.ArgumentParser(
        description='Wireless Network MST Optimization',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--scenario', 
        type=str, 
        choices=['smartcity', 'seismic', 'energy'],
        required=True,
        help='''Select scenario to solve:
smartcity - Optimize for smart city IoT network
seismic   - Optimize for seismic zone resilience
energy    - Optimize for power consumption'''
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['kruskal', 'prim'],
        default='kruskal',
        help='Select MST algorithm to use'
    )
    
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create and display initial network
        network = WirelessNetwork.create_fixed_network()
        logger.info("Initializing Wireless Communication Network...")

        """ for edge in network.graph.edges():
            node1 = network.nodes[edge[0]]
            node2 = network.nodes[edge[1]]
            vuln = node1.get_vulnerability_score(node2)
            network.graph.edges[edge]['weight'] = vuln  """


        visualizer = NetworkVisualizer()
        visualizer.display_initial_network(network)
        
        # Get scenario data
        scenario_data = get_scenario_data(args.scenario)
        if not scenario_data:
            raise ValueError(f"Invalid scenario selection: {args.scenario}")
        
        logger.info(f"Running {args.scenario} scenario: {scenario_data['description']}")
        
        # Solve the selected scenario
        mst_edges = None
        passed = None 
        error_edges = []
        tot_power_passed = None 

        if args.scenario == 'smartcity':
            mst_edges,passed,error_edges = solve_smartcity_scenario(
                network,
                args.algorithm,
                scenario_data['constraints']
            )
        elif args.scenario == 'seismic':
            mst_edges,passed,error_edges = solve_seismic_scenario(
                network,
                args.algorithm,
                scenario_data['constraints']
            )
        elif args.scenario == 'energy':
            mst_edges,passed,error_edges, tot_power_passed = solve_energy_scenario(
                network,
                args.algorithm,
                scenario_data['constraints']
            )
            
        if mst_edges:
            # Calculate and log total cost
            total_cost = calculate_network_cost(network, mst_edges)
            logger.info(f"Found optimal MST! Total cost: {total_cost:.2f}")
            
            # Visualize the solution
            visualizer.plot_scenario(
                network,
                scenario_type=args.scenario,
                mst_edges=mst_edges
            )
            visualizer.show_plot()
            # Validate the solution (tell the user if something's wrong)
            if passed==False:
                if args.scenario == "seismic":
                    logger.error("MST found but doesn't meet required constraints!")
                    logger.warning(f"Max vulnerability per node : {scenario_data["constraints"]["max_vulnerability"]}")
                    for edge in error_edges:
                        node1 = edge[0]
                        node2 = edge[1]
                        vuln = edge[2]
                        logger.warning(f"Link: node {node1} - node {node2}, Vulnerability score = {vuln:.2f}")
                elif args.scenario == "smartcity":
                    logger.error("MST found but doesn't meet required constraints!")
                    logger.warning(f"Max latency per node : {scenario_data["constraints"]["max_latency"]}, Max load : {scenario_data["constraints"]["bandwidth_factor"]}")
                    for edge in error_edges:
                        node1 = edge[0]
                        node2 = edge[1]
                        lat = edge[2]
                        load = edge[3]
                        logger.warning(f"Link: node {node1} - node {node2}, Latency = {lat:.2f}, Load = {load:.2f}")
                elif args.scenario =="energy":
                    logger.error("MST found but doesn't meet required constraints!")
                    logger.warning(f"Max power per node : {scenario_data["constraints"]["max_power_per_node"]}, Max total power : {scenario_data["constraints"]["total_power_budget"]}")
                    for edge in error_edges:
                        node1 = edge[0]
                        node2 = edge[1]
                        lat = edge[2]
                        load = edge[3]
                        logger.warning(f"Link: node {node1} - node {node2}, Latency = {lat:.2f}, Load = {load:.2f}")
                    if tot_power_passed == False:
                        logger.warning(f"Total solution power exceeds the constraint: {total_cost} > {scenario_data["constraints"]["total_power_budget"]}")


        else:
            logger.error("No valid MST found - Check your implementation!")
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())