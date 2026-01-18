"""
Prim's algorithm implementation for MST in wireless networks.
"""

from typing import List, Tuple, Optional, Dict, Set
from queue import PriorityQueue
from src.models.network import WirelessNetwork


class PrimMST:
    """
    Prim's algorithm for Minimum Spanning Tree.
    
    This class implements Prim's algorithm using a priority queue
    for efficient minimum edge selection.
    
    Algorithm Overview:
    1. Start from an arbitrary node (default: node 0)
    2. Mark the starting node as visited
    3. Add all edges from visited nodes to unvisited nodes to priority queue
    4. Repeat until all nodes are visited:
       - Extract minimum weight edge from priority queue
       - If the destination node is unvisited:
         * Add edge to MST
         * Mark destination as visited
         * Add all edges from destination to unvisited nodes
    
    Time Complexity: O(E log V) with priority queue
    Space Complexity: O(V + E) for visited set and priority queue
    
    Example Usage:
        network = WirelessNetwork.create_fixed_network()
        prim = PrimMST(network)
        mst_edges = prim.find_mst(start_node=0)
    """
    
    def __init__(self, network: WirelessNetwork):
        """
        Initialize Prim's algorithm with network.
        
        Args:
            network: The wireless network to find MST for
        """
        self.network = network
    
    def find_mst(self, start_node: int = 0) -> List[Tuple[int, int]]:
        """
        Find minimum spanning tree using Prim's algorithm.
        
        Args:
            start_node: Starting node for the algorithm (default: 0)
            
        Returns:
            List[Tuple[int, int]]: List of edges in the MST as (node1_id, node2_id) tuples
        """
        # TODO: Student Implementation
        # 1. Initialize data structures
        #    - visited: Set[int] to track visited nodes
        #    - mst_edges: List[Tuple[int, int]] to store MST edges
        #    - pq: PriorityQueue for edges, storing (weight, from_node, to_node)

        visited: Set[int] = set() # Insieme dei nodi visitati
        mst_edges: List[Tuple[int, int]] = [] # Lista degli archi dell'albero di copertura minimo
        pq: PriorityQueue = PriorityQueue() # Coda di priorità per gli archi
        
        # 2. Start from start_node
        #    - Add start_node to visited set
        #    - Add all edges from start_node to priority queue
        #      * Use network.graph.neighbors(start_node) to get neighbors
        #      * Get edge weight with network.graph.edges[start_node, neighbor]['weight']

        visited.add(start_node) # Aggiunge il nodo di partenza all'insieme dei nodi visitati
        self._add_edges_to_queue(start_node, visited, pq) # Aggiunge tutti gli archi del nodo di partenza alla coda di priorità

        for neighbors in self.network.graph.neighbors(start_node): # Per ogni vicino del nodo di partenza
            weight = self.network.graph.edges[start_node, neighbors]['weight'] # Ottiene il peso dell'arco
            pq.put((weight, start_node, neighbors)) # Aggiunge l'arco alla coda di priorità

        # 3. Main loop (while pq not empty AND len(visited) < total nodes)
        #    - Get minimum weight edge from priority queue: (weight, from_node, to_node)
        #    - If to_node is not in visited:
        #      * Add (from_node, to_node) to mst_edges
        #      * Add to_node to visited
        #      * Add all edges from to_node to unvisited neighbors to pq

        total_nodes = self.network.graph.number_of_nodes() # Numero totale di nodi nel grafo

        while not pq.empty() and len(visited) < total_nodes: # Finché la coda non è vuota e non sono stati visitati tutti i nodi
            weight, from_node, to_node = pq.get() # Prende e rimuove l'elemento con la priorità più alta (peso minimo, primo elemento della tupla)
            
            if to_node not in visited: # Se il nodo di destinazione non è stato ancora visitato
                mst_edges.append((from_node, to_node)) # Aggiunge l'arco alla lista degli archi dell'albero di copertura minimo
                visited.add(to_node) # Aggiunge il nodo di destinazione all'insieme dei nodi visitati
                self._add_edges_to_queue(to_node, visited, pq) # Aggiunge tutti gli archi del nodo di destinazione alla coda di priorità
    
        # 4. Return mst_edges
        return mst_edges
    
    def _add_edges_to_queue(self, node: int, visited: Set[int], 
                           pq: PriorityQueue) -> None:
        """
        Add all edges from node to unvisited neighbors to the priority queue.
        
        This is a helper method to avoid code duplication in the main algorithm.
        
        Args:
            node: Source node ID
            visited: Set of already visited node IDs
            pq: Priority queue to add edges to
        """
        # TODO: Student Implementation (Optional helper method)
        # For each neighbor of node:
        #   If neighbor not in visited:
        #     Get edge weight
        #     Add (weight, node, neighbor) to pq
        
        for neighbor in self.network.graph.neighbors(node): # Per ogni vicino del nodo
            if neighbor not in visited: # Se il vicino non è stato ancora visitato
                weight = self.network.graph.edges[node, neighbor]['weight'] # Ottiene il peso dell'arco
                pq.put((weight, node, neighbor)) # Aggiunge l'arco alla coda di priorità
