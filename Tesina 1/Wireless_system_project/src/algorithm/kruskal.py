"""
Kruskal's algorithm implementation for MST in wireless networks.
"""

from typing import List, Tuple, Optional, Dict
from src.models.network import WirelessNetwork


class KruskalMST:
    """
    Kruskal's algorithm for Minimum Spanning Tree.
    
    This class implements Kruskal's algorithm using the Union-Find 
    (Disjoint Set Union) data structure for efficient cycle detection.
    
    Algorithm Overview:
    1. Sort all edges by weight in ascending order
    2. Initialize each node as its own set (Union-Find)
    3. For each edge (in sorted order):
       - If the edge connects two different sets (no cycle):
         * Add edge to MST
         * Union the two sets
    4. Stop when MST has V-1 edges (V = number of vertices)
    
    Time Complexity: O(E log E) where E is the number of edges
    Space Complexity: O(V) for Union-Find data structure
    
    Example Usage:
        network = WirelessNetwork.create_fixed_network()
        kruskal = KruskalMST(network)
        mst_edges = kruskal.find_mst()
    """
    
    def __init__(self, network: WirelessNetwork):
        """
        Initialize Kruskal's algorithm with network.
        
        Args:
            network: The wireless network to find MST for
        """
        self.network = network
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}
    
    def find_mst(self) -> List[Tuple[int, int]]:
        """
        Find minimum spanning tree using Kruskal's algorithm.
        
        Returns:
            List[Tuple[int, int]]: List of edges in the MST as (node1_id, node2_id) tuples
        """
        # TODO: Student Implementation - DONE
        
        # 1. Initialize Union-Find data structure
        #    - Call _make_set() for each node in the network
        #    - This creates a forest where each node is its own tree
        
        # 2. Get all edges with their weights and sort by weight
        #    - Use network.graph.edges(data=True) to get edges with weights
        #    - Create list of (weight, node1, node2) tuples
        #    - Sort this list by weight (ascending order)
        
        # 3. Process edges in sorted order
        #    - For each edge, use _find() to check if nodes are in different sets
        #    - If different sets (no cycle would be created):
        #      * Add edge to MST result list
        #      * Call _union() to merge the sets
        #    - Stop when MST has (number_of_nodes - 1) edges
        
        # 4. Return the list of MST edges


        mst: List[Tuple[int, int]] = [] # lista vuota che conterrà gli archi dell’MST

        # chiama _make_set() per ogni nodo della rete, in pratica (credo) crea una rete di sottoalberi da un nodo, ogni nodo è "padre" di se stesso
        for node in self.network.graph.nodes():
            self._make_set(node)

        edges: List[Tuple[float, int, int]] = [] # lista che conterrà gli archi definiti come tuple (peso, nodo1, nodo2)

        # assegna un peso a tutti gli archi, se esiste il campo weight mette quello altrimenti 1
        for u, v, data in self.network.graph.edges(data=True): # (grazie gpitti) con data=True l'arco ha 3 elementi: (u,v,data) dove data è un dizionario di attributi dell'arco
                                                               # in add_link dovrebbe esserci una roba tipo "weight": cost che viene messa nel data
            weight = data.get("weight", 1)
            edges.append((weight, u, v)) # aggiungi l'arco alla lista degli archi

        edges.sort(key=lambda x: x[0]) # ordina per peso in ordine crescente la lista degli archi

        num_nodes = self.network.graph.number_of_nodes() # l'MST dovrebbe avere sempre num_nodes - 1 archi

        for weight, u, v in edges: # scorre gli archi dal piu leggero al piu pesante perchè prima abbiamo ordinato la lista
            if self._find(u) != self._find(v): # se i due nodi appartengono a sottoalberi diversi (genitori diversi trovati con _find() ) li aggiungo all'MST e unisco i sottoalberi
                mst.append((u, v))
                self._union(u, v)

                if len(mst) == num_nodes - 1: # se raggiungo num_nodes - 1 vuol dire che l'MST è completo
                    break
        
        return mst # restituisco l'MST completo
    
    def _make_set(self, v: int) -> None:
        """
        Initialize a disjoint set containing only vertex v.
        
        Each vertex starts as its own parent (root of its tree)
        with rank 0 (tree height).
        
        Args:
            v: Vertex ID to initialize
        """
        # TODO: Student Implementation - DONE
        # Hint: self.parent[v] = v, self.rank[v] = 0

        self.parent[v] = v  # il nodo v è inizialmente padre di se stesso
        self.rank[v] = 0  # l'albero ha altezza (rank) iniziale 0

        # e chest'è
    
    def _find(self, v: int) -> int:
        """
        Find the set representative (root) for vertex v.

        Uses path compression optimization: during the search,
        make every node on the path point directly to the root.
        This flattens the tree for faster future lookups.

        Args:
            v: Vertex ID to find root for

        Returns:
            int: Root (representative) of the set containing v
        """
        # TODO: Student Implementation - DONE
        # Hint: Recursively find root, with path compression:
        # if self.parent[v] != v:
        #     self.parent[v] = self._find(self.parent[v])
        # return self.parent[v]

        if self.parent[v] != v:
            # se v non è la radice, risali ricorsivamente
            self.parent[v] = self._find(self.parent[v])

        return self.parent[v]

        # e chest'è 2
    
    def _union(self, v1: int, v2: int) -> None:
        """
        Merge the sets containing v1 and v2.
        
        Uses union by rank optimization: attach the shorter tree
        under the root of the taller tree to keep trees balanced.
        
        Args:
            v1: First vertex ID
            v2: Second vertex ID
        """
        # TODO: Student Implementation - DONE
        # Steps:
        # 1. Find roots of both vertices
        # 2. If roots are same, already in same set (return)
        # 3. Attach smaller rank tree under root of higher rank tree
        # 4. If ranks are equal, increment rank of new root

        # trova le radici di entrambi i nodi
        root1 = self._find(v1)
        root2 = self._find(v2)

        # 2se le radici sono uguali, sono già nello stesso sottoalbero e quindi ritorni
        if root1 == root2:
            return

        # altrimenti union by rank, ovvero attacca l’albero più piccolo sotto quello più grande
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            # se hanno lo stesso rank scegli 1 come nuovo root
            self.parent[root2] = root1
            self.rank[root1] += 1  # aumenta il rank della nuova radice
