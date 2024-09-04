
import numpy as np

class Graph():
    """
    Graph object that stores the adjacency matrix of the graph
    
    Methods:
    :__init__: Initializes the Graph object
    :get_edges: Defines the edges of the graph
    :get_hop_distance: Computes the hop distance between nodes
    :get_adjacency: Computes the adjacency matrix of the graph
    :normalize_digraph: Normalizes the adjacency matrix
    
    Attributes:
    :param max_hop: Maximum number of hops to consider
    :param dilation: Dilation factor
    
    :return: Graph object
    """

    def __init__(self, max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.get_edges()
        self.hop_dis = self.get_hop_distance()
        self.get_adjacency()

    def get_edges(self):
        """
        Edge connectivity based on ibug68 landmarks for lip region only
        0-11: lip contour, 12-19: inner lip
        12-19: inner lip
        https://ibug.doc.ic.ac.uk/resources/300-W/
        """

        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                 (8, 9), (9, 10), (10, 11), (11, 0), (12, 13), (13, 14),
                 (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 12)]
        self.num_node = len(edges)
        self.edges = edges

    def __str__(self):
        return str(self.A)

    def get_hop_distance(self):
        """
        compute the hop distance between any two nodes
        """

        num_node = self.num_node
        edges = self.edges
        max_hop = self.max_hop

        A = np.zeros((num_node, num_node))
        for i, j in edges:
            A[j, i] = 1
            A[i, j] = 1

        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def get_adjacency(self):
        """
        compute the adjacency matrix
        """

        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        A = np.zeros((1, self.num_node, self.num_node))
        A[0] = normalize_adjacency
        self.A = A

    def normalize_digraph(A):
        """
        Row-normalize the adjacency matrix
        """
        
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD
