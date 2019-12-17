"""Random walker machines."""

import random
import numpy as np
import networkx as nx
from tqdm import tqdm

class SecondOrderRandomWalker:
    """
    Class to create second order random walks.
    """
    def __init__(self, G, p, q, num_walks, walk_length):
        """
        :param G: NetworkX object.
        :param p: Return parameter.
        :param q: In-out parameter.
        :param num_walks: Number of walks per source.
        :param walk_length: Number of nodes in walk.
        """
        self.G = G
        self.nodes = nx.nodes(self.G)
        print("Edge weighting.\n")
        for edge in tqdm(self.G.edges()):
            self.G[edge[0]][edge[1]]['weight'] = 1.0
            self.G[edge[1]][edge[0]]['weight'] = 1.0
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.preprocess_transition_probs()
        self.simulate_walks()

    def node2vec_walk(self, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break
        walk = [str(node) for node in walk]
        return walk

    def simulate_walks(self):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self.G
        self.walks = []
        nodes = list(G.nodes())
        for iteration in range(self.num_walks):
            print("\nRandom walk round: "+str(iteration+1)+"/"+str(self.num_walks)+".\n")
            random.shuffle(nodes)
            for node in tqdm(nodes):
                self.walks.append(self.node2vec_walk(start_node=node))

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G

        alias_nodes = {}
        print("")
        print("Preprocesing.\n")
        for node in tqdm(G.nodes()):
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        for edge in tqdm(G.edges()):
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []

    for kk, prob in enumerate(probs):
        q[kk] = K*prob
    if q[kk] < 1.0:
        smaller.append(kk)
    else:
        larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


class FirstOrderRandomWalker:
    """
    Class to create first order random walks.
    """
    def __init__(self, G, num_walks, walk_length):
        """
        :param G: NetworkX graph object.
        :param num_walks: Number of walks per source node.
        :param walk_length: Number of nodes in turnctaed randonm walk.
        """
        self.G = G
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.simulate_walks()

    def do_walk(self, node):
        """
        Doing a single truncated random walk from a source node.
        :param node: Source node of the truncated random walk.
        :return walk: A single random walk.
        """
        walk = [node]
        while len(walk) < self.walk_length:
            nebs = [n for n in nx.neighbors(self.G, walk[-1])]
            if len(nebs) == 0:
                break
            walk.append(random.choice(nebs))
        return walk

    def simulate_walks(self):
        """
        Doing a fixed number of truncated random walk from every node in the graph.
        """
        self.walks = []
        for iteration in range(self.num_walks):
            print("\nRandom walk round: "+str(iteration+1)+"/"+str(self.num_walks)+".\n")
            for node in tqdm(self.G.nodes()):
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)
