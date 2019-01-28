import math
import pandas as pd
import networkx as nx
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from networkx.generators.atlas import *

class MotifCounterMachine(object):
    """
    Motif and Orbit Counting Tool.
    """
    def __init__(self, graph, args):
        """
        Initializing the object.
        :param graph: NetworkX graph.
        :param args: Arguments object.
        """
        self.graph = graph
        self.args = args

    def create_edge_subsets(self):
        """
        Collecting nodes that form graphlets.
        """
        self.edge_subsets = dict()
        subsets = [[edge[0], edge[1]] for edge in self.graph.edges()]
        self.edge_subsets[2] = subsets
        unique_subsets = dict()
        for i in range(3,self.args.graphlet_size+1):
            for subset in tqdm(subsets):
                for node in subset:
                    for neb in self.graph.neighbors(node):
                        new_subset = subset+[neb]
                        if len(set(new_subset)) == i:
                            new_subset.sort()
                            unique_subsets[tuple(new_subset)] = 1
            subsets = [list(k) for k,v in unique_subsets.items()]
            self.edge_subsets[i] = subsets
            unique_subsets = dict()

    def enumerate_graphs(self):
        """
        Enumerating connected benchmark graphlets.
        """
        graphs = graph_atlas_g()
        self.interesting_graphs = {i:[] for i in range(2,self.args.graphlet_size+1)}
        for graph in graphs:
            if graph.number_of_nodes() > 1 and  graph.number_of_nodes() < self.args.graphlet_size+1:
                if nx.is_connected(graph):
                    self.interesting_graphs[graph.number_of_nodes()].append(graph)

    def enumerate_categories(self):
        """
        Enumerating orbits in graphlets.
        """
        main_index = 0
        self.categories = dict()
        for size, graphs in self.interesting_graphs.items():
            self.categories[size] = dict()
            for index, graph in enumerate(graphs):
                self.categories[size][index] = dict()
                degrees = list(set([graph.degree(node) for node in graph.nodes()]))
                for degree in degrees:
                    self.categories[size][index][degree] = main_index
                    main_index = main_index + 1
        self.unique_motif_count = main_index + 1

    def setup_features(self):
        """
        Calculating the graphlet orbit counts.
        """
        self.features = {node: {i:0 for i in range(self.unique_motif_count)}for node in self.graph.nodes()}
        for size, node_lists in self.edge_subsets.items():
            graphs = self.interesting_graphs[size]
            for nodes in tqdm(node_lists):
                sub_gr = self.graph.subgraph(nodes)
                for index, graph in enumerate(graphs):
                    if nx.is_isomorphic(sub_gr, graph):
                        for node in sub_gr.nodes():
                            self.features[node][self.categories[size][index][sub_gr.degree(node)]] += 1
                        break


    def create_tabular_motifs(self):
        """
        Creating tabular motifs for factorization.
        """
        self.binned_features = {node:[] for node in self.graph.nodes()}
        self.motifs = [[node]+[self.features[node][index] for index in  range(self.unique_motif_count )] for node in self.graph.nodes()]
        self.motifs = pd.DataFrame(self.motifs)
        self.motifs.columns = ["id"] + ["role_"+str(index) for index in range(self.unique_motif_count)]
        for index in range(self.unique_motif_count):
            features = self.motifs["role_"+str(index)].values.tolist()
            if sum(features)>0:
                features = [math.log(feature+1) for feature in features]
                features = pd.qcut(features,self.args.quantiles,duplicates="drop", labels=False)
                for node in self.graph.nodes():
                    self.binned_features[node].append(str(int(index*self.args.quantiles + features[node])))

    def join_strings(self):
        """
        Creating string labels by joining the individual quantile labels.
        """
        return {str(node): ["_".join(self.binned_features[node])] for node in self.graph.nodes()}

    def factorize_string_matrix(self):
        """
        Creating string labels by factorization.
        """
        rows = [node for node, features in self.binned_features.items() for feature in features]
        columns = [int(feature) for node, features in self.binned_features.items() for feature in features]
        scores = [1 for i in range(len(columns))]
        row_number = max(rows)+1
        column_number = max(columns)+1
        features = csr_matrix((scores, (rows, columns)), shape=(row_number, column_number))
        model = NMF(n_components=self.args.factors, init='random', random_state=self.args.seed, alpha = self.args.beta)
        factors = model.fit_transform(features)
        kmeans = KMeans(n_clusters=self.args.clusters, random_state=self.args.seed).fit(factors)
        labels = kmeans.labels_
        features = {str(node): str(labels[node]) for node in self.graph.nodes()}
        return features

    def create_string_labels(self):
        self.create_edge_subsets()
        self.enumerate_graphs()
        self.enumerate_categories()
        self.setup_features()
        self.create_tabular_motifs()
        if self.args.motif_compression == "string":
            features = self.join_strings()
        else:
            features = self.factorize_string_matrix()
        return features
