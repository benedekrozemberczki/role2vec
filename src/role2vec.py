import math
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec
from utils import load_graph, create_documents
from walkers import FirstOrderRandomWalker, SecondOrderRandomWalker
from weisfeiler_lehman_labeling import WeisfeilerLehmanMachine
from motif_count import MotifCounterMachine

class Role2Vec:
    """
    """
    def __init__(self, args):
        """
        Role2Vec machine constructor.
        :param args: Arguments object with the model hyperparameters. 
        """
        self.args = args
        self.graph = load_graph(args.graph_input)

    def do_walks(self):
        """
        
        """
        if self.args.sampling == "second":
            self.sampler = SecondOrderRandomWalker(self.graph, self.args.P, self.args.Q,  self.args.walk_number, self.args.walk_length)
        else:
            self.sampler = FirstOrderRandomWalker(self.graph, self.args.walk_number, self.args.walk_length)
        self.walks = self.sampler.walks
        del self.sampler

    def create_structural_features(self):
        if self.args.features == "wl":
            features = {str(node): str(int(math.log(self.graph.degree(node)+1,self.args.log_base))) for node in self.graph.nodes()}
            machine = WeisfeilerLehmanMachine(self.graph, features, self.args.labeling_iterations)
            machine.do_recursions()
            self.features = machine.extracted_features
        elif self.args.features == "degree":
            self.features = {str(node): [str(self.graph.degree(node))] for node in self.graph.nodes()}
        else:
            machine = MotifCounterMachine(self.graph, self.args)
            self.features  = machine.create_string_labels()

    def create_pooled_features(self):
        """
        """
        features = {str(node):[] for node in self.graph.nodes()}
        for walk in self.walks:
            for node_index in range(self.args.walk_length-self.args.window_size):
                for j in range(1,self.args.window_size+1):
                    features[str(walk[node_index])].append(self.features[str(walk[node_index+j])])
                    features[str(walk[node_index+j])].append(self.features[str(walk[node_index])])

        features = {node: [feature for feature_elems in feature_set for feature in feature_elems] for node, feature_set in features.items()}
        return features
   

    def create_single_embedding(self):
        """
        """
        document_collections = create_documents(self.pooled_features)

        model = Doc2Vec(document_collections,
                        vector_size = self.args.dimensions,
                        window = 0, 
                        min_count = self.args.min_count,
                        alpha = self.args.alpha,
                        dm = 0,
                        min_alpha = self.args.min_alpha,
                        sample = self.args.down_sampling,
                        workers = self.args.workers,
                        epochs = self.args.epochs)
        
        embedding = np.array([model.docvecs[str(node)] for node in self.graph.nodes()])
        return embedding

    def learn_embedding(self):
        """
        """
        self.pooled_features = self.create_pooled_features()
        self.embedding = self.create_single_embedding()

    def save_embedding(self):
        """
        Function to save the embedding.
        :param output_path: Path to the embedding csv.
        :param model: The embedding model object.
        :param files: The list of files.
        :param dimensions: The embedding dimension parameter.
        """
        columns = ["id"] + [ "x_" +str(x) for x in range(self.embedding.shape[1])]
        ids = np.array([node for node in self.graph.nodes()]).reshape(-1,1)
        self.embedding = pd.DataFrame(np.concatenate([ids, self.embedding], axis = 1), columns = columns)
        self.embedding = self.embedding.sort_values(by=['id'])
        self.embedding.to_csv(self.args.output, index = None)

