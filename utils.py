import networkx as nx
import pandas as pd
import json
import random
import pandas as pd
import networkx as nx
from gensim.models.doc2vec import TaggedDocument

def load_graph(graph_path):
    """

    """
    graph = nx.from_edgelist(pd.read_csv(graph_path).values.tolist())
    graph.remove_edges_from(graph.selfloop_edges())
    return graph

def create_documents(features):
    """

    """
    docs = [TaggedDocument(words = v, tags = [str(k)]) for k, v in features.items()]
    return docs
