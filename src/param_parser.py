"""Parsing the parameters."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by ID.
    """
    parser=argparse.ArgumentParser(description="Run Role2Vec.")

    parser.add_argument("--graph-input",
                        nargs="?",
                        default="./input/cora_edges.csv",
	                help="Input graph path -- edge list csv.")

    parser.add_argument("--output",
                        nargs="?",
                        default="./output/cora_role2vec.csv",
	                help="Embeddings path.")
	
    parser.add_argument("--window-size",
                        type=int,
                        default=5,
	                help="Window size for skip-gram. Default is 5.")

    parser.add_argument("--walk-number",
                        type=int,
                        default=10,
	                help="Number of random walks. Default is 10.")

    parser.add_argument("--walk-length",
                        type=int,
                        default=80,
	                help="Walk length. Default is 80.")

    parser.add_argument("--sampling",
                        nargs="?",
                        default="first",
	                help="Random walk order.")

    parser.add_argument("--P",
                        type=float,
                        default=1.00,
	                help="Return parameter. Default is 1.0.")

    parser.add_argument("--Q",
                        type=float,
                        default=1.00,
	                help="Inout parameter. Default is 1.0.")

    parser.add_argument("--dimensions",
                        type=int,
                        default=128,
	                help="Number of dimensions. Default is 128.")

    parser.add_argument("--down-sampling",
                        type=float,
                        default=0.001,
	                help="Down sampling frequency. Default is 0.001.")

    parser.add_argument("--alpha",
                        type=float,
                        default=0.025,
	                help="Initial learning rate. Default is 0.025.")

    parser.add_argument("--min-alpha",
                        type=float,
                        default=0.025,
	                help="Final learning rate. Default is 0.025.")

    parser.add_argument("--min-count",
                        type=int,
                        default=1,
	                help="Minimal feature count. Default is 1.")

    parser.add_argument("--workers",
                        type=int,
                        default=4,
	                help="Number of cores. Default is 4.")

    parser.add_argument("--epochs",
                        type=int,
                        default=10,
	                help="Number of epochs. Default is 10.")
	
    parser.add_argument("--features",
                        nargs="?",
                        default="wl",
	                help="Feature extraction mechanism. Default is wl.")

    parser.add_argument("--labeling-iterations",
                        type=int,
                        default=2,
	                help="Number of WL labeling iterations. Default is 2.")

    parser.add_argument("--log-base",
                        type=int,
                        default=1.5,
	                help="Log base for label creation. Default is 1.5.")

    parser.add_argument("--graphlet-size",
                        type=int,
                        default=4,
	                help="Maximal graphlet size. Default is 4.")

    parser.add_argument("--quantiles",
                        type=int,
                        default=5,
	                help="Number of quantiles for binning. Default is 5.")

    parser.add_argument("--motif-compression",
                        nargs="?",
                        default="string",
	                help="Motif compression procedure -- string or factorization.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
	                help="Sklearn random seed. Default is 42.")

    parser.add_argument("--factors",
                        type=int,
                        default=8,
	                help="Number of factors for motif compression. Default is 8.")

    parser.add_argument("--clusters",
                        type=int,
                        default=50,
	                help="Number of motif based labels. Default is 50.")

    parser.add_argument("--beta",
                        type=float,
                        default=0.01,
	                help="Motif compression factorization regularizer. Default is 0.01.")

    return parser.parse_args()
