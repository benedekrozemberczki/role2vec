import argparse

def parameter_parser():

    """
    A method to parse up command line parameters. By default it gives an embedding of the Facebook food dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by ID.
    """

    parser = argparse.ArgumentParser(description = "Run .")


    parser.add_argument('--graph-input',
                        nargs = '?',
                        default = "./input/cora_edges.csv",
	                help = 'Input folder with jsons.')

    parser.add_argument('--output',
                        nargs = '?',
                        default = './output/cora_role2vec.csv',
	                help = 'Embeddings path.')

    parser.add_argument('--features',
                        nargs = '?',
                        default = 'wl',
	                help = 'Embeddings path.')

    parser.add_argument('--dimensions',
                        type = int,
                        default = 128,
	                help = 'Number of dimensions. Default is 16.')

    parser.add_argument('--walk-number',
                        type = int,
                        default = 3,
	                help = 'Number of walks. Default is 5.')

    parser.add_argument('--walk-length',
                        type = int,
                        default = 10,
	                help = 'Walk length. Default is 80.')

    parser.add_argument('--sampling',
                        nargs = '?',
                        default = 'first',
	                help = 'Random walk order.')

    parser.add_argument('--P',
                        type = float,
                        default = 4.00,
	                help = 'Number of walks. Default is 1.0.')

    parser.add_argument('--Q',
                        type = float,
                        default = 0.25,
	                help = 'Number of walks. Default is 1.0.')

    parser.add_argument('--down-sampling',
                        type = float,
                        default = 0.0001,
	                help = 'Number of walks. Default is 1.0.')

    parser.add_argument('--alpha',
                        type = float,
                        default = 0.025,
	                help = 'Number of walks. Default is 1.0.')

    parser.add_argument('--min-alpha',
                        type = float,
                        default = 0.025,
	                help = 'Number of walks. Default is 1.0.')

    parser.add_argument('--window-size',
                        type = int,
                        default = 5,
	                help = 'Number of neighbor embeddings. Default is 3.')

    parser.add_argument('--min-count',
                        type = int,
                        default = 1,
	                help = 'Number of neighbor embeddings. Default is 3.')

    parser.add_argument('--workers',
                        type = int,
                        default = 4,
	                help = 'Number of cores. Default is 4.')

    parser.add_argument('--epochs',
                        type = int,
                        default = 10,
	                help = 'Number of cores. Default is 10.')

    parser.add_argument('--labeling-iterations',
                        type = int,
                        default = 2,
	                help = 'Number of cores. Default is 4.')

    parser.add_argument('--log-base',
                        type = int,
                        default = 1.5,
	                help = 'Number of cores. Default is 4.')


    parser.add_argument('--graphlet-size',
                        type = int,
                        default = 3,
	                help = 'Number of cores. Default is 10.')

    parser.add_argument('--quantiles',
                        type = int,
                        default = 5,
	                help = 'Number of cores. Default is 10.')

    parser.add_argument('--seed',
                        type = int,
                        default = 10,
	                help = 'Number of cores. Default is 10.')

    parser.add_argument('--factors',
                        type = int,
                        default = 8,
	                help = 'Number of cores. Default is 10.')

    parser.add_argument('--clusters',
                        type = int,
                        default = 50,
	                help = 'Number of cores. Default is 10.')

    parser.add_argument('--motif-compression',
                        nargs = '?',
                        default = 'string',
	                help = 'Embeddings path.')

    return parser.parse_args()
