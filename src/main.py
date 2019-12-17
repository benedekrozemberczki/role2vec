"""Model running."""

from param_parser import parameter_parser
from role2vec import Role2Vec
from utils import tab_printer

def main(args):
    """
    Role2Vec model fitting.
    :param args: Arguments object.
    """
    tab_printer(args)
    model = Role2Vec(args)
    model.do_walks()
    model.create_structural_features()
    model.learn_embedding()
    model.save_embedding()

if __name__ == "__main__":
    args = parameter_parser()
    main(args)

