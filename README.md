Role2Vec
============================================
A scalable gensim implementation of "Learning Role-based Graph Embeddings".
<div style="text-align:center"><img src ="gwnn.jpg" ,width=720/></div>
<p align="justify">
Random walks are at the heart of many existing network embedding methods. However, such algorithms have many limitations that arise from the use of random walks, e.g., the features resulting from these methods are unable to transfer to new nodes and graphs as they are tied to vertex identity. In this work, we introduce the Role2Vec framework which uses the flexible notion of attributed random walks, and serves as a basis for generalizing existing methods such as DeepWalk, node2vec, and many others that leverage random walks. Our proposed framework enables these methods to be more widely applicable for both transductive and inductive learning as well as for use on graphs with attributes (if available). This is achieved by learning functions that generalize to new nodes and graphs. We show that our proposed framework is effective with an average AUC improvement of 16.55% while requiring on average 853x less space than existing methods on a variety of graphs. </p>

This repository provides an implementation of Graph Wavelet Neural Network as described in the paper:

> Learning Role-based Graph Embeddings.
> Nesreen K. Ahmed, Ryan Rossi, John Boaz Lee, Theodore L. Willke, Rong Zhou, Xiangnan Kong, Hoda Eldardiry.
> ArXiV, 2018.
> [[Paper]](https://arxiv.org/abs/1802.02896)

### Requirements

The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          1.11
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
gensim            3.6.0
```
### Datasets

The code takes the **edge list** of the graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. A sample graph for `Cora` is included in the  `input/` directory. 

Training the model is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options

```
  --edge-path        STR   Input graph path.   Default is `input/cora_edges.csv`.
  --features-path    STR   Features path.      Default is `input/cora_features.json`.
  --target-path      STR   Target path.        Default is `input/cora_target.csv`.
  --log-path         STR   Log path.           Default is `logs/cora_logs.json`.
```

#### Model options

```
  --epochs                INT       Number of Adam epochs.         Default is 300.
  --learning-rate         FLOAT     Number of training epochs.     Default is 0.001.
  --weight-decay          FLOAT     Weight decay.                  Default is 5*10**-4.
  --filters               INT       Number of filters.             Default is 16.
  --dropout               FLOAT     Dropout probability.           Default is 0.5.
  --test-size             FLOAT     Test set ratio.                Default is 0.2.
  --seed                  INT       Random seeds.                  Default is 42.
  --approximation-order   INT       Chebyshev polynomial order.    Default is 20.
  --tolerance             FLOAT     Wavelet coefficient limit.     Default is 10**-4.
  --scale                 FLOAT     Heat kernel scale.             Default is 1.0.
```

### Examples

The following commands learn  the weights of a graph wavelet neural network and saves the logs. The first example trains a graph wavelet neural network on the default dataset with standard hyperparameter settings. Saving the logs at the default path.

```
python src/main.py
```
<p align="center">
<img style="float: center;" src="gwnn_run.jpg">
</p>

Training a model with more filters in the first layer.

```
python src/main.py --filters 32
```

Approximationg the wavelets with polynomials that have an order of 5.

```
python src/main.py --approximation-order 5
```
