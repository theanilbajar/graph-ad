import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

def node_iter(G):
    # if float(nx.__version__)<2.0:
    #     return G.nodes()
    # else:
    return G.nodes

def node_dict(G):
    # if float(nx.__version__)>2.1:
    node_dict = G.nodes
    # else:
    #     node_dict = G.node
    return node_dict

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


