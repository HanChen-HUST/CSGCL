import numpy as np
import networkx as nx
import torch
from cdlib.utils import convert_graph_formats

def drop_edge_by_modularity(edge_index, edge_weight, p, threshold=1.):
    edge_weight = edge_weight / edge_weight.mean() * (1. - p)
    edge_weight = edge_weight.where(edge_weight > (1. - threshold), torch.ones_like(edge_weight) * (1. - threshold))
    edge_weight = edge_weight.where(edge_weight < 1, torch.ones_like(edge_weight) * 1) 
    sel_mask = torch.bernoulli(edge_weight).to(torch.bool)
    return edge_index[:, sel_mask] 

def drop_feature_by_modularity_dense(feature, modularity, p, max_threshold: float = 0.7):
    x = feature.abs()
    w = x.t() @ torch.tensor(modularity).to(feature.device)
    w = w.log() 
    w = (w.max() - w) / (w.max() - w.min())
    w = w / w.mean() * p
    w = w.where(w < max_threshold, torch.ones_like(w) * max_threshold)
    drop_mask = torch.bernoulli(w).to(torch.bool)
    feature = feature.clone()
    feature[:, drop_mask] = 0.
    return feature

def drop_feature_by_modularity(feature, n_mod, p, max_threshold: float = 0.7):
    x = feature.abs()
    device = feature.device
    w = x.t() @ torch.tensor(n_mod).to(device) 
    w[torch.nonzero(w == 0)] = w.max() # for redundant attributes of Cora
    w = w.log() 
    w = (w.max() - w) / (w.max() - w.min()) 
    w = w / w.mean() * p
    w = w.where(w < max_threshold, max_threshold * torch.ones(1).to(device))
    w = w.where(w > 0, torch.zeros(1).to(device))
    drop_mask = torch.bernoulli(w).to(torch.bool)
    feature = feature.clone()
    feature[:, drop_mask] = 0. 
    return feature

def transition(communities, num_nodes):
    classes = np.full(num_nodes, -1)
    for i, node_list in enumerate(communities):
        classes[np.asarray(node_list)] = i
    return classes

def get_edge_weight(edge_index: torch.Tensor, com: np.ndarray, c_mod: np.ndarray):
    edge_mod = lambda x: c_mod[x[0]] if x[0] == x[1] else -(float(c_mod[x[0]]) + float(c_mod[x[1]])) # a or -(a+b)
    normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    edge_weight = np.asarray([edge_mod([com[u.item()], com[v.item()]]) for u, v in edge_index.T])
    edge_weight = normalize(edge_weight)
    return torch.from_numpy(edge_weight).to(edge_index.device)

def getmod(graph, communities):
    graph = convert_graph_formats(graph, nx.Graph)
    coms = {}
    for cid, com in enumerate(communities):
        for node in com:
            coms[node] = cid 
    inc = dict([])
    deg = dict([])
    links = graph.size(weight="weight")
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")
    for node in graph:
        try:
            com = coms[node] 
            deg[com] = deg.get(com, 0.0) + graph.degree(node, weight="weight") 
            for neighbor, dt in graph[node].items():
                weight = dt.get("weight", 1)
                if coms[neighbor] == com: 
                    if neighbor == node: 
                        inc[com] = inc.get(com, 0.0) + float(weight)
                    else:
                        inc[com] = inc.get(com, 0.0) + float(weight) 
        except:
            pass
    c_mod = []
    for idx, com in enumerate(set(coms.values())):
        c_mod.append((inc.get(com, 0.0) / links) - (deg.get(com, 0.0) / (2.0 * links)) ** 2)
    c_mod = np.asarray(c_mod)
    n_mod = np.zeros(graph.number_of_nodes(), dtype=np.float32)
    for i, w in enumerate(c_mod):
        for j in communities[i]:
            n_mod[j] = c_mod[i]
    return c_mod, n_mod 
