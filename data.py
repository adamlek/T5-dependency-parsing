import conllu
from IPython import embed
import numpy as np
import networkx as nx
import random

def mean_d_dist(tree):
    return np.mean([abs(wi-hi) for wi, hi in enumerate(tree, start=1) if hi != 0])

def mean_h_dist(tree):
    tree_1 = nx.DiGraph()
    tree_1.add_edges_from([(x,i) for i,x in enumerate(tree, start=1)])
    return np.mean([len(nx.shortest_path(tree_1, 0, n)) for n in tree_1 if n != 0])

def read_file(file):
    language = '<' + file.split('/')[-1].split('_')[0] + '>'
    labels = set([])
    dataset = []
    with open(file) as f:
        parsed_file = conllu.parse_incr(f)
        for i, sentence in enumerate(parsed_file):
            ws, hs, rs, ps = list(zip(*[(s['form'], s['head'], s['deprel'], s['upostag']) for 
                                        s in sentence if type(s['id']) is int]))
            # ignore sentences with <= 3 words
            if len(ws) <= 3:
                continue
            
            for l in rs:
                labels.add(l)
            
            dataset.append({'forms': list(map(lambda x: x.lower(), ws)), 
                            'heads': list(hs), 
                            'deprels': list(rs),
                            'upos': list(ps),
                            'len': len(ws),
                            'mhd': mean_d_dist(list(hs)),
                            'mdd': mean_h_dist(list(hs)),
                            'random': random.random(),
                            'language': language})
    return dataset, labels