# Code adapted from
# http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017
# Thanks for saving me some time!

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import pydot as pydot

def _get_markov_edges(df):
    edges = {}
    for col in df.columns:
        for row in df.index:
            edges[(row,col)] = df.loc[row,col]
    return edges

DATA_PATH = './data/generated_matrices/'
DATA = 'mm_monday.csv'
data = pd.read_csv(f'{DATA_PATH}{DATA}')

#add missing labels, str format floats

data = data.set_index('after')
data['checkout'] = '0'
data.loc['checkout','checkout'] = '1'

for column in list(data.columns):
    if column != 'checkout':
        data[f'{column}'] = data[f'{column}'].apply(lambda x:'{:,.2f}'.format(x))

entry_row = pd.Series('0', index=data.columns)
entry_row.name = 'entry'

data = data.append(entry_row)

#make graph edges
graph_edges = _get_markov_edges(data.transpose())

# create graph object
G = nx.MultiDiGraph()
G.name = 'Markov State Diagram'
# nodes from columns
G.add_nodes_from(list(data.columns))


# create edges for dot
for k, v in graph_edges.items():
    tmp_origin, tmp_destination = k[0], k[1]
    
    if  v=='0':
        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v, style='invis')
    elif tmp_origin=='checkout':
        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
    else:
        if tmp_origin == tmp_destination:
            G.add_edge(tmp_origin, tmp_destination, weight=v, label=v, color='red')
        else:
            G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

#make dot file and write to png
nx.drawing.nx_pydot.write_dot(G, 'markov.dot')
(graph,) = pydot.graph_from_dot_file('markov.dot')
graph.write_png('markov.png')