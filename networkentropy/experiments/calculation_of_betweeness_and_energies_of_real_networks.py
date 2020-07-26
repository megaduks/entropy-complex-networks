# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from bs4 import BeautifulSoup
import requests
import wget
import tarfile
import os
import shutil
import time
from tqdm import tqdm
# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import networkx as nx
import scipy, scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl

from sklearn import preprocessing, linear_model

import sys
sys.path.insert(0, '../../')

from networkentropy import network_energy as ne
import time

from multiprocessing import Pool
import time
import itertools

import matplotlib.pyplot as plt
import networkx as nx


# %%
def read_avalilable_datasets_konect():
    base_url = "http://konect.uni-koblenz.de/downloads/"
    response = requests.get(base_url)
    
    if response.status_code != 200:
        print("An error occurred while getting data.")
    else:
        html = response.content
        soup = BeautifulSoup(html, "html5lib")
        
        table_html = soup.find(id='sort1')
        
        thead_html = table_html.find('thead')
        tbody_html = table_html.find('tbody')
         
        column_names=[row.text for row in thead_html.findAll('td')]
        rows = tbody_html.findAll('tr')
        values=[[cell.get('href') for cell in value('a') if 'tsv' in cell.get('href')] for value in rows]
        return [val[0].replace('.tar.bz2','').replace('tsv/','') for val in values]
        
def download_tsv_dataset_konect(network_name):
    assert (network_name in read_avalilable_datasets_konect()),"No network named: '"+network_name+"' found in Konect!"
    
    tsv_file = 'http://konect.uni-koblenz.de/downloads/tsv/'+network_name+'.tar.bz2'
    output_file=network_name+'.tar.bz2'
    file_name = wget.download(tsv_file, out=output_file)
    if os.path.exists(output_file):
        shutil.move(file_name,output_file)
    
    return output_file
    
def unpack_tar_bz2_file(file_name):
    tar = tarfile.open("./"+file_name, "r:bz2")
    output_dir="./network_"+file_name.replace('.tar.bz2','')+"/"
    tar.extractall(output_dir)
    tar.close()
    return output_dir

def build_network_from_out_konect(network_name):
    file_name=download_tsv_dataset_konect(network_name=network_name)
    output_dir=unpack_tar_bz2_file(file_name)+network_name+"/"
    files = [file for file in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, file))]
    out_file = [file for file in files if 'out.' in file]
    assert (len(out_file)>0), 'No out. file in the directory.'
    
    #building network
    G=nx.read_adjlist(output_dir+out_file[0], comments='%')
    
    return G


# %%
def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def _betmap(G_normalized_weight_sources_tuple):
    """Pool for multiprocess only accepts functions with one argument.
    This function uses a tuple as its only argument. We use a named tuple for
    python 3 compatibility, and then unpack it when we send it to
    `betweenness_centrality_source`
    """
    return nx.betweenness_centrality_source(*G_normalized_weight_sources_tuple)


def betweenness_centrality_parallel(G, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G.nodes(), int(G.order() / node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.map(_betmap,
                  zip([G] * num_chunks,
                      [True] * num_chunks,
                      [None] * num_chunks,
                      node_chunks))

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
            
    p.close()
    return bt_c

def calculate_betweenes(graph, k):
    return nx.betweenness_centrality(graph, k=k)

def calculate_randic_energy(graph):
    results={}
    for n in graph.nodes:
        g = nx.ego_graph(G=graph, n=n, radius=1)
        results[n]=ne.get_randic_energy(g)
    return results

def calculate_graph_energy(graph):
    time_evaluation={}
    time_evaluation['ego']=0
    time_evaluation['graph_energy']=0
     
    results={}
    for n in graph.nodes:
        start = time.clock()
        g = nx.ego_graph(G=graph, n=n, radius=1)
        time_evaluation['ego']=time_evaluation['ego']+(time.clock() - start)
        start = time.clock()
        results[n]=ne.get_graph_energy(g)
        time_evaluation['graph_energy']=time_evaluation['graph_energy']+(time.clock() - start)
    return results, time_evaluation

def calculate_graph_energy_numpy(graph):
    results={}
    for n in graph.nodes:
        g = nx.ego_graph(G=graph, n=n, radius=1)
        results[n]=get_graph_energy_numpy(g)
    return results

def get_graph_energy_numpy(G):
    M = nx.adjacency_matrix(G).todense()
    graph_energy = np.abs(np.linalg.eigvals(M).real).sum()
    return graph_energy

def normalize_df_column(df_column):
    x = df_column.values.astype(float)
    min_max_scaler = skl.preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1))
    return x_scaled


# %%

# %%

# %%

# %%
networks_names=['moreno_beach',
 'moreno_bison',
 'moreno_blogs',
 'moreno_cattle',
 'moreno_crime',
 'moreno_health',
 'moreno_hens',
 'moreno_highschool',
 'moreno_innovation',
 'moreno_kangaroo',
 'moreno_lesmis',
 'moreno_mac',
 'moreno_names',
 'moreno_oz',
 'moreno_propro',
 'moreno_rhesus',
 'moreno_sampson',
 'moreno_seventh',
 'moreno_sheep',
 'moreno_taro',
 'moreno_train',
 'moreno_vdb',
 'moreno_zebra',
 'brunson_club-membership',
 'brunson_southern-women',
 'brunson_corporate-leadership',
 'brunson_revolution',
 'brunson_south-africa',
 'ucidata-gama',
 'ucidata-zachary',
 'opsahl-collaboration',
 'opsahl-openflights',
 'opsahl-powergrid',
 'opsahl-southernwomen',
 'opsahl-ucforum',
 'opsahl-ucsocial',
 'opsahl-usairport',
 'contiguous-usa',
 'dolphins',
 'adjnoun_adjacency',
 'mit',
 'foodweb-baydry',
 'foodweb-baywet',
 'sociopatterns-hypertext',
 'sociopatterns-infectious',
 'radoslaw_email',
 'maayan-foodweb',
 'arenas-jazz']

# %%
networks=[]
for network_name in tqdm(networks_names):
    networks.append(build_network_from_out_konect(network_name))

# %%
real_data_measures=pd.DataFrame(columns=['node', 'value_type','value','network'])

for i in tqdm(range(len(networks))):
    G = networks[i]
    
    be=calculate_betweenes(G,k=None)
    tmp_df=pd.DataFrame({'node': [i[0] for i in be.items()],
                         'value_type': ['betweenness' for i in be.items()],
                         'value': [i[1] for i in be.items()],
                         'network': [networks_names[i] for j in be.items()]
                        })
    tmp_df['value']=normalize_df_column(tmp_df['value'])
    real_data_measures=pd.concat([real_data_measures,tmp_df])
        
        
    re=calculate_randic_energy(G)
    tmp_df=pd.DataFrame({'node': [i[0] for i in re.items()],
                         'value_type': ['randic' for i in re.items()],
                         'value': [i[1] for i in re.items()],
                         'network': [networks_names[i] for j in be.items()]
                        })
    tmp_df['value']=normalize_df_column(tmp_df['value'])
    real_data_measures=pd.concat([real_data_measures,tmp_df])

    ge,_=calculate_graph_energy(G)
    tmp_df=pd.DataFrame({'node': [i[0] for i in ge.items()],
                         'value_type': ['graph' for i in ge.items()],
                         'value': [i[1] for i in ge.items()],
                         'network': [networks_names[i] for j in be.items()]
                        })
    tmp_df['value']=normalize_df_column(tmp_df['value'])
    real_data_measures=pd.concat([real_data_measures,tmp_df])
    real_data_measures.to_pickle('./all_real_networks_calulated_betweenness_and_energy.pickle')

# %%
read_avalilable_datasets_konect()

# %%
