from networkentropy import __networks__

import pandas as pd
import numpy as np
import networkx as nx
import urllib.request
import tarfile
import zipfile


def get_network_metadata(network_name):
    """
    Reads network metadata from the JSON file stored in the repository

    :param network_name: name of the network
    :return: dictionary with metadata items as keys and values
    """
    network_data = __networks__[network_name]
    return network_data

def get_network_stats(g):
    """
    Compute basic properties of a network

    :param g: input network as an NetworkX graph
    :return: dictionary with basic network properties as keys
    """
    result = {}

    result['num_nodes'] = nx.number_of_nodes(g)
    result['num_edges'] = nx.number_of_edges(g)
    result['transitivity'] = nx.transitivity(g)

    if nx.is_directed(g):
        if nx.is_weakly_connected(g):
            result['average_shortest_path'] = nx.average_shortest_path_length(g)
        if nx.is_strongly_connected(g):
            result['diameter'] = nx.diameter(g)

    else:
        result['average_shortest_path'] = nx.average_shortest_path_length(g)
        result['diameter'] = nx.diameter(g)

    result['reciprocity'] = nx.reciprocity(g)

    return result

def load_python_dependency(u):
    """
    Loads the Python Dependency network of packages

    :param u: URL of the source file with nodes and edges
    :return: network
    """

    response = urllib.request.urlopen(u)
    df = pd.read_csv(response)

    g = nx.DiGraph()
    g.add_nodes_from(df.package_name.unique())
    edges = df.loc[df.requirement.notnull(), ['package_name', 'requirement']].values
    g.add_edges_from(edges)

    g.remove_nodes_from(['.', 'nan', np.nan])

    degree_list = g.degree()
    to_remove = [node for (node, degree) in degree_list if degree <= 0]
    g.remove_nodes_from(to_remove)

    return g

def load_r_dependency(u):
    """
    Loads the R Dependency network of packages

    :param u: URL of the source file with nodes and edges
    :return: network
    """

    response = urllib.request.urlopen(u)
    df = pd.read_csv(response)
    df.columns = ['id','from_vertex','to_vertex']

    # remove the ID column
    df.drop('id', axis=1, inplace=True)
    # remove the empty last row of the csv file
    df.drop(df.tail(1).index, axis=0, inplace=True)

    g = nx.DiGraph()

    g.add_edges_from(df.values)

    return g

def load_st_mark_ecosystem(u):
    """
    Loads the St.Mark National Wildlife Refuge food dependency network

    :param u: URL of the source file with nodes and edges
    :return: network
    """
    _tmpfile = '/tmp/stmark.net'

    local_file, headers = urllib.request.urlretrieve(u, _tmpfile)
    lines = open(local_file).readlines()

    # read only the *.net part of Pajek's *.paj project file
    open(_tmpfile, "w").writelines(lines[67:480])

    g = nx.read_pajek(_tmpfile)

    return g

def load_power_grid(u):
    """
    Loads the Western US Power Grid network from the GML file

    :param u: URL of the source file with nodes and edges
    :return: network
    """

    local_file, headers = urllib.request.urlretrieve(u, '/tmp/powergrid')
    filename = 'opsahl-powergrid/out.opsahl-powergrid'

    tar = tarfile.open(local_file, mode="r:bz2")
    network_file = tar.extractfile(filename)
    # tar.close()

    g = nx.read_edgelist(network_file, comments='%')

    return g

def load_celegans(u):
    """
    Loads the metabolic network of the nematode C. elegans

    :param u: URL of the source file with nodes and edges
    :return: network
    """

    local_file, headers = urllib.request.urlretrieve(u, '/tmp/celegans')
    filename = 'celegans_metabolic.net'

    zip = zipfile.ZipFile(local_file, mode="r")
    network_file = zip.extract(filename)
    # tar.close()

    g = nx.read_pajek(network_file)

    return g

def load_cat_brain(u):
    """
    Loads the network of interactions among cortical regions in the cat brain,

    :param u: URL of the source file with nodes and edges
    :return: network
    """

    response = urllib.request.urlopen(u)

    g = nx.read_graphml(response)

    return g

def load_bisons(u):
    """
    Loads the dominance relations among a group of American bisons in the National Bison Range

    :param u: URL of the source file with nodes and edges
    :return: network
    """

    local_file, headers = urllib.request.urlretrieve(u, '/tmp/bisons')
    filename = 'moreno_bison/out.moreno_bison_bison'

    tar = tarfile.open(local_file, mode="r:bz2")
    network_file = tar.extractfile(filename)

    g = nx.read_weighted_edgelist(network_file, comments='%', create_using=nx.DiGraph())

    return g

def load_network(network_name):
    """
    Loads a network based on the metadata stored in a JSON file in the repository

    :param network_name: name of the network
    :return: network
    """

    network_data = __networks__[network_name]
    network_data_url = network_data['url']

    if network_name == 'python_dependency':
        network = load_python_dependency(network_data_url)
    if network_name == 'power_grid':
        network = load_power_grid(network_data_url)
    if network_name == 'R_dependency':
        network = load_r_dependency(network_data_url)
    if network_name == 'st_mark_ecosystem':
        network = load_st_mark_ecosystem(network_data_url)
    if network_name == 'celegans':
        network = load_celegans(network_data_url)
    if network_name == 'cat_brain':
        network = load_cat_brain(network_data_url)
    if network_name == 'bisons':
        network = load_bisons(network_data_url)

    return network


if __name__ == "__main__":

    g = load_network('bisons')

    print()
    print('number of nodes: ', g.number_of_nodes())
    print('number of edges: ', g.number_of_edges())

    gstats = get_network_stats(g)

    for k in gstats.keys():
        print(k, gstats[k])