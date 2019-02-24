import os
import shutil
import tarfile
from typing import List
from urllib.request import HTTPError

import networkx as nx
import requests
import wget
import pandas as pd
import copy
from bs4 import BeautifulSoup

NAME = "name"
CATEGORY = "category"
NUM_NODES = "num_nodes"
NUM_EDGES = "num_edges"


class DatasetsStrategy:
    def get_networks_url(self) -> str:
        raise NotImplementedError("Method get_networks_url must be implemented")

    def get_networks_from_response(self, response) -> List[object]:
        raise NotImplementedError("Method get_networks_from_response must be implemented")


class Datasets:
    def __init__(self, datasets_strategy: DatasetsStrategy):
        self.datasets_strategy = datasets_strategy
        self.networks = pd.DataFrame()

        response = self._request_networks()
        if response.status_code != 200:
            print("An error occurred while getting data.")
        else:
            networks = self._get_networks_from_response(response)
            self.networks = self._map_to_dataframe(networks)

    def _request_networks(self):
        return requests.get(self._get_networks_url())

    def _get_networks_url(self):
        return self.datasets_strategy.get_networks_url()

    def _get_networks_from_response(self, response):
        return self.datasets_strategy.get_networks_from_response(response)

    def _map_to_dataframe(self, networks) -> pd.DataFrame:
        transposed_networks = list(map(list, zip(*networks)))
        return pd.DataFrame(data={
            NAME: transposed_networks[0],
            CATEGORY: transposed_networks[1],
            NUM_NODES: transposed_networks[2],
            NUM_EDGES: transposed_networks[3]
        })

    def to_list(self) -> List[List[object]]:
        """
        Returns datasets information as list
        :return: datasets information as list
        """
        return self.networks.values.tolist()

    def to_df(self) -> pd.DataFrame:
        """
        Returns datasets information as DataFrame
        :return: datasets information as DataFrame
        """
        return self.networks

    def filter(self,
               inplace: bool = False,
               query_expr: str = None,
               categories: List[str] = None,
               min_size: int = None,
               max_size: int = None,
               min_density: int = None,
               max_density: float = None) -> 'Datasets':
        """
        :param inplace: specifies whether Dataset should be modified or a modified copy should be returned
        :param query_expr: query expression, if specified, other arguments are ignored
        :param categories: categories to be included
        :param min_size: minimum number of nodes required in the network
        :param max_size: maximum number of nodes allowed in the network
        :param min_density: minimum density of network allowed
        :param max_density: maximum density of network allowed
        :return: Modified Datasets object
        """
        if query_expr is None:
            query_expr = self._build_query(categories, min_size, max_size, min_density, max_density)
        if not query_expr:
            raise ValueError("Either query_expr or other filtering parameter must be specified")
        if inplace:
            datasets = self
        else:
            datasets = copy.deepcopy(self)
        datasets.networks.query(query_expr, inplace=True)
        return datasets

    def _build_query(self, categories, min_size, max_size, min_density, max_density) -> str:
        query = []
        if categories is not None:
            query.append("{} in @categories".format(CATEGORY))
        if min_size is not None:
            query.append("{} >= @min_size".format(NUM_NODES))
        if max_size is not None:
            query.append("{} <= @max_size".format(NUM_NODES))
        if min_density is not None:
            query.append("({m} / ({n} * ({n} - 1))) >= @min_density".format(m=NUM_EDGES, n=NUM_NODES))
        if max_density is not None:
            query.append("({m} / ({n} * ({n} - 1))) <= @max_density".format(m=NUM_EDGES, n=NUM_NODES))
        return " and ".join(query)


class KonnctCCStrategy(DatasetsStrategy):
    networks_url = "http://konect.cc/networks/"

    def get_networks_url(self) -> str:
        return self.networks_url

    def get_networks_from_response(self, response) -> List[object]:
        html = response.content
        soup = BeautifulSoup(html, "lxml")

        table_html = soup.find('table')
        rows = table_html.findAll('tr')

        networks = [
            (row.find_all('td')[1].a.get('href').replace('/', ''),
             row.find_all('td')[2].div.get('title'),
             int(row.find_all('td')[3].text.replace(',', '')),
             int(row.find_all('td')[4].text.strip('\n').replace(',', ''))
             )
            for row
            in rows[1:]
            if row
        ]
        return networks


def create_datasets(name):
    if name == "konect.cc":
        strategy = KonnctCCStrategy()
    else:
        raise ValueError("Strategy with {} does not exist".format(name))
    return Datasets(strategy)


def read_available_datasets_konect(name="konect.cc") -> List[object]:
    datasets = create_datasets(name)
    return datasets.to_list()


def download_tsv_dataset_konect(network_name: str,
                                dir_name: str,
                                num_nodes: int = None,
                                num_edges: int = None,
                                min_size: int = None,
                                max_size: int = None,
                                max_density: float = None) -> str:
    """
    Downloads the compressed network file into local directory

    :param network_name: name of the network to download
    :param dir_name: name of the local directory to which the compressed file should be saved
    :param num_nodes: number of vertices in the network
    :param num_edges: number of edges in the network
    :param min_size: minimum number of nodes required in the network
    :param max_size: maximum number of nodes allowed in the network
    :param max_density: maximum density of network allowed
    :return: name of the downloaded file
    """

    assert (network_name in [name for (name, cat, vsize, esize) in read_available_datasets_konect()]), \
        "No network named: '" + network_name + "' found in Konect!"

    if min_size:
        if num_nodes < min_size:
            return None
    if max_size:
        if num_nodes > max_size:
            return None
    if max_density:
        if num_edges / (num_nodes * (num_nodes - 1)) > max_density:
            return None

    tsv_file = 'http://konect.cc/files/download.tsv.' + network_name + '.tar.bz2'
    output_file = network_name + '.tar.bz2'

    try:
        file_name = wget.download(tsv_file, out=output_file)

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        if os.path.exists(output_file):
            shutil.move(file_name, dir_name + output_file)
    except HTTPError:
        return None

    return output_file


def unpack_tar_bz2_file(file_name: str, dir_name: str) -> str:
    """
    Unpacks the downloaded compressed file on disk

    :param file_name: name of the compressed file
    :param dir_name: name of the directory in which unpacking happens
    :return: name of the directory where unpacked files are
    """

    tar = tarfile.open(dir_name + file_name, "r:bz2")
    output_dir = dir_name + "network_" + file_name.replace('.tar.bz2', '') + "/"

    tar.extractall(output_dir)
    tar.close()

    return output_dir + file_name.replace('.tar.bz2', '/')


def build_network_from_out_konect(network_name: str,
                                  dir_name: str,
                                  num_nodes: int = None,
                                  num_edges: int = None,
                                  min_size: int = None,
                                  max_size: int = None,
                                  max_density: float = None) -> nx.Graph:
    """
    Reads network files stored on disk and builds a proper NetworkX graph object

    :param network_name: name of the network to build
    :param dir_name: name of the directory to download files to
    :param num_nodes: number of vertices in the network
    :param num_edges: number of edges in the network
    :param min_size: minimum number of nodes required in the network
    :param max_size: maximum number of nodes allowed in the network
    :param max_density: maximum density of network allowed
    :return: NetworkX graph object, or None if the network is too large
    """

    kwargs = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'min_size': min_size,
        'max_size': max_size,
        'max_density': max_density
    }

    file_name = download_tsv_dataset_konect(network_name=network_name,
                                            dir_name=dir_name,
                                            **kwargs)

    # if one of network parameters exceeds the limit
    if not file_name:
        return None

    output_dir = unpack_tar_bz2_file(file_name=file_name, dir_name=dir_name)

    files = [
        file
        for file
        in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, file))
    ]

    out_file = next(filter(lambda x: 'out.' in x, files), None)

    assert (out_file), 'No out. file in the directory.'

    G = nx.read_adjlist(output_dir + out_file, comments='%')

    return G


def precision_at_k(y_true, y_pred, k=1):
    """
    Computes precision@k metric for ranking lists

    params:
    :param y_true: list of real ranking of items
    :param y_pred: list of predicted ranking of items
    :param k: cut off value


    """

    assert isinstance(k, int), 'k must be an integer'
    assert (k > 0), 'k must be positive'
    assert isinstance(y_pred, List), 'y_pred must be a list'

    common = set(y_pred[:k]).intersection(set(y_true[:k]))
    return len(common) / k
