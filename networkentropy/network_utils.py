import os
import shutil
import tarfile
import networkx as nx
import requests
import wget
import pandas as pd
import copy
import glob

from collections import namedtuple
from typing import List, Iterable, Optional
from urllib.request import HTTPError
from bs4 import BeautifulSoup
from bs4.element import ResultSet
from requests import Response

NAME = 'name'
CATEGORY = 'category'
DIRECTED = 'directed'
NUM_NODES = 'num_nodes'
NUM_EDGES = 'num_edges'
TSV_URL = 'tsv_url'

Dataset = namedtuple('Dataset', [NAME, CATEGORY, DIRECTED, NUM_NODES, NUM_EDGES, TSV_URL])


class DatasetsStrategy:
    def get_networks_url(self) -> str:
        raise NotImplementedError('Method get_networks_url must be implemented')

    def get_networks_from_response(self, response: Response) -> Iterable[Dataset]:
        raise NotImplementedError('Method get_networks_from_response must be implemented')


class Datasets:
    def __init__(self, datasets_strategy: DatasetsStrategy):
        self.datasets_strategy = datasets_strategy
        self.networks = pd.DataFrame()

        response = self._request_networks()
        if response.status_code != 200:
            print('An error occurred while getting data.')
        else:
            networks = self._get_networks_from_response(response)
            self.networks = self._map_to_df(networks)

    def _request_networks(self) -> Response:
        return requests.get(self._get_networks_url())

    def _get_networks_url(self) -> str:
        return self.datasets_strategy.get_networks_url()

    def _get_networks_from_response(self, response: Response) -> Iterable[Dataset]:
        return self.datasets_strategy.get_networks_from_response(response)

    @staticmethod
    def _map_to_df(networks: Iterable[Dataset]) -> pd.DataFrame:
        networks_as_dicts = [d._asdict() for d in networks]
        return pd.DataFrame(networks_as_dicts)

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
               combine_queries: bool = False,
               categories: List[str] = None,
               directed: bool = None,
               min_size: int = None,
               max_size: int = None,
               min_density: int = None,
               max_density: float = None,
               only_downloadable: bool = True) -> 'Datasets':
        """
        Filters datasets
        :param directed: if True, only directed graphs will be returned, if False, only undirected, if None, both types
        :param inplace: specifies whether Dataset should be modified or a modified copy should be returned (default is False)
        :param query_expr: query expression, if specified and combine is False, other arguments are ignored
        :param combine_queries: specifies whether query_expr should be combined with other conditions (default is False)
        :param categories: categories to be included
        :param min_size: minimum number of nodes required in the network
        :param max_size: maximum number of nodes allowed in the network
        :param min_density: minimum density of network allowed
        :param max_density: maximum density of network allowed
        :param only_downloadable: if True, only datasets available to download will be included
        :return: Modified Datasets object
        """
        args = [categories, directed, min_size, max_size, min_density, max_density, only_downloadable]
        if query_expr is None:
            query_expr = self._build_query(*args)
        elif combine_queries:
            query_expr = self._build_query(*args, query_expr)
        if inplace:
            datasets = self
        else:
            datasets = copy.deepcopy(self)
        if query_expr:
            datasets.networks.query(query_expr, inplace=True)
        return datasets

    @staticmethod
    def _build_query(categories, directed, min_size, max_size, min_density, max_density, only_downloadable,
                     base_query=None) -> str:
        query = []
        if base_query is not None:
            query.append(f'({base_query})')
        if directed is not None:
            query.append(f'{DIRECTED} == @directed')
        if categories is not None:
            query.append(f'{CATEGORY} in @categories')
        if min_size is not None:
            query.append(f'{NUM_NODES} >= @min_size')
        if max_size is not None:
            query.append(f'{NUM_NODES} <= @max_size')
        if min_density is not None:
            query.append(f'({NUM_EDGES} / ({NUM_NODES} * ({NUM_NODES} - 1))) >= @min_density')
        if max_density is not None:
            query.append(f'({NUM_EDGES} / ({NUM_NODES} * ({NUM_NODES} - 1))) <= @max_density')
        if only_downloadable:
            # using the fact that NaN != NaN, somehow it works for None values in queries as well
            query.append(f'{TSV_URL} == {TSV_URL}')
        return ' and '.join(query)

    def download_and_build_networks(self, dir_name='data/') -> pd.DataFrame:
        """
        Downloads networks data and creates NetworkX graph objects from the data
        :param dir_name: name of the directory to download files to
        :return: DataFrame of NetworkX graph objects (DataFrame may contain None for graphs that couldn't be downloaded
        """
        networks = self.networks.apply(
            lambda s: build_network_from_out_konect(s[NAME], s[TSV_URL], s[DIRECTED], dir_name), axis=1)
        return self.networks.assign(graph=networks)


class KonectCCStrategy(DatasetsStrategy):
    networks_url = 'http://konect.cc/networks/'

    def get_networks_url(self) -> str:
        return self.networks_url

    def get_networks_from_response(self, response: Response) -> List[Dataset]:
        html = response.content
        soup = BeautifulSoup(html, 'lxml')

        table_html = soup.find('table')
        rows = table_html.findAll('tr')

        networks = []
        for row in rows[1:]:
            if row:
                tds = row.find_all('td')
                name = tds[1].a.get('href').replace('/', '')
                tsv_url = self._get_tsv_url(name, tds)
                networks.append(Dataset(name=name,
                                        category=tds[2].div.get('title'),
                                        nun_nodes=int(tds[3].text.replace(',', '')),
                                        num_edges=int(tds[4].text.strip('\n').replace(',', '')),
                                        tsv_url=tsv_url))
        return networks

    @staticmethod
    def _get_tsv_url(name: str, tds: ResultSet) -> Optional[str]:
        download_icon_title = tds[2].img.get('title')
        if 'is not available' in download_icon_title:
            return None
        else:
            return f'http://konect.cc/files/download.tsv.{name}.tar.bz2'


class KonectUniStrategy(DatasetsStrategy):

    def get_networks_url(self) -> str:
        return 'http://konect.uni-koblenz.de/networks/'

    def get_networks_from_response(self, response: Response) -> List[Dataset]:
        html = response.content
        soup = BeautifulSoup(html, 'lxml')

        table_html = soup.find('table')
        rows = table_html.tbody.find_all('tr')

        networks = []
        for row in rows:
            if row:
                tds = row.find_all('td')
                tsv_url = self._get_tsv_url(tds)
                networks.append(Dataset(name=tds[1].a.get('href').replace('/', ''),
                                        category=tds[2].span.text,
                                        directed=tds[3].a.img.get('title').startswith('Directed'),
                                        num_nodes=int(tds[6].text.replace(',', '')),
                                        num_edges=int(tds[7].text.replace(',', '')),
                                        tsv_url=tsv_url))
        return networks

    @staticmethod
    def _get_tsv_url(tds: ResultSet):
        a = tds[8].find_all('div')[1].a
        if a is not None:
            relative_url = a.get('href').replace('../', '')
            return f'http://konect.uni-koblenz.de/{relative_url}'
        else:
            return None


def create_datasets(name: str):
    if name == 'konect.cc':
        strategy = KonectCCStrategy()
    elif name == "konect.uni":
        strategy = KonectUniStrategy()
    else:
        raise ValueError(f'Strategy with name {name} does not exist')
    return Datasets(strategy)


def read_available_datasets_konect(name: str = 'konect.cc') -> List[object]:
    datasets = create_datasets(name)
    return datasets.to_list()


def download_tsv_dataset_konect(output_file_name: str, tsv_url: str, dir_name: str) -> Optional[str]:
    """
    Downloads the compressed network file into local directory

    :param output_file_name: name of the network to download
    :param tsv_url: url to network data as tsv
    :param dir_name: name of the local directory to which the compressed file should be saved
    :return: name of the downloaded file
    """
    output_file = f'{output_file_name}.tar.bz2'
    try:

        file_name = wget.download(tsv_url, out=output_file)

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        if os.path.exists(output_file):
            shutil.move(file_name, dir_name + output_file)
    except HTTPError:
        return None
    return output_file


def unpack_tar_bz2_file(file_name: str, dir_name: str, output_dir_name: str):
    """
    Unpacks the downloaded compressed file on disk

    :param output_dir_name: name of the directory to which file will be extracted
    :param file_name: name of the compressed file
    :param dir_name: name of the directory in which unpacking happens
    :return: name of the directory where unpacked files are
    """

    tar = tarfile.open(dir_name + file_name, 'r:bz2')
    tar.extractall(output_dir_name)
    tar.close()


def build_network_from_out_konect(network_name: str, tsv_url: str, directed: bool, dir_name: str) -> Optional[nx.Graph]:
    """
    Reads network files stored on disk and builds a proper NetworkX graph object

    :param network_name: name of the network to build
    :param tsv_url: url to network data as tsv
    :param dir_name: name of the directory to download files to
    :return: NetworkX graph object, or None if the network is too large
    """

    output_dir_name = f'{dir_name}network_{network_name}/'

    if not os.path.exists(output_dir_name):
        file_name = download_tsv_dataset_konect(output_file_name=network_name,
                                                tsv_url=tsv_url,
                                                dir_name=dir_name)
        # network could not be downloaded
        if not file_name:
            return None
        unpack_tar_bz2_file(file_name=file_name,
                            dir_name=dir_name,
                            output_dir_name=output_dir_name)

    out_file = next(glob.iglob(f'{output_dir_name}/**/out.*', recursive=True))
    assert out_file, 'No out. file in the directory.'
    if directed:
        graph_class = nx.DiGraph
    else:
        graph_class = nx.Graph
    g = nx.read_adjlist(out_file, create_using=graph_class, comments='%')
    return g


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
