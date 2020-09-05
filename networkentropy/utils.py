from typing import Dict, Tuple, Callable

import numpy as np
import sklearn.datasets
import os
import shutil
import tarfile
import networkx as nx
import requests
import wget
import pandas as pd
import copy
import glob

from collections import namedtuple, UserDict
from itertools import combinations
from typing import List, Iterable, Optional
from bs4 import BeautifulSoup
from bs4.element import ResultSet
from requests import Response, HTTPError
from sklearn.preprocessing import StandardScaler, LabelEncoder

NETWORK_NAME = 'network_name'
CATEGORY = 'category'
DIRECTED = 'directed'
BIPARTITE = 'bipartite'
NUM_NODES = 'num_nodes'
NUM_EDGES = 'num_edges'
TSV_URL = 'tsv_url'

Dataset = namedtuple('Dataset', [NETWORK_NAME, CATEGORY, DIRECTED, BIPARTITE, NUM_NODES, NUM_EDGES, TSV_URL])

file_path = os.path.dirname(__file__)


# TODO: finish typing data loader functions


def precision_at_k(y_true: List, y_pred: List, k: int = 1) -> float:
    """
    Computes precision@k metric for ranking lists

    params:
    :param y_true: list of real ranking of items
    :param y_pred: list of predicted ranking of items
    :param k: cut off value

    :returns the value of the precision@k metric
    """

    assert isinstance(k, int), 'k must be an integer'
    assert (k > 0), 'k must be positive'
    assert isinstance(y_pred, List), 'y_pred must be a list'

    common = set(y_pred[:k]).intersection(set(y_true[:k]))

    return len(common) / k


def gini(x: np.array) -> float:
    """
    Computes the value of the Gini index of a distribution

    params:
    :param x: array with the distribution

    :returns the value of the Gini index
    """

    assert isinstance(x, np.ndarray), 'x must be an array'

    if x.sum() == 0:
        gini_index = 0
    else:

        # Mean absolute difference
        mad = np.abs(np.subtract.outer(x, x)).mean()

        # Relative mean absolute difference
        rmad = mad / np.mean(x)

        # Gini coefficient
        gini_index = 0.5 * rmad

    return gini_index


def theil(x: np.array) -> float:
    """
    Computes the Theil index of the inequality of distribution (https://en.wikipedia.org/wiki/Theil_index)

    params:
    :param x: array with the distribution

    :returns the value of the Theil index of the distribution
    """

    assert isinstance(x, np.ndarray), 'x must be an array'

    mi = x.mean()
    N = len(x)

    if mi == 0:
        theil_index = 0
    else:
        theil_index = (1 / N) * np.nansum((x / mi) * np.log(x / mi))

    return theil_index


def normalize_dict(d: Dict, target: float = 1.0) -> Dict:
    """
    Normalizes the values in the dictionary so that they sum up to factor

    params:
    :param d: dict to be normalized
    :param factor: value to which all values in the dictionary should sum up to

    :returns normalized dictionary
    """

    assert isinstance(d, Dict), 'd must be a dictionary'
    raw = sum(d.values())

    if raw > 0:
        factor = target / raw
    else:
        factor = target

    return {key: value * factor for key, value in d.items()}


def _load_diagnosis() -> Tuple[pd.DataFrame, List]:
    """
    Helper function to load Diagnosis dataset

    :returns dataframe with the dataset and a list of feature type descriptors
    """

    diagnosis_path = os.path.join(file_path, 'data/mixed/diagnosis.data.txt')

    description = ['numerical',
                   'categorical',
                   'categorical',
                   'categorical',
                   'categorical',
                   'categorical']

    names = ['Temperature',
             'Num_nausea',
             'Lumbar_pain',
             'Urine_pushing',
             'Micturition_pains',
             'Burning_urethra',
             'Inflammation',
             'Nephritis']

    types = {'Temperature': np.float64,
             'Num_nausea': 'category',
             'Lumbar_pain': 'category',
             'Urine_pushing': 'category',
             'Micturition_pains': 'category',
             'Burning_urethra': 'category',
             'Inflammation': 'category',
             'Nephritis': 'category'}

    dt = pd.read_csv(diagnosis_path, header=None, names=names, dtype=types, delim_whitespace=True)
    dt_categorical = dt[[i for i in list(dt.columns) if i != 'Temperature']]
    dt[[i for i in list(dt.columns) if i != 'Temperature']] = dt_categorical.apply(
        LabelEncoder().fit_transform)

    return dt, description


def load_diagnosis_inflammation() -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Loads Diagnosis dataset with the Inflammation feature as target

    :returns array-like training set, target feature, and a list of feature type descriptors
    """
    dt, description = _load_diagnosis()

    return dt.drop(['Nephritis', 'Inflammation'], axis=1).values, dt['Inflammation'].values, description


def load_diagnosis_nephritis() -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Loads Diagnosis dataset with the Nephritis feature as target

    :returns array-like training set, target feature, and a list of feature type descriptors
    """
    dt, description = _load_diagnosis()

    return dt.drop(['Nephritis', 'Inflammation'], axis=1).values, dt['Nephritis'].values, description


def load_iris() -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Loads Iris dataset

    :returns array-like training set, target feature, and a list of feature type descriptors
    """
    description = ['numerical',
                   'numerical',
                   'numerical',
                   'numerical']
    iris = sklearn.datasets.load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.float64)

    return X, y, description


def load_titanic() -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Loads Titanic dataset

    :returns array-like training set, target feature, and a list of feature type descriptors
    """

    le = LabelEncoder()

    description = ['numerical',
                   'numerical',
                   'numerical',
                   'numerical',
                   'numerical',
                   'numerical',
                   'numerical']
    titanic_path = os.path.join(file_path, 'data/mixed/titanic3.xls')
    titanic_df = pd.read_excel(titanic_path, 'titanic3', index_col=None, na_values=['NA'])
    titanic_df = titanic_df.drop(['body', 'cabin', 'boat'], axis=1)
    titanic_df['home.dest'] = titanic_df['home.dest'].fillna('NA')
    titanic_df = titanic_df.dropna()
    titanic_df.sex = le.fit_transform(titanic_df.sex)
    titanic_df.embarked = le.fit_transform(titanic_df.embarked)
    titanic_df = titanic_df.drop(['name', 'ticket', 'home.dest'], axis=1)

    X = titanic_df.drop(['survived'], axis=1).values
    y = titanic_df['survived'].values

    return X, y, description


def load_lenses() -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Loads Lenses dataset

    :returns array-like training set, target feature, and a list of feature type descriptors
    """
    description = 'categorical'
    lenses_path = os.path.join(file_path, 'data/categorical/lenses.data.txt')
    df = pd.read_csv(lenses_path, header=None, delim_whitespace=True)
    df = df.drop(df.columns[0], axis=1)

    return df.drop(df.columns[-1], axis=1).values, df.iloc[:, -1].values, description


def load_mushrooms() -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Loads Mushrooms dataset

    :returns array-like training set, target feature, and a list of feature type descriptors
    """
    description = 'categorical'
    mushrooms_path = os.path.join(file_path, 'data/categorical/agaricus-lepiota.data.txt')
    df = pd.read_csv(mushrooms_path, header=None)
    df = df.apply(LabelEncoder().fit_transform)

    return df.drop(df.columns[0], axis=1).values, df.iloc[:, 0].values, description


def load_breast_cancer_short() -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Loads Breast Cancer dataset

    :returns array-like training set, target feature, and a list of feature type descriptors
    """
    description = 'numerical'
    predictor_var = ['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concave points_mean']

    breast_path = os.path.join(file_path, 'data/numerical/breast-cancer-kaggle.csv')
    df = pd.read_csv(breast_path, header=0)
    df.drop('id', axis=1, inplace=True)
    df.drop('Unnamed: 32', axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    X = df[predictor_var].values
    y = df['diagnosis'].values

    return X, y, description


def _load_wine_quality():
    wine_path = os.path.join(file_path, 'data/numerical/winequality-red.csv')
    df = pd.read_csv(wine_path, header=0)
    return df


def load_wine_quality(bins: Tuple = None, groups: List = None) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Loads Wine Quality dataset for regression

    params:
    :param bins: if provided with a tuple, the target feature will be discretized
    :param groups: if provided, the list contains names of bins after discretization

    :returns array-like training set, target feature, and a list of feature type descriptors
    """
    description = 'numerical'

    wine_path = os.path.join(file_path, 'data/numerical/winequality-red.csv')
    df = pd.read_csv(wine_path, header=0)

    if bins and groups:
        df['quality'] = pd.cut(df['quality'], bins=bins, labels=groups)
        df['quality'] = LabelEncoder().fit_transform(df['quality'])

    X = df.drop(['quality'], axis=1)
    y = df['quality']

    return X.values, y.values, description


def load_pima_diabetes():
    description = 'numerical'
    diabetes_path = os.path.join(file_path, 'numerical/diabetes.csv')
    df = pd.read_csv(diabetes_path, header=0)
    X = df.drop(['Outcome'], axis=1)
    y = df['Outcome']
    return X.values, y.values, description


def _load_internet_ads():
    ads_path = os.path.join(file_path, 'numerical/internet-advertisements.csv')
    df = pd.read_csv(ads_path, low_memory=False)

    # remove empties
    df = df.applymap(lambda val: np.nan if str(val).strip() == '?' else val)
    df = df.dropna()

    # map classes
    df['1558'] = df['1558'].map({'ad.': 1, 'nonad.': 0})
    # remove the first column, it's useless
    df = df.iloc[:, 1:].reset_index(drop=True)
    return df


def load_internet_ads_full():
    description = 'numerical'
    df = _load_internet_ads()
    sc = StandardScaler()
    X = df.iloc[:, :-1]
    X = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)
    y = df.iloc[:, -1]
    return X.values, y.values, description


def load_housing_prices_short() -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Loads Melbourne Housing dataset

    :returns array-like training set, target feature, and a list of feature type descriptors
    """
    description = ['numerical'] * 14
    description[1] = 'categorical'

    le = LabelEncoder()
    houses_path = os.path.join(file_path, 'data/mixed/melbourne-housing.csv')

    df = pd.read_csv(houses_path, header=0)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.dropna(inplace=True)

    # change dates to numbers
    min_days = df['Date'].min()
    days_since_start = [(x - min_days).days for x in df['Date']]
    df['Days'] = days_since_start
    df['Type'] = le.fit_transform(df['Type'])

    X = df.drop(['Address', 'Price', 'Date', 'SellerG', 'Suburb', 'Method', 'CouncilArea', 'Regionname'], axis=1)
    y = df['Price']

    return X.values, y.values, description


class NetworkDict(UserDict):
    """
    Intercepts __setitem__ method to assign dict key as graph name
    """

    def __setitem__(self, name, method):
        def wrapper(*args):
            result = method(*args)
            result.graph['name'] = name
            return result

        super(NetworkDict, self).__setitem__(name, wrapper)


class DatasetsStrategy:
    def get_networks_url(self) -> str:
        raise NotImplementedError('Method get_networks_url must be implemented')

    def get_networks_from_response(self, response: Response) -> Iterable[Dataset]:
        raise NotImplementedError('Method get_networks_from_response must be implemented')


class Datasets:
    def __init__(self, datasets_strategy: DatasetsStrategy):
        self.datasets_strategy = datasets_strategy
        self.networks = pd.DataFrame()
        self.current_iter = 0

        response = self._request_networks()
        if response.status_code != 200:
            print('An error occurred while getting data.')
        else:
            networks = self._get_networks_from_response(response)
            self.networks = self._map_to_df(networks)

    def __len__(self):
        return len(self.networks)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_iter == len(self.networks):
            raise StopIteration

        self.current_iter += 1
        return self.networks.iloc[self.current_iter - 1]

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
               bipartite: bool = None,
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
        :param directed: if True, only directed networks are included
        :param bipartite: if True, only bipartite networks are included
        :param min_size: minimum number of nodes required in the network
        :param max_size: maximum number of nodes allowed in the network
        :param min_density: minimum density of network allowed
        :param max_density: maximum density of network allowed
        :param only_downloadable: if True, only datasets available to download will be included
        :return: Modified Datasets object
        """
        args = [categories, directed, bipartite, min_size, max_size, min_density, max_density, only_downloadable]
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
    def _build_query(categories, directed, bipartite, min_size, max_size, min_density, max_density, only_downloadable,
                     base_query=None) -> str:
        query = []
        if base_query is not None:
            query.append(f'({base_query})')
        if directed is not None:
            query.append(f'{DIRECTED} == @directed')
        if bipartite is not None:
            query.append(f'{BIPARTITE} == @bipartite')
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
            lambda s: build_network_from_out_konect(s[NETWORK_NAME], s[TSV_URL], s[DIRECTED], dir_name), axis=1)
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
                networks.append(Dataset(network_name=name,
                                        category=tds[2].div.get('title'),
                                        directed='directed' in tds[2].find_all('img')[2].get('title').lower(),
                                        bipartite='bipartite' in tds[2].find_all('img')[2].get('title').lower(),
                                        num_nodes=int(tds[3].text.replace(',', '')),
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
                networks.append(Dataset(network_name=tds[1].a.get('href').replace('/', ''),
                                        category=tds[2].span.text,
                                        directed='directed' in tds[3].a.img.get('title').lower(),
                                        bipartite='bipartite' in tds[3].a.img.get('title').lower(),
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


def read_available_datasets_konect(name: str = 'konect.cc') -> Datasets:
    datasets = create_datasets(name)
    return datasets


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
    :param directed: is network directed
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

    try:
        g = nx.read_edgelist(out_file, create_using=graph_class, comments='%')
        g.graph['name'] = network_name

        return g

    except TypeError:

        try:
            g = nx.read_weighted_edgelist(out_file, create_using=graph_class, comments='%')
            g.graph['name'] = network_name

            return g

        except TypeError:

            return None


def node_attribute_setter(name: str) -> Callable:
    """
    A simple decorator which assigns an attribute value for each node in the network
    Decorated function must return a dictionary with a scalar value for each key (node)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            _dict = func(*args, **kwargs)
            nx.set_node_attributes(kwargs['graph'], values=_dict, name=name)
            return kwargs['graph']

        return wrapper
    return decorator


def lipschitz_constant(x: Iterable, y: Iterable) -> float:
    """Computes the Lipschitz constant for an  empirical function"""

    assert len(x) == len(y), "Both input lists must be of the same length"
    assert max(y) < np.inf and min(y) > -np.inf, "Function values cannot be infinite"

    return max([(np.abs(y[i] - y[j]) / np.abs(x[i] - x[j])) for (i, j) in combinations(range(len(x)), 2)])
