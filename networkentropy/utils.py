import numpy as np
from typing import List, Dict, Tuple

import os

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import date

file_path = os.path.dirname(__file__)

# TODO: finish typing data loader functions


def precision_at_k(y_true: List, y_pred: List, k: int=1) -> float:
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
    titanic_df = titanic_df.drop(['name','ticket','home.dest'], axis=1)

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
    wine_path = os.path.join(file_path, 'numerical/winequality-red.csv')
    df = pd.read_csv(wine_path, header=0)
    return df


def load_wine_quality_regression():
    description = 'numerical'
    df = _load_wine_quality()
    X = df.drop(['quality'], axis=1)
    y = df['quality']
    return X.values, y.values, description


def load_wine_quality_classification():
    description = 'numerical'
    df = _load_wine_quality()
    bins = (2, 6.5, 8)
    group_names = ['bad', 'good']
    df['quality'] = pd.cut(df['quality'], bins=bins, labels=group_names)
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


def _load_housing_prices():
    houses_path = os.path.join(file_path, 'mixed/melbourne-housing.csv')
    df = pd.read_csv(houses_path, header=0)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.dropna(inplace=True)

    # change dates to numbers
    min_days = df['Date'].min()
    days_since_start = [(x - min_days).days for x in df['Date']]
    df['Days'] = days_since_start
    return df


def load_housing_prices_short():
    description = ['numerical'] * 14
    # field Type
    description[1] = 'categorical'

    df = _load_housing_prices()
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])

    X = df.drop(['Address', 'Price', 'Date', 'SellerG', 'Suburb', 'Method', 'CouncilArea', 'Regionname'], axis=1)
    y = df['Price']
    return X.values, y.values, description

