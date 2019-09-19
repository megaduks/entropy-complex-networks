import os

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from datetime import date


file_path = os.path.dirname(__file__)


def _load_diagnosis():
    diagnosis_path = os.path.join(file_path, 'mixed/diagnosis.data.txt')
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


def load_diagnosis_inflammation():
    dt, description = _load_diagnosis()
    return dt.drop(['Nephritis', 'Inflammation'], axis=1).values, dt['Inflammation'].values, description


def load_diagnosis_nephritis():
    dt, description = _load_diagnosis()
    return dt.drop(['Nephritis', 'Inflammation'], axis=1).values, dt['Nephritis'].values, description


def load_iris():
    description = ['numerical',
                   'numerical',
                   'numerical',
                   'numerical']
    iris = sklearn.datasets.load_iris()
    X_iris = iris.data.astype(np.float64)
    y_iris = iris.target.astype(np.float64)
    return X_iris, y_iris, description


def load_titanic():
    description = ['numerical',
                   'numerical',
                   'numerical',
                   'numerical',
                   'numerical',
                   'numerical',
                   'numerical']
    titanic_path = os.path.join(file_path, 'mixed/titanic3.xls')
    titanic_df = pd.read_excel(titanic_path, 'titanic3', index_col=None, na_values=['NA'])
    titanic_df = titanic_df.drop(['body', 'cabin', 'boat'], axis=1)
    titanic_df['home.dest'] = titanic_df['home.dest'].fillna('NA')
    titanic_df = titanic_df.dropna()
    processed_df = _preprocess_titanic_df(titanic_df)

    X = processed_df.drop(['survived'], axis=1).values
    y = processed_df['survived'].values

    return X, y, description


def _preprocess_titanic_df(df):
    processed_df = df.copy()
    le = LabelEncoder()
    processed_df.sex = le.fit_transform(processed_df.sex)
    processed_df.embarked = le.fit_transform(processed_df.embarked)
    processed_df = processed_df.drop(['name', 'ticket', 'home.dest'], axis=1)
    return processed_df


def load_lenses():
    description = 'categorical'
    lenses_path = os.path.join(file_path, 'categorical/lenses.data.txt')
    df = pd.read_csv(lenses_path, header=None, delim_whitespace=True)
    df = df.drop(df.columns[0], axis=1)
    return df.drop(df.columns[-1], axis=1).values, df.iloc[:, -1].values, description


def load_mushrooms():
    description = 'categorical'
    mushrooms_path = os.path.join(file_path, 'categorical/agaricus-lepiota.data.txt')
    df = pd.read_csv(mushrooms_path, header=None)
    df = df.apply(LabelEncoder().fit_transform)
    return df.drop(df.columns[0], axis=1).values, df.iloc[:, 0].values, description


def _load_breast_cancer():
    breast_path = os.path.join(file_path, 'numerical/breast-cancer-kaggle.csv')
    df = pd.read_csv(breast_path, header=0)
    df.drop('id', axis=1, inplace=True)
    df.drop('Unnamed: 32', axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df


def load_breast_cancer_short():
    description = 'numerical'
    predictor_var = ['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concave points_mean']
    df = _load_breast_cancer()
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


def load_internet_ads_pca():
    description = 'numerical'
    df = _load_internet_ads()
    sc = StandardScaler()
    X = df.iloc[:, :-1]
    X = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)
    pca = PCA(n_components=4)
    pca.fit(X)
    X = pca.transform(X)
    y = df.iloc[:, -1]
    return X, y.values, description


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


def load_ionosphere():
    description = 'numerical'
    ion_path = os.path.join(file_path, 'numerical/ionosphere.data.txt')
    df = pd.read_csv(ion_path, header=None)

    le = LabelEncoder()
    df.iloc[:, 34] = le.fit_transform(df.iloc[:, 34])

    X = df.iloc[:, 0:33]
    Y = df.iloc[:, 34]

    sc = StandardScaler()
    X = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)

    return X.values, Y.values, description


def load_monks_1():
    description = 'categorical'
    monk_path = os.path.join(file_path, 'categorical/monks-1.csv')
    df = pd.read_csv(monk_path, header=0)
    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]
    return X.values, Y.values, description


def load_monks_2():
    description = 'categorical'
    monk_path = os.path.join(file_path, 'categorical/monks-2.csv')
    df = pd.read_csv(monk_path, header=0)
    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]
    return X.values, Y.values, description


def load_monks_3():
    description = 'categorical'
    monk_path = os.path.join(file_path, 'categorical/monks-3.csv')
    df = pd.read_csv(monk_path, header=0)
    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]
    return X.values, Y.values, description


def load_yeast():
    description = 'numerical'
    yeast_path = os.path.join(file_path, 'numerical/yeast.data.txt')
    df = pd.read_csv(yeast_path, header=None, sep='\s+')

    le = LabelEncoder()
    df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])

    X = df.iloc[:, 1:-1]
    Y = df.iloc[:, -1]

    sc = StandardScaler()
    X = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)
    return X.values, Y.values, description


def load_heart_statlog():
    description = 'numerical'
    path = os.path.join(file_path, 'mixed/heart.csv')
    df = pd.read_csv(path, header=0)

    sc = StandardScaler()
    X = df.iloc[:, :-1]
    X = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)

    Y = df.iloc[:, -1]
    return X.values, Y.values, description


def load_haberman():
    description = 'numerical'
    path = os.path.join(file_path, 'numerical/haberman.csv')
    df = pd.read_csv(path, header=None)

    sc = StandardScaler()
    X = df.iloc[:, :-1]
    X = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)

    Y = df.iloc[:, -1]
    return X.values, Y.values, description


def load_hepatitis():
    description = 'numerical'
    path = os.path.join(file_path, 'mixed/hepatitis.csv')
    df = pd.read_csv(path, header=0)

    sc = StandardScaler()
    X = df.iloc[:, 1:]
    X = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)

    # pca = PCA(n_components=5)
    # pca.fit(X)
    # X = pca.transform(X)

    Y = df.iloc[:, 1]
    return X.values, Y.values, description


def load_dermatology():
    description = ['categorical'] * 34
    description[33] = 'numerical'

    path = os.path.join(file_path, 'mixed/dermatology.data.txt')
    df = pd.read_csv(path, header=None)

    df = df.applymap(lambda val: np.nan if str(val).strip() == '?' else val)
    df = df.dropna()

    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    return X.values, Y.values, description


def load_glass():
    description = 'numerical'

    path = os.path.join(file_path, 'numerical/glass.csv')
    df = pd.read_csv(path, header=0)

    sc = StandardScaler()
    X = df.iloc[:, :-1]
    X = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)

    Y = df.iloc[:, -1]

    return X.values, Y.values, description


def load_ecoli():
    description = 'numerical'

    path = os.path.join(file_path, 'numerical/ecoli.data.txt')
    df = pd.read_csv(path, header=None, sep='\s+')

    # drop annotation
    df.drop(df.columns[0], axis=1, inplace=True)

    le = LabelEncoder()
    df.iloc[:, -1] = le.fit_transform(df.iloc[:, -1])

    sc = StandardScaler()
    X = df.iloc[:, :-1]
    X = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)

    Y = df.iloc[:, -1]

    return X.values, Y.values, description


def load_cmc():
    # description = ['numerical'] * 9
    # description[4] = 'categorical'
    # description[5] = 'categorical'
    # description[6] = 'categorical'
    # description[8] = 'categorical'
    description = 'numerical'

    path = os.path.join(file_path, 'mixed/cmc.data.txt')
    df = pd.read_csv(path, header=None)

    sc = StandardScaler()
    X = df.iloc[:, :-1]
    X = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)

    Y = df.iloc[:, -1]

    return X.values, Y.values, description


def load_zoo():
    description = 'categorical'

    path = os.path.join(file_path, 'categorical/zoo.data.txt')
    df = pd.read_csv(path, header=None)

    # drop annotation
    df.drop(df.columns[0], axis=1, inplace=True)

    X = df.iloc[:, :-1]

    Y = df.iloc[:, -1]

    return X.values, Y.values, description


def load_balance_scale():
    description = 'categorical'

    path = os.path.join(file_path, 'categorical/balance-scale.data.txt')
    df = pd.read_csv(path, header=None)

    le = LabelEncoder()
    df.iloc[:, 0] = le.fit_transform(df.iloc[:, 0])

    X = df.iloc[:, 1:]

    Y = df.iloc[:, 0]

    return X.values, Y.values, description


def load_segmentation():
    description = 'numerical'

    path = os.path.join(file_path, 'numerical/segmentation.data.txt')
    df = pd.read_csv(path, header=0)

    le = LabelEncoder()
    df.iloc[:, 0] = le.fit_transform(df.iloc[:, 0])

    sc = StandardScaler()
    X = df.iloc[:, 1:]
    X = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)

    Y = df.iloc[:, 0]

    return X.values, Y.values, description


def load_car():
    description = 'numerical'

    path = os.path.join(file_path, 'numerical/car.data.txt')
    df = pd.read_csv(path, header=None)

    df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'classes']
    df.buying.replace(('vhigh', 'high', 'med', 'low'), (1, 2, 3, 4), inplace=True)
    df.maint.replace(('vhigh', 'high', 'med', 'low'), (1, 2, 3, 4), inplace=True)
    df.doors.replace(('2', '3', '4', '5more'), (1, 2, 3, 4), inplace=True)
    df.persons.replace(('2', '4', 'more'), (1, 2, 3), inplace=True)
    df.lug_boot.replace(('small', 'med', 'big'), (1, 2, 3), inplace=True)
    df.safety.replace(('low', 'med', 'high'), (1, 2, 3), inplace=True)
    df.classes.replace(('unacc', 'acc', 'good', 'vgood'), (1, 2, 3, 4), inplace=True)

    sc = StandardScaler()
    X = df.iloc[:, :-1]
    X = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)

    Y = df.iloc[:, -1]


    return X.values, Y.values, description


def load_house_voting():
    description = 'categorical'

    path = os.path.join(file_path, 'categorical/house-votes-84.data.txt')
    df = pd.read_csv(path, header=None)

    df = df.dropna()

    le = LabelEncoder()
    df = df.apply(le.fit_transform)

    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]

    return X.values, Y.values, description