import numpy as np
from .concat import weighted_average_algorithm


class Measure(object):
    """
    Measure:
    Objects of class Measure are responsible for measuring and concatenation of distances between examples
    Args:
        numerical_strategy:
            numerical strategy function
        categorical_strategy:
            categorical strategy function
    """
    def __init__(self, numerical_strategy, categorical_strategy, concat_strategy=weighted_average_algorithm):
        self.numerical_algorithm = numerical_strategy
        self.categorical_algorithm = categorical_strategy
        self.concat_algorithm = concat_strategy

    def compute(self, x_data, column_descriptions=[]):
        """
        Args:
            x_data:
                Array with records
            column_descriptions:
                Pass "categorical" or "numerical" to specify type of variables.
                Pass array with those names in case mixed variables.
                By default, it presumes "numerical"
        """
        # only numerical
        if not column_descriptions or column_descriptions == "numerical" or "categorical" not in column_descriptions:
            # normalize numerical attributes to <0;1>
            num_matrix = self.numerical_algorithm(x_data)
            num_matrix = (num_matrix - np.min(num_matrix)) / (np.max(num_matrix) - np.min(num_matrix))
            # convert distance to simalarity
            return 1 - num_matrix
        # only categorical
        elif column_descriptions == "categorical" or "numerical" not in column_descriptions:
            # already similarity
            return self.categorical_algorithm(x_data)
        # mixed, use column_description array
        else:
            column_descriptions = np.array([c.lower() for c in column_descriptions])
            return self.concat_algorithm(x_data, column_descriptions, self.numerical_algorithm, self.categorical_algorithm)

