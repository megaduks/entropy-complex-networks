import numpy as np

SIGNATURE_SIZE = 64

def encode(M: np.array) -> np.array:
    """
    Encodes an input array M using hierarchical bitmap compression

    :param M: input array
    :return: hierarchical bitmap encoding of input array
    """

    n_rows, n_columns = M.shape

    pass


def arr2int(a: np.array, signature_size: int = SIGNATURE_SIZE) -> int:
    """
    Encodes a single signature represented as a binary array into an integer

    :param a: input array
    :param signature_size: size of a single signature
    :return: integer representation of an array
    """

    assert signature_size <= SIGNATURE_SIZE, f"Size of binary signature cannot be larger than {SIGNATURE_SIZE}"
    assert len(a) <= signature_size, f"Input array size cannot be larger than {SIGNATURE_SIZE}"
    assert np.unique(a).tolist() == [0,1], f"Input array must be binary"

    str_array = ''.join(map(str, a))
    int_value_of_array = int(str_array, base=2)

    return int_value_of_array