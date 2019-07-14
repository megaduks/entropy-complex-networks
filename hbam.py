import numpy as np
from typing import Tuple

SIGNATURE_SIZE = 64

#TODO: add code to compute the algorithmic complexity of a string
#TODO: add code to shuffle arrays and compare algorithmic complexities


def complexity(M: np.array, signature_size: int = SIGNATURE_SIZE) -> Tuple[float, np.array]:
    """
    Encodes an input array M using hierarchical bitmap compression

    :param M: input array
    :param signature_size: size of a single signature
    :return: encoding complexity and hierarchical bitmap encoding of the input array
    """
    #TODO: add code to change adjacency matrix into sequence of integers
    #TODO: add tests for encode() function

    n_rows, n_columns = M.shape
    M = M.reshape(n_rows*n_columns,)
    _M = unbinarize(M, signature_size=signature_size)

    hbam_encoding = seq2hbseq(_M, signature_size=signature_size)

    original_length = len(M)
    hbam_encoding_length = len(hbam_encoding)

    return hbam_encoding_length / original_length, hbam_encoding


def unbinarize(a: np.array, signature_size: int = SIGNATURE_SIZE) -> np.array:
    """
    Converts a binary array into an array of integers

    :param a: binary array
    :param signature_size: size of a single signature
    :return: output array
    """

    # length of the input array must be the multiple of the signature size
    if len(a) % signature_size:
        a = np.append(a, np.zeros(signature_size - len(a) % signature_size))
    a = a.reshape(len(a) // signature_size, signature_size).astype(int)
    result = np.apply_along_axis(arr2int, axis=1, arr=a, signature_size=signature_size)

    return result


def binarize(a: np.array) -> np.array:
    """
    Converts an array of integers into a binary array

    :param a: input array
    :return: binary array
    """

    return a.astype(bool).astype(int)


def arr2int(a: np.array, signature_size: int = SIGNATURE_SIZE) -> int:
    """
    Encodes a single signature represented as a binary array into an integer

    :param a: input array
    :param signature_size: size of a single signature
    :return: integer representation of an array
    """

    assert signature_size <= SIGNATURE_SIZE, f"Size of binary signature cannot be larger than {SIGNATURE_SIZE}"
    assert len(a) <= signature_size, f"Input array size cannot be larger than {SIGNATURE_SIZE}"
    assert np.unique(a).tolist() in [[0], [1], [0,1]], f"Input array must be binary"

    str_array = ''.join(map(str, a))
    int_value_of_array = int(str_array, base=2)

    return int_value_of_array


def seq2hbseq(a: np.array, signature_size: int = SIGNATURE_SIZE) -> np.array:
    """
    Converts a sequence of integers into a hierarchical bitmap sequence

    :param a: input array
    :param signature_size: size of a single signature
    :return:  array of ints forming the condensed hierarchical bitmap sequence
    """

    #TODO: add tests for this method

    # length of the input array must be the multiple of the signature size
    a = np.append(a, np.zeros(signature_size - len(a) % signature_size))

    result = np.empty(0)

    current_level = a

    while len(current_level) >= signature_size:

        result = np.insert(result, 0, current_level)

        if len(current_level) % signature_size:
            current_level = np.append(current_level, np.zeros(signature_size - len(current_level) % signature_size))
        current_level = current_level.reshape(len(current_level) // signature_size, signature_size)

        if len(current_level) >= signature_size:
            next_level = np.apply_along_axis(binarize, axis=1, arr=current_level)
        else:
            next_level = binarize(current_level)

        if len(next_level) % signature_size and len(next_level) > 1:
            next_level = np.append(next_level,
                                   np.zeros((signature_size - len(next_level) % signature_size, signature_size)))
            next_level = next_level.reshape(len(next_level) // signature_size, signature_size).astype(int)

        next_level = np.apply_along_axis(arr2int, axis=1, arr=next_level, signature_size=signature_size)
        current_level = next_level

    result = np.insert(result, 0, current_level)

    return result[result > 0].astype(int)
