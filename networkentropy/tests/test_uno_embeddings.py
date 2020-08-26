import unittest

from networkentropy.embeddings import extract_context_pairs, sliding


class TestUnoEmbeddings(unittest.TestCase):

    def test_sliding_window(self):
        _lst = list(range(1, 10))
        result = sliding(iterable=_lst, n=4)
        expected = [
            (1, 2, 3, 4),
            (2, 3, 4, 5),
            (3, 4, 5, 6),
            (4, 5, 6, 7),
            (5, 6, 7, 8),
            (6, 7, 8, 9),
        ]
        self.assertEqual(sorted(list(expected)), sorted(result))

    def test_path_to_bon(self):
        path = list(range(1, 7))
        result = extract_context_pairs(path=path, context_size=2)
        expected = [
            (1, 3), (2, 3), (3, 4), (3, 5),
            (2, 4), (3, 4), (4, 5), (4, 6),
        ]
        self.assertEqual(sorted(expected), sorted(result))


if __name__ == '__main__':
    unittest.main()
