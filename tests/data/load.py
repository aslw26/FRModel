import unittest

from matplotlib import pyplot as plt

from frmodel.data.load import load_spec
from tests.path import RSC_PATH


class TestClassification(unittest.TestCase):

    def test_load(self):
        f, trees = load_spec(RSC_PATH + "imgs/spec/chestnut/18Dec2020/", 1)
        for tree in trees:
            assert tree.frame.size() > 0


if __name__ == '__main__':
    unittest.main()