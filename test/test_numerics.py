import unittest
from unittest import TestCase
from unittest.mock import Mock, patch, call
import numpy as np
from numpy.testing import assert_array_equal

from vb_toolbox import numerics


class Numerics_TestCase(TestCase):

    def test_force_symmetric(self):
        N = np.random.randint(low=2, high=200, size=10)
        for n in N:
            test_arr = np.random.rand(n,n)
            ret = numerics.force_symmetric(test_arr)
            assert_array_equal(np.triu(ret), np.triu(test_arr))
            assert_array_equal(ret, ret.T)

    def test_get_fiedler_eigenpair(self):
        pass

    def test_spectral_reorder(self):
        pass

    def test_create_affinity_matrix(self):
        pass


if __name__ == "__main__":
    unittest.main()
