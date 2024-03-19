import unittest

import pandas as pd

from GainRatio import GainRatio


class TestGainRatio(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('walk.csv')
        self.gr = GainRatio(self.df, 'Walk')

    def test_gain_ratio(self):
        res = self.gr.gain_ratio()
        self.assertEqual(res, [('Weather', 0.3478230328118343)])

    def test_intrinsic_information(self):
        res = self.gr.intrinsic_information('Weather')
        self.assertEqual(round(res, 2), 0.93)

    def test_entropy(self):
        res = self.gr.entropy(self.df, 'Walk')
        self.assertEqual(round(res, 2), 0.88)

    def test_information_gain(self):
        res = self.gr.information_gain('Weather')
        self.assertEqual(round(res, 2), 0.32)





if __name__ == "__main__":
  unittest.main()