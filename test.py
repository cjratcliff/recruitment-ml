import unittest

from main import preprocess

class TestPreprocessing(unittest.TestCase):

    def test_dashes(self):
        self.assertEqual(preprocess("Sorrow came--a gentle sorrow--but"), "sorrow came -- a gentle sorrow -- but")

    def test_newlines(self):
        self.assertEqual(preprocess("the poet\n    of wickedness"), "the poet of wickedness")

if __name__ == '__main__':
    unittest.main()