import unittest
from train import offset_data
import numpy as np


class TrainTests(unittest.TestCase):

    def test_offset(self):
        num_sentences = 1
        max_length = 3
        vocab_size = 6
        # make input and target the same size
        input = np.zeros((num_sentences, max_length, vocab_size))
        input[0, 0, 1] = 1 # will be ignored
        input[0, 1, 2] = 1 # second word, vector-dimension 2 has value
        input[0, 1, 5] = 1 # second word, vector-dimension 2 has value
        input[0, 2, 3] = 1 # third word, vector-dimension 3 has value
        target = np.zeros((num_sentences, max_length, vocab_size))
        target[0, 0, 2] = 1  # second word, vector-dimension 2 has value
        target[0, 0, 5] = 1  # second word, vector-dimension 2 has value
        target[0, 1, 3] = 1  # third word, vector-dimension 3 has value
        output = offset_data(input)
        np.testing.assert_array_equal(target, output)
