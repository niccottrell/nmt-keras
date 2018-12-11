import unittest
from helpers import *
import numpy as np

class HelpersTests(unittest.TestCase):

    # first word lowercased since it's not a proper noun
    def test_encode_sequences_fixed(self):
        lines = [['The', 'cat', 'sat'], ['The', 'dog', 'barked']]
        tokenizer = create_tokenizer(lines) # more common words get higher indexes
        sequences = encode_sequences(tokenizer, lines, 4)
        np.testing.assert_array_equal([1, 2, 3, 0], sequences[0])
        np.testing.assert_array_equal([1, 4, 5, 0], sequences[1])

    def test_encode_sequences_unlimited(self):
        lines = [['The', 'cat', 'sat'], ['The', 'dog', 'barked']]
        tokenizer = create_tokenizer(lines)
        sequences = encode_sequences(tokenizer, lines)
        np.testing.assert_array_equal([1, 2, 3], sequences[0])
        np.testing.assert_array_equal([1, 4, 5], sequences[1])



