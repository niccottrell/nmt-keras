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


    def test_offset2(self):

        target_texts = []
        target_characters = set()

        target_text = 'this is a test'

        max_decoder_seq_length = len(target_text)

        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_texts.append(target_text)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

        target_characters = sorted(list(target_characters))

        num_decoder_tokens = len(target_characters)

        target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)])

        len1 = 1 # since we have only one target_text here
        decoder_input_data = np.zeros(
            (len1, max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len1, max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')

        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[0, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[0, t - 1, target_token_index[char]] = 1.

        output = offset_data(decoder_input_data)

        np.testing.assert_array_equal(decoder_target_data, output)