import abc
import numpy as np

from config import epochs_default


class BaseModel(object):

    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    def __str__(self):
        return "Name: %s" % (self.name)

    @abc.abstractmethod
    def train_save(self, tokenizer_func, filename, optimizer='adam', epochs=epochs_default):
        """
        Trains a given model with tokenizer and checkpoints it to a file for later
        :param tokenizer_func: the function to tokenize input strings
        :param filename: the model name (no extension)
        :param optimizer: the optimizer to use during training
        :param int epochs: the number of epochs to repeat during training
        """
        return

    @abc.abstractmethod
    def define_model(self, src_vocab, target_vocab, src_timesteps, target_timesteps, n_units=100):
        return

    @staticmethod
    def offset_data(trainY):
        decoder_target_data = np.zeros_like(trainY)  # same dtype
        for sent_idx, blah in enumerate(trainY):
            for pos_idx, vals in enumerate(blah):
                # where pos_idx is the position in the sentence, 0 = first word
                # and val is the one-hot encoding: 0 or 1
                if pos_idx > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start value.
                    decoder_target_data[sent_idx, pos_idx - 1] = vals
        return decoder_target_data
