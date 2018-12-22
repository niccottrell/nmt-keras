import abc
import numpy as np
from keras.preprocessing.text import Tokenizer

from config import epochs_default
from helpers import word_for_id


class BaseModel(object):

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def __str__(self):
        return "Name: %s" % (self.name)

    def translate(self, model, tokenizer, source):
        """
        :param model: Model
        :param tokenizer: string
        :param source: list(int)
        :return: string the most likely translation as a string
        """
        return self.predict_sequence(model, tokenizer, source)

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

    def predict_sequence(self, model, tokenizer, source):
        """
        generate target given source sequence (for all predictions)
        :type model: Model
        :type tokenizer: Tokenizer
        :type source: The input data
        :return the most likely prediction
        """
        translations = list()
        predictions = model.predict(source, verbose=0)
        for preduction in predictions:
            integers = [np.argmax(vector) for vector in preduction]
            target = list()
            for i in integers:
                word = word_for_id(i, tokenizer)
                if word is None:
                    break
                target.append(word)
            translations.append(' '.join(target))
        print("Candidates=", translations)
        return translations[0]

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
