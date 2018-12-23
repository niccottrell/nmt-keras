import abc
import numpy as np
from keras.preprocessing.text import Tokenizer

from config import epochs_default
from helpers import word_for_id, encode_sequences, lang2, create_tokenizer, max_length, simple_lines


class BaseModel(object):

    model = None
    optimizer = 'adam'
    tokenizer_func = simple_lines

    def __init__(self, name, tokenizer_func, optimizer):
        """
        :param name: the model name (no extension), e.g. "let2let_a_adam_201812"
        :param tokenizer_func: the function to tokenize input strings
        :param optimizer: the optimizer to use during training, eg. "adam"

        """
        self.name = name
        self.tokenizer_func = tokenizer_func
        self.optimzer = optimizer

    def get_name(self):
        return self.name

    def __str__(self):
        return "Name: %s" % (self.name)

    def translate(self, source):
        """
        :param model: Model
        :param tokenizer: string
        :param source: the input natural language
        :type source: string
        :return: string the most likely translation as a string
        """
        return self.predict_sequence(source)

    @abc.abstractmethod
    def train_save(self, epochs=epochs_default):
        """
        Trains a given model with tokenizer and checkpoints it to a file for later
        :param  epochs: the number of epochs to repeat during training
        """
        return

    @abc.abstractmethod
    def define_model(self, src_vocab, target_vocab, src_timesteps, target_timesteps, n_units=100):
        return

    def predict_sequence(self, source):
        """
        generate target given source sequence (for all predictions)
        :type model: Model
        :type tokenizer: Tokenizer
        :type source: The input data
        :return the most likely prediction
        """
        # prepare/encode/pad data (pad to length of target language)

        pad_length = self.other_length if self.name.startswith('simple') else None

        testX = encode_sequences(self.tokenizer, self.tokenizer_func(list(source), lang2), pad_length)

        translations = list()
        predictions = self.model.predict(source, verbose=0)
        for preduction in predictions:
            integers = [np.argmax(vector) for vector in preduction]
            target = list()
            for i in integers:
                word = word_for_id(i, self.tokenizer)
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

    def update(self, dataset):
        """
        :type dataset: ndarray
        """
        dataset_lang2 = dataset[:, 1]
        self.other_tokenizer = create_tokenizer(self.tokenizer_func(dataset_lang2, lang2))
        other_tokenized = self.tokenizer_func(dataset_lang2, lang2)
        self.other_length = max_length(other_tokenized)
        # prepare/encode/pad data (pad to length of target language)
        self.pad_length = self.other_length if self.name.startswith('simple') else None
