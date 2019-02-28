import abc
import json
import time

import numpy as np
from keras.callbacks import ModelCheckpoint, Callback

from config import epochs_default
from helpers import word_for_id, lang2, create_tokenizer, max_length, load_clean_sentences

import tokenizers


class BaseModel(object):

    model = None # The keras.Model
    other_tokenizer = None  # The keras.preprocessing.text.Tokenizer for mapping Swedish tokens to integers
    eng_tokenizer = None  # The keras.preprocessing.text.Tokenizer for mapping English tokens to integers
    optimizer = 'adam'
    tokenizer = None

    def __init__(self, name, tokenizer, optimizer):
        """
        :param name: the model name (no extension), e.g. "let2let_a_adam_201812"
        :param tokenizer: the function to tokenize input strings
        :type tokenizer: tokenizers.BaseTokenizer
        :param optimizer: the optimizer to use during training, eg. "adam"

        """
        self.name = name
        self.tokenizer = tokenizer
        self.optimizer = optimizer

        # load datasets
        dataset = load_clean_sentences( 'both')

        print("## " + tokenizer.__class__.__name__)
        print("Prepare English tokenizer")
        self.eng_texts = dataset[:, 0] # English
        self.eng_tokenized = tokenizer.tokenize(self.eng_texts, 'en')
        self.eng_tokenizer = create_tokenizer(self.eng_tokenized)
        self.eng_vocab_size = len(self.eng_tokenizer.word_index) + 1
        self.eng_length = max_length(self.eng_tokenized)
        print('English Vocabulary Size: %d' % self.eng_vocab_size)
        print('English Max Length: %d' % self.eng_length)

        print("Prepare other language tokenizer")
        self.other_texts = dataset[:, 1]
        self.other_tokenized = tokenizer.tokenize(self.other_texts, lang2)
        self.other_tokenizer = create_tokenizer(self.other_tokenized)
        self.other_vocab_size = len(self.other_tokenizer.word_index) + 1
        self.other_length = max_length(self.other_tokenized)
        print('Other Vocabulary Size: %d' % self.other_vocab_size)
        print('Other Max Length: %d' % self.other_length)

        self.num_samples = len(self.eng_texts)


    def get_name(self):
        return self.name

    def __str__(self):
        return "Name: %s" % (self.name)

    def translate(self, source, verbose=True):
        """
        :param source: the input natural language
        :type source: string
        :return: string the most likely translation as a string
        """
        return self.predict_sequence(source, verbose)

    @abc.abstractmethod
    def train_save(self, epochs=epochs_default):
        """
        Trains a given model with tokenizer and checkpoints it to a file for later
        :param  epochs: the number of epochs to repeat during training (0 means load saved model but don't fit any more)
        """
        return

    def predict_sequence(self, source, verbose=True):
        """
        generate target given source sequence (for all predictions)
        :type source: The input data in Swedish
        :return the most likely prediction in English
        """
        # prepare/encode/pad data (pad to length of target language)

        testX = self.get_x(source)
        translations = list()
        predictions = self.model.predict(testX, verbose=0)
        for prediction in predictions:
            integers = [np.argmax(vector) for vector in prediction]
            target = list()
            for i in integers:
                # turn integers back to words
                word = word_for_id(i, self.eng_tokenizer)
                if word is None:
                    break
                target.append(word)
            translations.append(self.tokenizer.join(target, 'en'))
        if verbose: print("Candidates=", translations)
        return translations[0]

    def get_checkpoint(self, filename):
        return ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

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
        :param dataset: The dataset (sentence pairs in both languages)
        :type dataset: ndarray
        """
        dataset_lang2 = dataset[:, 1]
        self.other_tokenizer = create_tokenizer(self.tokenizer.tokenize(dataset_lang2, lang2))
        other_tokenized = self.tokenizer.tokenize(dataset_lang2, lang2)
        self.other_length = max_length(other_tokenized)
        # prepare/encode/pad data (pad to length of target language)
        self.pad_length = self.other_length if self.name.startswith('simple') else None

    @staticmethod
    def post_fit(filename_prefix, history, time_callback):
        """
        Log the training history (particularly acc and val_acc since CSVLogger doesn't seem to save them
        :param time_callback:
        """
        # Print the training history
        print(history.history['val_loss'])
        # Write this history to a file for later analysis
        filename = filename_prefix + '-history.json'
        with open(filename, 'a') as file:
            file.write(json.dumps(history.history, indent=2))
            file.close()
        print("Wrote history to %s" % filename)
        # Record time taken
        filename = filename_prefix + '-time.txt'
        with open(filename, 'a') as file:
            file.writelines("%d\n" % item for item in time_callback.times)
            file.close()
        print("Wrote times to %s" % filename)


class TimeHistory(Callback):
    """
    Record the time taken to train each epoch
    """
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)