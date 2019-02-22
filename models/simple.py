"""
This module defines a simple model with fixed lengths, with input passed as a 1-hot encoded 3-d ndarray

See: https://chunml.github.io/ChunML.github.io/project/Sequence-To-Sequence/

"""
import os

from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.layers import Dense, Embedding, LSTM, RepeatVector, TimeDistributed, Activation
from keras.models import Sequential, load_model
from keras.utils import plot_model

from helpers import load_clean_sentences, lang2, encode_sequences, encode_1hot
from models.base import BaseModel
from config import batch_size, epochs_default


class Simple(BaseModel):

    def __init__(self, name, tokenizer, optimizer):
        BaseModel.__init__(self, name, tokenizer, optimizer)


    def define_model(self, src_vocab, target_vocab, src_timesteps, target_timesteps, n_units=100):
        """
         define a simple "naive?" NMT model using LSTM as the RNN Cell Type with no attention mechanism
         with fixed/max sentence length
        :param :src_timesteps: max length of src language sentences
        :param :target_timesteps: max length of targets language sentences
        :param :n_units: dimensionality of the LSTM layers (more dimensions => more accurate meaning => longer to train => better. After 2k not much improvement) Also often labelled latent_dim ?
        :return: A Keras model instance.
        """
        model = Sequential()
        model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))  # builds "thought" mappings
        model.add(LSTM(n_units))  # RNN Cell (encoder?) return_state=False by default
        model.add(RepeatVector(target_timesteps))  # Direction of Encoder Input = forward?
        model.add(LSTM(n_units, return_sequences=True))  # RNN Cell (decoder?)
        model.add(TimeDistributed(Dense(target_vocab, activation='softmax')))
        return model

    def get_x(self, source):
        tokenized = self.tokenizer.tokenize([source], lang2)
        encoded_int = encode_sequences(self.other_tokenizer, tokenized, self.other_length)
        # encoded_1hot = encode_1hot(encoded_int, self.other_vocab_size)
        return encoded_int

    def train_save(self, epochs=epochs_default):
        """
        Trains a given model with tokenizer and checkpoints it to a file for later
        """
        tokenizer = self.tokenizer
        print("\n###\nAbout to train model %s with tokenizer %s and optimizer %s\n###\n\n"
              % (__name__, tokenizer.__class__.__name__, self.optimizer))
        # load datasets
        train = load_clean_sentences('train')
        test = load_clean_sentences('test')

        print("Prepare training data")
        trainX = encode_sequences(self.other_tokenizer, tokenizer.tokenize(train[:, 1], lang2), self.other_length)
        trainY = encode_sequences(self.eng_tokenizer, tokenizer.tokenize(train[:, 0], 'en'), self.eng_length)
        print("Prepare validation data")
        testX = encode_sequences(self.other_tokenizer, tokenizer.tokenize(test[:, 1], lang2), self.other_length)
        testY = encode_sequences(self.eng_tokenizer, tokenizer.tokenize(test[:, 0], 'en'), self.eng_length)

        # One-hot encode
        trainY = encode_1hot(trainY, self.eng_vocab_size)
        testY = encode_1hot(testY, self.eng_vocab_size)

        ### todo: try reversing the order of Y tokens (for both training and evaluation of course)

        self.model = self.define_model(self.other_vocab_size, self.eng_vocab_size, self.other_length, self.eng_length)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')

        filename = 'checkpoints/' + self.name

        if os.path.isfile(filename):
            # Load the previous model (layers and weights but NO STATE)
            self.model.load_weights(filename + '.h5')
        else:
            print("No existing model file found: %s" % filename)

        if not os.path.isfile(filename + '.png'):
            # Plot the model and save it too
            plot_model(self.model, to_file=(filename + '.png'), show_shapes=True)

        print("\n")

        if epochs == 0:  # a 'hack' to prepare the models without actually doing any fitting
            return

        # summarize defined model
        print(self.model.summary())
        # plot_model(model, to_file=('checkpoints/' + self.name + '.png'), show_shapes=True)
        print("Fit model")
        checkpoint = self.get_checkpoint(filename + '.h5')
        logger = CSVLogger(filename + '.csv', separator=',', append=True)
        earlyStopping = EarlyStopping()
        # the model is saved via a callback checkpoint
        X = trainX
        y = trainY
        # where `X` is Training data and `y` are Target values
        # model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpoint], verbose=2)
        history = self.model.fit(X, y, validation_data=(testX, testY),
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[checkpoint, logger, earlyStopping],
                       verbose=1)
        # Print/save history for later analysis
        self.post_fit(filename, history)



class Simple2(Simple):
    """
    An extension of the simple model with multiple LSTM decoder layers and an Activation layer
    """

    def define_model(self, src_vocab, target_vocab, src_timesteps, target_timesteps, n_units=100):
        """
         define a simple "naive?" NMT model using LSTM as the RNN Cell Type with no attention mechanism
         with fixed/max sentence length
        :param :src_timesteps: max length of src language sentences
        :param :target_timesteps: max length of targets language sentences
        :param :n_units: dimensionality of the LSTM layers (more dimensions => more accurate meaning => longer to train => better. After 2k not much improvement) Also often labelled latent_dim ?
        :return: A Keras model instance.
        """
        hidden_size = n_units
        num_layers = 4
        model = Sequential()
        model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))  # builds "thought" mappings
        model.add(LSTM(hidden_size))  # RNN Cell (encoder?) return_state=False by default
        model.add(RepeatVector(target_timesteps))  # Direction of Encoder Input = forward?
        for _ in range(num_layers):
           model.add(LSTM(hidden_size, return_sequences=True))  # RNN Cell (decoder?)
        model.add(TimeDistributed(Dense(target_vocab, activation='softmax')))
        model.add(Activation('softmax'))
        return model
