"""
This module defines a simple model with fixed lengths.

See: https://chunml.github.io/ChunML.github.io/project/Sequence-To-Sequence/

"""

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Embedding , LSTM, RepeatVector, TimeDistributed
from keras.models import Sequential, load_model
from keras.utils import plot_model

from helpers import load_clean_sentences, create_tokenizer, max_length, lang2, encode_sequences, encode_1hot
from models.base import BaseModel
from config import batch_size, epochs_default


class Simple(BaseModel):

    def __init__(self, name, tokenizer_func, optimizer):
        BaseModel.__init__(self, name, tokenizer_func, optimizer)


    def define_model(self, src_vocab, target_vocab, src_timesteps, target_timesteps, n_units=100):
        """
         define a simple "naive?" NMT model using LSTM as the RNN Cell Type with no attention mechanism
         with fixed/max sentence length
        :param :src_timesteps: max length of src language sentences
        :param :target_timesteps: max length of targets language sentences
        :param :n_units: dimensionality of the LSTM layers (more dimensions => more accurate meaning => longer to train => better. After 2k not much improvement) Also often labelled latent_dim ?
        :return: A Keras model instance.
        """
        hidden_size = n_units # TODO: Does this need to be the same value??
        model = Sequential()
        model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True)) # builds "thought" mappings
        model.add(LSTM(hidden_size)) # RNN Cell (encoder?) return_state=False by default
        model.add(RepeatVector(target_timesteps)) # Direction of Encoder Input = forward?
        # for _ in range(num_layers): can add multiple layers here:
        model.add(LSTM(hidden_size, return_sequences=True))  # RNN Cell (decoder?)
        model.add(TimeDistributed(Dense(target_vocab, activation='softmax')))
        # todo: Activation layer?
        return model

    def train_save(self, epochs=epochs_default):
        """
        Trains a given model with tokenizer and checkpoints it to a file for later
        """
        tokenizer_func = self.tokenizer_func
        print("\n###\nAbout to train model %s with tokenizer %s and optimizer %s\n###\n\n"
              % (__name__, tokenizer_func.__name__, self.optimizer))
        # load datasets
        dataset = load_clean_sentences('both')
        train = load_clean_sentences('train')
        test = load_clean_sentences('test')

        print("Prepare english tokenizer")
        dataset_lang1 = dataset[:, 0]
        eng_tokenized = tokenizer_func(dataset_lang1, 'en')
        eng_tokenizer = create_tokenizer(eng_tokenized)
        eng_vocab_size = len(eng_tokenizer.word_index) + 1
        eng_length = max_length(eng_tokenized)
        print('English Vocabulary Size: %d' % eng_vocab_size)
        print('English Max Length: %d' % eng_length)

        print("Prepare other language tokenizer")
        dataset_lang2 = dataset[:, 1]
        other_tokenized = tokenizer_func(dataset_lang2, lang2)
        other_tokenizer = create_tokenizer(other_tokenized)
        other_vocab_size = len(other_tokenizer.word_index) + 1
        other_length = max_length(other_tokenized)
        print('Other Vocabulary Size: %d' % other_vocab_size)
        print('Other Max Length: %d' % other_length)

        print("Prepare training data")
        trainX = encode_sequences(other_tokenizer, tokenizer_func(train[:, 1], lang2), other_length)
        trainY = encode_sequences(eng_tokenizer, tokenizer_func(train[:, 0], 'en'), eng_length)
        print("Prepare validation data")
        testX = encode_sequences(other_tokenizer, tokenizer_func(test[:, 1], lang2), other_length)
        testY = encode_sequences(eng_tokenizer, tokenizer_func(test[:, 0], 'en'), eng_length)

        # One-hot encode
        trainY = encode_1hot(trainY, eng_vocab_size)
        testY = encode_1hot(testY, eng_vocab_size)

        ### todo: try reversing the order of Y tokens (for both training and evaluation of course)

        print("\n")
        try:
            # try and load checkpointed model to continue
            model = load_model('checkpoints/' + self.name + '.h5')
            print("Loaded checkpointed model")
        except:
            print("Define and compile model")
            model = self.define_model(other_vocab_size, eng_vocab_size, other_length, eng_length)
            model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')

        if epochs == 0:  # a 'hack' to prepare the models without actually doing any fitting
            return

        # summarize defined model
        print(model.summary())
        # plot_model(model, to_file=('checkpoints/' + self.name + '.png'), show_shapes=True)
        print("Fit model")
        checkpoint = ModelCheckpoint('checkpoints/' + self.name + '.h5', monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min')
        # the model is saved via a callback checkpoint
        X = trainX
        y = trainY
        # where `X` is Training data and `y` are Target values
        # model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpoint], verbose=2)
        model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), callbacks=[checkpoint], verbose=1)