"""
This module trains a model and stores it in a file
"""
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

from helpers import *


def simple_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    """
     define a simple NMT model
    :param :n_units: dimensionality of the LTSM layers
    :return: A Keras model instance.
    """
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model


def train_save(model_function, tokenizer_func, filename):
    # load datasets
    dataset = load_clean_sentences('eng-' + lang2 + '-both.pkl')
    train = load_clean_sentences('eng-' + lang2 + '-train.pkl')
    test = load_clean_sentences('eng-' + lang2 + '-test.pkl')

    # prepare english tokenizer
    eng_tokenizer = tokenizer_func(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(dataset[:, 0])
    print('English Vocabulary Size: %d' % eng_vocab_size)
    print('English Max Length: %d' % (eng_length))
    # prepare german tokenizer
    other_tokenizer = tokenizer_func(dataset[:, 1])
    other_vocab_size = len(other_tokenizer.word_index) + 1
    other_length = max_length(dataset[:, 1])
    print('Other Vocabulary Size: %d' % other_vocab_size)
    print('Other Max Length: %d' % (other_length))

    # prepare training data
    trainX = encode_sequences(other_tokenizer, other_length, train[:, 1])
    trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
    trainY = encode_output(trainY, eng_vocab_size)
    # prepare validation data
    testX = encode_sequences(other_tokenizer, other_length, test[:, 1])
    testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
    testY = encode_output(testY, eng_vocab_size)

    # define model
    n_units = 256
    model = model_function(other_vocab_size, eng_vocab_size, other_length, eng_length, n_units)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # summarize defined model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    # fit model
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # the model is saved via a callback checkpoint
    model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint],
              verbose=2)

models = {'simple': simple_model}
tokenizers = {'a': create_tokenizer_simple}

def train_all():
    """Train the models and tokenizer permutations"""
    for model_name, model_func in models.items():
        for token_id, tokenizer in tokenizers.items():
            # save each one
            train_save(model_func, tokenizer, model_name +'_' + token_id + '.h5')

# Start the training
train_all()