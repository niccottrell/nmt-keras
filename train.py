"""
This module trains a model and stores it in a file
"""
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

from helpers import *


def train_save(model_function, tokenizer_func, filename, optimizer='adam'):
    """
    Trains a given model with tokenizer and checkpoints it to a file for later
    :param model_function: the function to define the model
    :param tokenizer_func: the function to tokenize input strings
    :param filename: the model name (no extension)
    :return:
    """
    print("About to train model %s with tokenizer %s and optimizer %s"
          % (model_function, tokenizer_func, optimizer))
    # load datasets
    dataset = load_clean_sentences('eng-' + lang2 + '-both.pkl')
    train = load_clean_sentences('eng-' + lang2 + '-train.pkl')
    test = load_clean_sentences('eng-' + lang2 + '-test.pkl')
    # prepare english tokenizer
    eng_tokenizer = create_tokenizer(tokenizer_func(dataset[:, 0], 'en'))
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(dataset[:, 0])
    print('English Vocabulary Size: %d' % eng_vocab_size)
    print('English Max Length: %d' % eng_length)
    # prepare other language tokenizer
    other_tokenizer = create_tokenizer(tokenizer_func(dataset[:, 1], lang2))
    other_vocab_size = len(other_tokenizer.word_index) + 1
    other_length = max_length(dataset[:, 1])
    print('Other Vocabulary Size: %d' % other_vocab_size)
    print('Other Max Length: %d' % other_length)

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
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    # summarize defined model
    print(model.summary())
    plot_model(model, to_file=('checkpoints/' + filename + '.png'), show_shapes=True)
    # fit model
    checkpoint = ModelCheckpoint('checkpoints/'+ filename + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # the model is saved via a callback checkpoint
    model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint],
              verbose=2)

def train_all():
    """Train the models and tokenizer permutations"""
    for model_name, model_func in models.items():
        for token_id, tokenizer in tokenizers.items():
            # save each one
            train_save(model_func, tokenizer, model_name + '_' + token_id, optimizer)


def summarize_tokenizers():
    # load full dataset
    dataset = load_clean_sentences('eng-' + lang2 + '-both.pkl')
    for tokenizer_id, tokenizer_func in tokenizers.items():
        print('Summary of Tokenizer: %s' % tokenizer_func)
        # prepare english tokenizer
        eng_tokenizer = create_tokenizer(tokenizer_func(dataset[:, 0], 'en'))
        eng_vocab_size = len(eng_tokenizer.word_index) + 1
        eng_length = max_length(dataset[:, 0])
        print('English Vocabulary Size: %d' % eng_vocab_size)
        print('English Max Length: %d' % eng_length)
        # prepare german tokenizer
        other_tokenizer = create_tokenizer(tokenizer_func(dataset[:, 1], lang2))
        other_vocab_size = len(other_tokenizer.word_index) + 1
        other_length = max_length(dataset[:, 1])
        print('Other Vocabulary Size: %d' % other_vocab_size)
        print('Other Max Length: %d' % other_length)

summarize_tokenizers()

# Start the training
train_all()
