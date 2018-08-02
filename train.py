"""
This module trains a model and stores it in a file
"""
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

from helpers import *

epochs = 60

n_units = 256  # Dimensionality of word-embedding (and so LSTM layer)

batch_size = 64 # TODO: Is this batch size too big?

def train_save(model_function, tokenizer_func, filename, optimizer='adam'):
    """
    Trains a given model with tokenizer and checkpoints it to a file for later
    :param model_function: the function to define the model
    :param tokenizer_func: the function to tokenize input strings
    :param filename: the model name (no extension)
    """
    print("About to train model %s with tokenizer %s and optimizer %s"
          % (model_function.__name__, tokenizer_func.__name__, optimizer))
    # load datasets
    dataset = load_clean_sentences('both')
    train = load_clean_sentences('train')
    test = load_clean_sentences('test')
    # prepare english tokenizer
    dataset_lang1 = dataset[:, 0]
    eng_tokenized = tokenizer_func(dataset_lang1, 'en')
    eng_tokenizer = create_tokenizer(eng_tokenized)
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(eng_tokenized)
    print('English Vocabulary Size: %d' % eng_vocab_size)
    print('English Max Length: %d' % eng_length)
    # prepare other language tokenizer
    dataset_lang2 = dataset[:, 1]
    other_tokenized = tokenizer_func(dataset_lang2, lang2)
    other_tokenizer = create_tokenizer(other_tokenized)
    other_vocab_size = len(other_tokenizer.word_index) + 1
    other_length = max_length(other_tokenized)
    print('Other Vocabulary Size: %d' % other_vocab_size)
    print('Other Max Length: %d' % other_length)

    # prepare training data
    trainX = encode_sequences(other_tokenizer, other_length, tokenizer_func(train[:, 1], lang2))
    trainY = encode_sequences(eng_tokenizer, eng_length, tokenizer_func(train[:, 0], 'en'))
    trainY = encode_output(trainY, eng_vocab_size)
    # prepare validation data
    testX = encode_sequences(other_tokenizer, other_length, tokenizer_func(test[:, 1], lang2))
    testY = encode_sequences(eng_tokenizer, eng_length, tokenizer_func(test[:, 0], 'en'))
    testY = encode_output(testY, eng_vocab_size)

    # define model
    model = model_function(other_vocab_size, eng_vocab_size, other_length, eng_length, n_units)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    # summarize defined model
    print(model.summary())
    plot_model(model, to_file=('checkpoints/' + filename + '.png'), show_shapes=True)
    # fit model
    checkpoint = ModelCheckpoint('checkpoints/' + filename + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # the model is saved via a callback checkpoint
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)

def train_all():
    """Train the models and tokenizer permutations"""
    for model_name, model_func in models.items():
        for token_id, tokenizer in tokenizers.items():
            for opt_id, optimizer in optimizers.items():
                # save each one
                filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                train_save(model_func, tokenizer, filename, optimizer)


if __name__ == '__main__':
    # Start the training
    train_all()
