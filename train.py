"""
This module trains a model and stores it in a file
"""
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.models import load_model

from helpers import *
import numpy as np
import tensorflow as tf

epochs_default = 20

latent_dim = 256  # Dimensionality of word-embedding (and so LSTM layer)

batch_size = 64 # TODO: Is this batch size too big?

print ("VERSION", tf.Session(config=tf.ConfigProto(log_device_placement=True)))

def train_save(model_function, tokenizer_func, filename, optimizer='adam', epochs=epochs_default):
    """
    Trains a given model with tokenizer and checkpoints it to a file for later
    :param model_function: the function to define the model
    :param tokenizer_func: the function to tokenize input strings
    :param filename: the model name (no extension)
    """
    print("\n###\nAbout to train model %s with tokenizer %s and optimizer %s\n###\n\n"
          % (model_function.__name__, tokenizer_func.__name__, optimizer))
    # load datasets
    dataset = load_clean_sentences('both')
    train = load_clean_sentences('train')
    test = load_clean_sentences('test')

    print("Prepare english tokenizer")
    dataset_lang1 = dataset[:, 0]
    eng_tokenized = tokenizer_func(dataset_lang1, 'en')
    if model_function != simple.simple_model: eng_tokenized = mark_ends(eng_tokenized)
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
    train_tokenized = tokenizer_func(train[:, 0], 'en')
    if model_function != simple.simple_model: train_tokenized = mark_ends(train_tokenized)
    trainY = encode_sequences(eng_tokenizer, train_tokenized, eng_length)
    if model_function == attention.dense_model:  trainY = encode_output(trainY, eng_vocab_size)
    print("Prepare validation data")
    testX = encode_sequences(other_tokenizer, tokenizer_func(test[:, 1], lang2), other_length)
    validation_tokenized = tokenizer_func(test[:, 0], 'en')
    if model_function != simple.simple_model: validation_tokenized = mark_ends(validation_tokenized)
    testY = encode_sequences(eng_tokenizer, validation_tokenized, eng_length)
    if model_function == attention.dense_model: testY = encode_output(testY, eng_vocab_size)

    print("\n")
    try:
        # try and load checkpointed model to continue
        model = load_model('checkpoints/' + filename + '.h5')
        print("Loaded checkpointed model")
    except:
        print("Define and compile model")
        model = model_function(other_vocab_size, eng_vocab_size, other_length, eng_length)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    if epochs == 0:  # a 'hack' to prepare the models without actually doing any fitting
        return

    # summarize defined model
    print(model.summary())
    plot_model(model, to_file=('checkpoints/' + filename + '.png'), show_shapes=True)
    print("Fit model")
    checkpoint = ModelCheckpoint('checkpoints/' + filename + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # the model is saved via a callback checkpoint
    if model_function == simple.simple_model:
        X = trainX
        y = trainY
    else: # the dense/attention model
        # Need to encode the source input
        if model_function == attention.dense_model:
            trainX = encode_output(trainX, other_vocab_size)
            testX = encode_output(testX, other_vocab_size)
        # see attention.py: [encoder_inputs, decoder_inputs]
        X = [trainX, trainY]
        testX = [testX, testY]
        # prepare decoder target offset by 1
        if model_function == attention2.dense_model:
            y = encode_output(offset_data(trainY), eng_vocab_size)
            testY = encode_output(offset_data(testY), eng_vocab_size)
        else:
            y = offset_data(trainY)
            testY = offset_data(testY)
    # where `X` is Training data and `y` are Target values
    # model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpoint], verbose=2)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), callbacks=[checkpoint], verbose=1)


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


def train_all():
    """Train the models and tokenizer permutations"""
    for model_name, model_func in models.items():
        for token_id, tokenizer in tokenizers.items():
            for opt_id, optimizer in optimizers.items():
                # save each one
                filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                try:
                    train_save(model_func, tokenizer, filename, optimizer)
                except:
                    print("Error training model: " + filename)
                    traceback.print_exc()
                    pass


if __name__ == '__main__':
    # Start the training
    train_all()
