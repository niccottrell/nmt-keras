"""
This module defines the various models that will be tested.
Based on a character-level model at https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py and lstm_seq2seq_restore.py and https://github.com/roatienza/Deep-Learning-Experiments/blob/master/keras/seq2seq/seq2seq_translate.py
"""
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Input, LSTM
from keras.models import Model, load_model
from keras.utils.vis_utils import plot_model

from models.base import BaseModel

import numpy as np

from helpers import *
from config import epochs_default


class Attention(BaseModel):

    # Assume a problem is the decoded length exceeds this
    max_decoder_seq_length = 40

    print(__file__)

    def __init__(self, name, tokenizer_func, optimizer):
        BaseModel.__init__(self, name, tokenizer_func, optimizer)

    def dense_model(self, src_vocab, target_vocab, src_timesteps, target_timesteps, latent_dim=256):
        """
        To train:
        encoder_input_data is a 3D array of shape (num_pairs, max_english_sentence_length, num_english_characters) containing a one-hot vectorization of the English sentences.
        decoder_input_data is a 3D array of shape (num_pairs, max_french_sentence_length, num_french_characters) containg a one-hot vectorization of the French sentences.
        decoder_target_data is the same as decoder_input_data but offset by one timestep. decoder_target_data[:, t, :] will be the same as decoder_input_data[:, t + 1, :].
        Requires teacher forcing
        See also https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
        :param src_vocab: int: the number of different tokens/words in the source language
        :param target_vocab: int: the number of different tokens/words in the target language
        :param src_timesteps: int: max sentence length of source language (English)
        :param target_timesteps: max sentence length of target language (Swedish)
        :param latent_dim: the dimensionality of the
        :return: The training model
        """

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, src_vocab))
        # The return_state constructor argument configures a RNN layer to return a list where the first entry is the outputs
        # and the next entries are the internal RNN states. This is used to recover the states of the encoder.
        encoder = LSTM(latent_dim, return_sequences=True)(encoder_inputs)
        encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]
        # Set up the decoder, using `encoder_states` as initial state of the RNN.
        decoder_inputs = Input(shape=(None, target_vocab))
        # decoder_embedding = Embedding(target_vocab, latent_dim)(decoder_inputs)
        # The return_sequences constructor argument, configuring a RNN to return its full sequence of outputs (instead of
        # just the last output, which the defaults behavior).
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs_interim, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        # Attach through a dense layer
        decoder_dense = Dense(target_vocab, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs_interim)
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return training_model


    def infer_models(self, model, latent_dim=256):
        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states
        # Define sampling models

        encoder_inputs = model.input[0]  # input_1 (Input(shape=(None, src_vocab)))
        encoder_outputs, state_h_enc, state_c_enc = model.layers[3].output  # lstm_1 (LSTM(latent_dim, return_state=True))
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = model.input[1]  # input_2 (Input(shape=(None, target_vocab)))
        decoder_state_input_h = Input(shape=(latent_dim,), name='input_3') # named to avoid conflict
        decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        # Use decoder_lstm from the training model
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_dense = model.layers[5]  # should match Dense(target_vocab, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return (encoder_model, decoder_model)


    def decode_sequence(self, input_seq):
        """
        Decode (translate) the input sequence into natural language in the target language
        :param input_seq: The one-hot encoded 3-d numpy array
        :param model: Model
        :param tokenizer: Tokenizer
        :return: the target language sentence output

        """

        encoder_model, decoder_model = self.infer_models(self.model)

        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # the target vocab size (e.g. English)
        target_vocab = decoder_model.input_shape[0][2]

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, target_vocab))
        # Populate the first character of target sequence with the start character.
        start_seq = self.tokenizer.word_index['\t'] # tokenizer.texts_to_sequences(['\t'])
        target_seq[0, 0, start_seq] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = word_for_id(sampled_token_index, self.tokenizer)
            decoded_sentence += sampled_word

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_word == '\n' or
                    len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True
            else:
                decoded_sentence += ' '

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, target_vocab))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence


    def train_save(self, epochs=epochs_default):
        """
        Trains a given model with tokenizer and checkpoints it to a file for later
        :param tokenizer_func: the function to tokenize input strings
        :param filename: the model name (no extension)
        """
        print("\n###\nAbout to train model %s with tokenizer %s and optimizer %s\n###\n\n"
              % (self.__name__, self.tokenizer_func.__name__, self.optimizer))
        # load datasets
        dataset = load_clean_sentences('both')
        train = load_clean_sentences('train')
        test = load_clean_sentences('test')

        print("Prepare english tokenizer")
        dataset_lang1 = dataset[:, 0]
        eng_tokenized = self.tokenizer_func(dataset_lang1, 'en')
        eng_tokenized = mark_ends(eng_tokenized)
        eng_tokenizer = create_tokenizer(eng_tokenized)
        eng_vocab_size = len(eng_tokenizer.word_index) + 1
        eng_length = max_length(eng_tokenized)
        print('English Vocabulary Size: %d' % eng_vocab_size)
        print('English Max Length: %d' % eng_length)

        print("Prepare other language tokenizer")
        dataset_lang2 = dataset[:, 1]
        other_tokenized = self.tokenizer_func(dataset_lang2, lang2)
        other_tokenizer = create_tokenizer(other_tokenized)
        other_vocab_size = len(other_tokenizer.word_index) + 1
        other_length = max_length(other_tokenized)
        print('Other Vocabulary Size: %d' % other_vocab_size)
        print('Other Max Length: %d' % other_length)

        print("Prepare training data")
        trainX = encode_sequences(other_tokenizer, self.tokenizer_func(train[:, 1], lang2), other_length)
        train_tokenized = self.tokenizer_func(train[:, 0], 'en')
        train_tokenized = mark_ends(train_tokenized)
        trainY = encode_sequences(eng_tokenizer, train_tokenized, eng_length)
        trainY = encode_output(trainY, eng_vocab_size)
        print("Prepare validation data")
        testX = encode_sequences(other_tokenizer, self.tokenizer_func(test[:, 1], lang2), other_length)
        validation_tokenized = self.tokenizer_func(test[:, 0], 'en')
        validation_tokenized = mark_ends(validation_tokenized)
        testY = encode_sequences(eng_tokenizer, validation_tokenized, eng_length)
        testY = encode_output(testY, eng_vocab_size)

        print("\n")
        try:
            # try and load checkpointed model to continue
            model = load_model('checkpoints/' + self.name + '.h5')
            print("Loaded checkpointed model")
        except:
            print("Define and compile model")
            model = self.define_model(other_vocab_size, eng_vocab_size, other_length, eng_length, eng_vocab_size)
            model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')

        if epochs == 0:  # a 'hack' to prepare the models without actually doing any fitting
            return

        # summarize defined model
        print(model.summary())
        plot_model(model, to_file=('checkpoints/' + self.name + '.png'), show_shapes=True)
        print("Fit model")
        checkpoint = ModelCheckpoint('checkpoints/' + self.name + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # the model is saved via a callback checkpoint
        # Need to encode the source input
        trainX = encode_output(trainX, other_vocab_size)
        testX = encode_output(testX, other_vocab_size)
        # see attention.py: [encoder_inputs, decoder_inputs]
        X = [trainX, trainY]
        testX = [testX, testY]
        # prepare decoder target offset by 1
        y = self.offset_data(trainY)
        testY = self.offset_data(testY)
        # where `X` is Training data and `y` are Target values
        # model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[checkpoint], verbose=2)
        model.fit(X, y, epochs=epochs, batch_size=train.batch_size, validation_data=(testX, testY), callbacks=[checkpoint], verbose=1)


    def translate(self, source):
        """
        :param model: Model
        :param source: list(int)
        :param tokenizer: Tokenizer
        """
        source = source.reshape((1, source.shape[0]))
        vocab_size = self.model.input_shape[0][2]
        # encode to one-hot ndarray (3-dimensions)
        source_encoded = encode_output(source, vocab_size)
        return self.decode_sequence(source_encoded)
