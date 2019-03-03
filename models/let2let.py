"""
A letter-by-letter model for translation with Keras.
Ignores input tokenizer and just does letter-by-letter
Based on https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras.utils import plot_model

from config import epochs_default
from models.base import BaseModel, TimeHistory

import numpy as np
from helpers import *
import os.path


class Let2Let(BaseModel):

    CH_START = '\t'
    CH_END = '\n'

    CHARS_BASIC = "abcdefghijklmnopqrstuvwxyzäåö" + "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÅÖ" + "0123456789" + "!@#$%^&*()[]{}?<>,.;:"

    batch_size = 64  # Batch size for training.
    epochs = 10  # Number of epochs to train for.
    latent_dim = 256  # Latent dimensionality of the encoding space.

    # mode = 'restart' # Start from default weights
    mode = 'continue'  # Load previous weights and continue fitting
    # mode = 'readonly' # Use the pre-trained model but don't do any more fitting

    input_token_index = dict()
    target_token_index = dict()

    def __init__(self, name, tokenizer, optimizer):
        BaseModel.__init__(self, name, tokenizer, optimizer)

        # Vectorize the data.
        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.target_characters = set()

        for ch in self.CHARS_BASIC:
            self.input_characters.add(ch)
            self.target_characters.add(ch)

        lines = load_clean_sentences('both')

        for line in lines:
            input_text = line[1]  # Swedish
            target_text = line[0]  # English
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = self.CH_START + target_text + self.CH_END
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                if char not in self.input_characters:
                    self.input_characters.add(char)
            for char in target_text:
                if char not in self.target_characters:
                    self.target_characters.add(char)

        self.input_characters = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))
        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])

        print('Number of samples:', len(self.input_texts))
        print('Number of unique input tokens:', self.num_encoder_tokens)
        print('Number of unique output tokens:', self.num_decoder_tokens)
        print('Max sequence length for inputs:', self.max_encoder_seq_length)
        print('Max sequence length for outputs:', self.max_decoder_seq_length)

        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(self.target_characters)])

        self.encoder_input_data = np.zeros(
            (len(self.input_texts), self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')
        self.decoder_input_data = np.zeros(
            (len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')
        self.decoder_target_data = np.zeros(
            (len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())

    def train_save(self, epochs=epochs_default):

        self.model = self.define_model()

        filename = 'checkpoints/' + self.name + '.h5'

        if  os.path.isfile(filename):
            # Load the previous model (layers and weights but NO STATE)
            self.model.load_weights(filename)

        if not os.path.isfile(filename + '.png'):
            # Plot the model and save it too
            plot_model(self.model, to_file=(filename + '.png'), show_shapes=True)

        if epochs > 0:
            # Prepare checkpoints
            checkpoint = self.get_checkpoint(filename + '.h5')
            logger = CSVLogger(filename + '.csv', separator=',', append=True)
            earlyStopping = EarlyStopping(patience=2, verbose=1)
            time_callback = TimeHistory()  # record the time taken to train each epoch
            # Run training
            history = self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                           batch_size=self.batch_size,
                           epochs=epochs,
                           validation_split=0.2,
                           callbacks=[checkpoint, logger, earlyStopping, time_callback])
            # Save model
            self.post_fit(filename, history,  time_callback)

    def define_model(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder_lstm = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs_interim, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs_interim)
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')
        return model

    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    def prep_models(self, model):
        """
        Prepare the models from the saved one
        :param model: The trained encoder model
        :return: (encoder_model, decoder_model)
        """

        # Redefine encoder model (needs to work even when the model has just been loaded from an h5 file)
        encoder_inputs = model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = model.input[1]  # input_2
        decoder_state_input_h = Input(shape=(self.latent_dim,), name='input_3')
        decoder_state_input_c = Input(shape=(self.latent_dim,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        return (encoder_model, decoder_model)

    def translate(self, input_text, verbose=True):

        encoder_input_data = np.zeros(
            (1, self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')

        for t, char in enumerate(input_text):
            try:
                idx = self.input_token_index[char]
                encoder_input_data[0, t, idx] = 1.
            except KeyError:
                print("No match for char=" + str(char))
                pass

        return self.decode_sequence(encoder_input_data[0:1])

    def decode_sequence(self, input_seq):
        """
        :param input_seq: tuple(1, src_vocab, max_length_src) one-hot encoded
        :return: str: translated natural language
        """

        encoder_model, decoder_model = self.prep_models(self.model)

        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index[self.CH_START]] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            model_input = [target_seq] + states_value
            output_tokens, h, c = decoder_model.predict(model_input)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == self.CH_END or
                    len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def quick_test(self):
        """
        Run a quick test of the first 100 lines
        """
        for seq_index in range(100):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            input_seq = self.encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = self.decode_sequence(input_seq)
            print('-')
            print('Input sentence:', self.input_texts[seq_index])
            print('Decoded sentence:', decoded_sentence)
