"""
Fork of let2let.py

"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils import plot_model

from config import epochs_default
from models.base import BaseModel

import numpy as np
from helpers import *
import os.path


class Attention3(BaseModel):
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    CH_START = '\t'
    CH_END = '\n'

    batch_size = 64  # Batch size for training.
    latent_dim = None  # Latent dimensionality of the encoding space.

    input_token_index = dict()
    target_token_index = dict()

    # Cache the encode models too
    encoder_model = None
    decoder_model = None

    def __init__(self, name, tokenizer, optimizer, include_dropout=False, latent_dim=256):
        BaseModel.__init__(self, name, tokenizer, optimizer)

        # Collection all tokens across all input lines
        self.include_dropout = include_dropout
        self.latent_dim = latent_dim
        self.other_tokens = set()  # input
        self.eng_tokens = {self.CH_START, self.CH_END}  # target

        for idx, line in enumerate(self.eng_texts):
            self.eng_texts[idx] = self.CH_START + self.eng_texts[idx] + self.CH_END
            self.eng_tokenized[idx] = [self.CH_START] + self.eng_tokenized[idx] + [self.CH_END]
            for token in self.other_tokenized[idx]:
                self.other_tokens.add(token)
            for token in self.eng_tokenized[idx]:
                self.eng_tokens.add(token)

        self.other_tokens = sorted(list(self.other_tokens))
        self.eng_tokens = sorted(list(self.eng_tokens))
        self.num_encoder_tokens = len(self.other_tokens)
        self.num_decoder_tokens = len(self.eng_tokens)
        self.max_encoder_seq_length = max([len(txt) for txt in self.other_tokenized])
        self.max_decoder_seq_length = max([len(txt) for txt in self.eng_tokenized])

        print('Number of samples:', self.num_samples)
        print('Number of unique input tokens:', self.num_encoder_tokens)
        print('Number of unique output tokens:', self.num_decoder_tokens)
        print('Max sequence length for inputs:', self.max_encoder_seq_length)
        print('Max sequence length for outputs:', self.max_decoder_seq_length)

        self.input_token_index = dict(
            [(token, i) for i, token in enumerate(self.other_tokens)])
        self.target_token_index = dict(
            [(token, i) for i, token in enumerate(self.eng_tokens)])

        self.encoder_input_data = np.zeros(
            (self.num_samples, self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='uint8')
        self.decoder_input_data = np.zeros(
            (self.num_samples, self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='uint8')
        self.decoder_target_data = np.zeros(
            (self.num_samples, self.max_decoder_seq_length, self.num_decoder_tokens),
            dtype='uint8')

        # Create one-hot encoded values directly
        for i, (input_text, target_text) in enumerate(zip(self.other_tokenized, self.eng_tokenized)):
            for t, token in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[token]] = 1.
            for t, token in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self.decoder_input_data[i, t, self.target_token_index[token]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.decoder_target_data[i, t - 1, self.target_token_index[token]] = 1.

        # Reverse-lookup token index to decode sequences back to something readable.
        self.reverse_input_token_index = dict(
            (i, token) for token, i in self.input_token_index.items())
        self.reverse_target_token_index = dict(
            (i, token) for token, i in self.target_token_index.items())

    def train_save(self, epochs=epochs_default):

        self.model = self.define_model()

        print("\n###\nAbout to train model %s with tokenizer %s and optimizer %s\n###\n\n"
              % (__name__, self.tokenizer.__class__.__name__, self.optimizer))

        filename = 'checkpoints/' + self.name

        if os.path.isfile(filename + '.h5'):
            # Load the previous model (layers and weights but NO STATE)
            print("Loading previous weights: %s" % filename)
            self.model.load_weights(filename + '.h5')
        else:
            print("No existing model file found: %s" % filename)

        if not os.path.isfile(filename + '.png'):
            # Plot the model and save it too
            plot_model(self.model, to_file=(filename + '.png'), show_shapes=True)

        if epochs > 0:
            # Prepare checkpoints
            checkpoint = self.get_checkpoint(filename + '.h5')
            logger = CSVLogger(filename + '.csv', separator=',', append=True)
            # Run training
            print("About to fit with batch_size=%d" % self.batch_size)
            history = self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                                     batch_size=self.batch_size,
                                     epochs=epochs,
                                     validation_split=0.2,
                                     callbacks=[checkpoint, logger])
            # Save model
            self.model.save(filename + '.h5')
            # Print the training history
            print(history.history['val_loss'])

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
        if self.include_dropout:
            decoder_dropout = Dropout(0.2)
            decoder_outputs = decoder_dense(decoder_dropout(decoder_outputs_interim))
        else:
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

        if self.encoder_model is None:
            # Redefine encoder model (needs to work even when the model has just been loaded from an h5 file)
            encoder_inputs = model.input[0]  # input_1
            encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
            encoder_states = [state_h_enc, state_c_enc]
            self.encoder_model = Model(encoder_inputs, encoder_states)

            decoder_inputs = model.input[1]  # input_2
            decoder_state_input_h = Input(shape=(self.latent_dim,), name='dec_input_h')
            decoder_state_input_c = Input(shape=(self.latent_dim,), name='dec_input_c')
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_lstm = model.layers[3]
            decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
                decoder_inputs, initial_state=decoder_states_inputs)
            decoder_states = [state_h_dec, state_c_dec]
            decoder_dense = model.layers[4]
            decoder_outputs = decoder_dense(decoder_outputs)
            self.decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs] + decoder_states)

        return self.encoder_model, self.decoder_model

    def translate(self, input_text, verbose=True):

        # tokenize the input sentence
        tokens = self.tokenizer.tokenize([input_text], lang2)

        # Prepare one-hot encoded ndarray like during training
        encoder_input_data = np.zeros(
            (1, self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype='float32')
        for t, token in enumerate(tokens[0]):
            try:
                idx = self.input_token_index[token]
                encoder_input_data[0, t, idx] = 1.
            except KeyError:
                print("No match for char=" + str(token))
                pass

        decoded = self.decode_sequence(encoder_input_data[0:1])
        return self.tokenizer.post_edit(decoded)

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
        decoded_tokens = []
        while not stop_condition:
            model_input = [target_seq] + states_value
            output_tokens, h, c = decoder_model.predict(model_input)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.reverse_target_token_index[sampled_token_index]
            decoded_tokens.append(sampled_token)

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_token == self.CH_END or
                    len(decoded_tokens) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return self.tokenizer.join(decoded_tokens, 'en')

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
            print('Input sentence:', self.other_texts[seq_index])
            print('Decoded sentence:', decoded_sentence)


class AttentionWithDropout(Attention3):

    def __init__(self, name, tokenizer, optimizer):
        Attention3.__init__(self, name, tokenizer, optimizer, include_dropout=True)
        self.batch_size = 12  # Lower batch size since it's more complex


class Attention512(Attention3):

    def __init__(self, name, tokenizer, optimizer):
        Attention3.__init__(self, name, tokenizer, optimizer, latent_dim=512)
