"""
This module defines the various models that will be tested.
Based on a character-level model at https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
"""
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input, LSTM
from keras.models import Model
import numpy as np

training_model = None
encoder_model = None
decoder_model = None

# Assume a problem is the decoded length exceeds this
max_decoder_seq_length=40

def dense_model(src_vocab, target_vocab, src_timesteps, target_timesteps, n_units=256):
    """
    To train:
    encoder_input_data is a 3D array of shape (num_pairs, max_english_sentence_length, num_english_characters) containing a one-hot vectorization of the English sentences.
    decoder_input_data is a 3D array of shape (num_pairs, max_french_sentence_length, num_french_characters) containg a one-hot vectorization of the French sentences.
    decoder_target_data is the same as decoder_input_data but offset by one timestep. decoder_target_data[:, t, :] will be the same as decoder_input_data[:, t + 1, :].
    Requires teacher forcing
    See also https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
    :param src_vocab:
    :param target_vocab:
    :param src_timesteps: int: max sentence length of source language (English)
    :param target_timesteps: max sentence length of target language (Swedish)
    :param n_units:
    :return: The training model
    """
    prepare(src_vocab, target_vocab, n_units)
    return training_model


def prepare(src_vocab, target_vocab, latent_dim):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,src_vocab))
    # Add an embedding layer to process the integer encoded words to give some 'sense' before the LSTM layer
    # encoder_embedding = Embedding(src_vocab, latent_dim)(encoder_inputs)
    # The return_state contructor argument configures a RNN layer to return a list where the first entry is the outputs
    # and the next entries are the internal RNN states. This is used to recover the states of the encoder.
    encoder_lstm = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    # Set up the decoder, using `encoder_states` as initial state of the RNN.
    decoder_inputs = Input(shape=(None,target_vocab))
    # decoder_embedding = Embedding(target_vocab, latent_dim)(decoder_inputs)
    # The return_sequences constructor argument, configuring a RNN to return its full sequence of outputs (instead of
    # just the last output, which the defaults behavior).
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs_interim, _, _  = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    # Attach through a dense layer
    decoder_dense = Dense(target_vocab, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs_interim)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    global training_model
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states
    # Define sampling models
    if 1==2:
        global encoder_model
        encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        # Use decoder_lstm from the training model
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        global decoder_model
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)


def infer_dense(src_vocab, target_vocab, src_timesteps, target_timesteps, n_units):
    prepare(src_vocab, target_vocab, n_units)
    return decoder_model


def decode_sequence(input_seq, target_vocab):
    """
    Decode (translate) the input sequence into natural language in the target language
    :param input_seq:
    :param target_vocab: int: the target vocab size
    :return: the target language sentence output

    """
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, target_vocab))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_word

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, target_vocab))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence
