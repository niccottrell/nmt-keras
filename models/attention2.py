"""
This module defines a word-level model based on a character-level model at https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py and lstm_seq2seq_restore.py and https://github.com/roatienza/Deep-Learning-Experiments/blob/master/keras/seq2seq/seq2seq_translate.py
This starts with a built-in Embedding layer - which accepts a sequence of integers rather than one-hot encoded input
"""
from keras.layers import Dense, Embedding, Input, LSTM
from keras.models import Model

import numpy as np

import helpers

print(helpers.__file__)

# Assume a problem is the decoded length exceeds this
max_decoder_seq_length = 40


def dense_model(src_vocab, target_vocab, src_timesteps, target_timesteps, latent_dim=256):
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

    # Define an input sequence and process it - where the shape is a sequence of integer of variable length
    encoder_inputs = Input(shape=(None,), name='enc_inputs')
    # Add an embedding layer to process the integer encoded words to give some 'sense' before the LSTM layer
    encoder_embedding = Embedding(src_vocab, latent_dim, name='enc_embedding')(encoder_inputs)
    # The return_state constructor argument configures a RNN layer to return a list where the first entry is the outputs
    # and the next entries are the internal RNN states. This is used to recover the states of the encoder.
    encoder_outputs, state_h, state_c = LSTM(latent_dim,
                                        return_state=True, name='encoder_lstm')(encoder_embedding)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state of the RNN.
    decoder_inputs = Input(shape=(None, ), name='dec_inputs')
    decoder_embedding = Embedding(target_vocab, latent_dim, name='dec_embedding')(decoder_inputs)
    # The return_sequences constructor argument, configuring a RNN to return its full sequence of outputs (instead of
    # just the last output, which the defaults behavior).
    decoder_lstm = LSTM(latent_dim, return_sequences=True, name='dec_lstm')(decoder_embedding, initial_state=encoder_states)
    decoder_outputs = Dense(target_vocab, activation='softmax', name='dec_outputs')(decoder_lstm)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return training_model


def infer_models(model, latent_dim=256):
    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states
    # Define sampling models

    # target_vocab = model.output_shape[2]

    encoder_inputs = model.input[0]  # name=enc_inputs (Input(shape=(None,)))
    encoder_lstm = model.get_layer(name='encoder_lstm')
    encoder_outputs, state_h_enc, state_c_enc = encoder_lstm.output  # lstm_1 (LSTM(latent_dim, return_state=True))
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1] # (Input(shape=(None,)))
    decoder_embedding = model.get_layer(name='dec_embedding')(decoder_inputs) # (Embedding(target_vocab, latent_dim, ...))
    decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')  # named to avoid conflict
    decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    # Use decoder_lstm from the training model
    decoder_lstm = model.get_layer(name='dec_lstm') # dec_lstm
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    # decoder_dense = model.layers[6]  # name=dec_outputs, should match Dense(target_vocab, activation='softmax')
    decoder_dense = model.get_layer(name='dec_outputs') # should match Dense(target_vocab, activation='softmax')
    # decoder_dense = Dense(target_vocab, activation='softmax', name='dec_outputs')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return (encoder_model, decoder_model)


def decode_sequence(input_seq, model, tokenizer):
    """
    Decode (translate) the input sequence into natural language in the target language
    :param input_seq: list(int): Sequence of integers representing words from the tokenizer
    :param model: Model
    :param tokenizer: Tokenizer
    :return: the target language sentence output

    """

    encoder_model, decoder_model = infer_models(model)

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # the target vocab size (e.g. English)
    target_vocab = decoder_model.input_shape[0][2]

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, target_vocab))
    # Populate the first character of target sequence with the start character.
    start_seq = tokenizer.word_index['\t']  # tokenizer.texts_to_sequences(['\t'])
    target_seq[0, 0, start_seq] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = helpers.word_for_id(sampled_token_index, tokenizer)
        decoded_sentence += sampled_word

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        else:
            decoded_sentence += ' '

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, target_vocab))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence
