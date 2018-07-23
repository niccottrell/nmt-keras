"""
This module defines the various models that will be tested
"""
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input, LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential, Model


def simple_model(src_vocab, target_vocab, src_timesteps, target_timesteps, n_units):
    """
     define a simple NMT model
    :param :n_units: dimensionality of the LTSM layers
    :return: A Keras model instance.
    """
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(target_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(target_vocab, activation='softmax')))
    return model

def dense_model(num_encoder_tokens, target_vocab, src_timesteps, target_timesteps, n_units):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,))
    x = Embedding(num_encoder_tokens, n_units)(encoder_inputs)
    x, state_h, state_c = LSTM(n_units,
                               return_state=True)(x)
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    x = Embedding(target_vocab, n_units)(decoder_inputs)
    x = LSTM(n_units, return_sequences=True)(x, initial_state=encoder_states)
    decoder_outputs = Dense(target_vocab, activation='softmax')(x)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
