"""
This module defines the various models that will be tested
"""
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input, LSTM, GRU
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential, Model


# TODO: model with GRU rather than LSTM
def gru_model(src_vocab, target_vocab, src_timesteps, target_timesteps, n_units):
    """
    See https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
    :param src_vocab:
    :param target_vocab:
    :param src_timesteps:
    :param target_timesteps:
    :param n_units:
    :return:
    """
    encoder_inputs = Input(shape=(None, src_vocab))
    encoder = GRU(n_units, return_state=True)
    encoder_outputs, state_h = encoder(encoder_inputs)

    decoder_inputs = Input(shape=(None, target_vocab))
    decoder_gru = GRU(n_units, return_sequences=True)
    decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
    decoder_dense = Dense(target_vocab, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
