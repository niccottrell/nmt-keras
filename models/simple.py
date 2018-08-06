"""
This module defines the various models that will be tested
"""
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input, LSTM, GRU
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential, Model


def simple_model(src_vocab, target_vocab, src_timesteps, target_timesteps, n_units):
    """
     define a simple "naive?" NMT model using LSTM as the RNN Cell Type with no attention mechanism
     with fixed/max sentence length
    :param :src_timesteps: max length of src language sentences
    :param :target_timesteps: max length of targets language sentences
    :param :n_units: dimensionality of the LTSM layers (more dimensions => more accurate meaning => longer to train => better. After 2k not much improvement) Also often labelled latent_dim ?
    :return: A Keras model instance.
    """
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True)) # builds "thought" mappings
    model.add(LSTM(n_units)) # RNN Cell (encoder?) return_state=False by default
    model.add(RepeatVector(target_timesteps)) # Direction of Encoder Input = forward?
    model.add(LSTM(n_units, return_sequences=True))  # RNN Cell (decoder?)
    model.add(TimeDistributed(Dense(target_vocab, activation='softmax')))
    return model
