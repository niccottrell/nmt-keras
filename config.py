import nltk
nltk_data='./nltk_data/'
nltk.data.path.append(nltk_data)

epochs_default = 20

latent_dim = 256  # Dimensionality of word-embedding (and so LSTM layer)

batch_size = 64   # TODO: Is this batch size too big?

