import nltk

from data import Mongo

nltk_data='./nltk_data/'
nltk.data.path.append(nltk_data)

epochs_default = 200

latent_dim = 256  # Dimensionality of word-embedding (and so LSTM layer)

batch_size = 64   # Default is 32, and we'll need to turn this down for some models so they fit in the GPU's RAM

data = Mongo()  # Files()