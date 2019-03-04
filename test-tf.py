import tensorflow as tf
import sys

# Print TensorFlow info
print(tf.__version__)
tf.Session()

# Print memory max size
print("%x" % sys.maxsize, sys.maxsize > 2**32) # if second result is true we're running 64-bit