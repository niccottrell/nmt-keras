"""
This module trains a model and stores it in a file
"""
import os

import tensorflow as tf
from helpers import *

from models import let2let


print("VERSION", tf.Session(config=tf.ConfigProto(log_device_placement=True)))

models = {
    # 'simple': simple.Simple(),
    # 'dense': attention.Attention()
    'let2let': let2let.Let2Let(),
#    'dense2': attention2.Attention2()
}

tokenizers = {
    'a': simple_lines,
    # 'b': hyphenate_lines,
    # 'c': word2phrase_lines,
    # 'd': replace_proper_lines,
    # 'e': pos_tag_lines
}

# key becomes part of the model name, the value is passed in the optimizer= parameter
optimizer_opts = {
    # 'sgd': 'sgd',  # default parameters (reported to be more 'stable' than adam)
    # 'rmsprop': 'sgd',  # default lr=0.001
    # 'rmsprop2': optimizers.RMSprop(lr=0.01),  # same as previous but with 10x higher learning rate
    'adam': 'adam'
}


def train_all():
    """Train the models and tokenizer permutations"""
    for model_name, model_func in models.items():
        for token_id, tokenizer in tokenizers.items():
            for opt_id, optimizer in optimizer_opts.items():
                # save each one
                filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                try:
                    model_func.train_save(tokenizer, filename, optimizer)
                except:
                    print("Error training model: " + filename)
                    traceback.print_exc()
                    pass


if __name__ == '__main__':
    # Avoid memory errors on Mac
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # or `pip install nomkl`
    # Start the training
    train_all()
