"""
This module trains a model and stores it in a file
"""
import tensorflow as tf
from keras import optimizers
from keras.backend.tensorflow_backend import set_session

from config import epochs_default
from helpers import *

from models import attention3, simple, let2let
from tokenizers import SimpleLines, Word2Phrase, ReplaceProper, PosTag, LetterByLetter, Hyphenate

print("VERSION", tf.Session(config=tf.ConfigProto(log_device_placement=True)))

models = {
    'simple': simple.Simple,
    'simple2': simple.Simple2,
    # 'let2let': let2let.Let2Let,
    'att': attention3.Attention3,
    'att512': attention3.Attention512,
    'attdropout': attention3.AttentionWithDropout,
    'attrev': attention3.AttentionReverse,
    'attbidi': attention3.AttentionBidi
}

tokenizers = {
    'a': SimpleLines(),
    'b': Hyphenate(),
    'c': Word2Phrase(),
    'd': ReplaceProper(),
    'e': PosTag(),
    'l': LetterByLetter(),
}

# key becomes part of the model name, the value is passed in the optimizer= parameter
optimizer_opts = {
    'sgd': 'sgd',  # default parameters (reported to be more 'stable' than adam)
    'sgd2': optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),  # small decay
    'rmsprop': 'rmsprop',  # default lr=0.001
    'rmsprop2': optimizers.RMSprop(lr=0.01),  # same as previous but with 10x higher learning rate
    'rmsprop3': optimizers.RMSprop(lr=0.01, decay=0.00001),  # same as previous but with decay
    'adam': 'adam'
}


def train_all(model_filter=None, token_filter=None, opt_filter=None, epochs=epochs_default):
    """Train the models and tokenizer permutations"""
    log = []
    for model_name, model_class in models.items():
        if matches(model_filter , model_name):
            for token_id, tokenizer in tokenizers.items():
                if matches(token_filter, token_id):
                    for opt_id, optimizer in optimizer_opts.items():
                        if matches(opt_filter, opt_id):
                            # save each one
                            filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                            try:
                                print("About to train %s" % filename)
                                model_obj = model_class(filename, tokenizer, optimizer)
                                model_obj.train_save(epochs=epochs)
                                log.append(filename + " OK")
                            except:
                                print("Error training model: " + filename)
                                traceback.print_exc()
                                log.append(filename + " Error")
                                pass
    print("Training complete: %s" % "\n".join(log))


def summarize_tokenizers():
    # load full dataset
    dataset = config.data.load_clean_sentences('both')
    for tokenizer_id, tokenizer in tokenizers.items():
        print('Summary of Tokenizer: %s' % tokenizer_id)
        # prepare english tokenizer
        dataset_lang1 = dataset[:, 0]
        eng_lines = tokenizer.tokenize(dataset_lang1, 'en')
        for i, source in enumerate(eng_lines):
            if i < 20:
                print('English tokens=%s' % (source))
            else:
                break
        eng_tokenizer = create_tokenizer(eng_lines)
        eng_vocab_size = len(eng_tokenizer.word_index) + 1
        eng_length = max_length(eng_lines)
        print('English Vocabulary Size: %d' % eng_vocab_size)
        print('English Max Length: %d' % eng_length)
        # prepare other (source language) tokenizer
        dataset_lang2 = dataset[:, 1]
        other_tokenized = tokenizer.tokenize(dataset_lang2, lang2)
        for i, source in enumerate(other_tokenized):
            if i < 20:
                print('Other tokens=%s' % (source))
            else:
                break
        other_tokenizer = create_tokenizer(other_tokenized)
        other_vocab_size = len(other_tokenizer.word_index) + 1
        other_length = max_length(other_tokenized)
        print('Other Vocabulary Size: %d' % other_vocab_size)
        print('Other Max Length: %d' % other_length)


def matches(filter: string, name: string):
    """
    :return: True if we should train this component based on the filter
    """
    return filter is None or name.startswith(filter)


if __name__ == '__main__':
    # Avoid memory errors on Mac
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # or `pip install nomkl`
    # Tweak TensorFlow config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    # summarize_tokenizers()
    # Start the training
    # train_all(model_filter='attdropout')
    # train_all(model_filter='att')
    # train_all(model_filter='att512')
    train_all(model_filter='attbidi', token_filter='a', opt_filter='rmsprop2')
    # train_all(opt_filter='sgd2')
