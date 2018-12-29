"""
This module trains a model and stores it in a file
"""
import tensorflow as tf
from helpers import *

from models import attention3
from tokenizers import SimpleLines, Word2Phrase, ReplaceProper, PosTag, LetterByLetter, Hyphenate

print("VERSION", tf.Session(config=tf.ConfigProto(log_device_placement=True)))

models = {
    # 'simple': simple.Simple,
    # 'dense': attention.Attention
    # 'let2let': let2let.Let2Let,
    #    'dense2': attention2.Attention2
    'att3': attention3.Attention3
}

tokenizers = {
  #  'a': SimpleLines(),
    'b': Hyphenate(),
    'c': Word2Phrase(),
    'd': ReplaceProper(),
    'e': PosTag(),
     'l': LetterByLetter(),
}

# key becomes part of the model name, the value is passed in the optimizer= parameter
optimizer_opts = {
    # 'sgd': 'sgd',  # default parameters (reported to be more 'stable' than adam)
    # 'rmsprop': 'rmsprop',  # default lr=0.001
    # 'rmsprop2': optimizers.RMSprop(lr=0.01),  # same as previous but with 10x higher learning rate
    'adam': 'adam'
}


def train_all():
    """Train the models and tokenizer permutations"""
    for model_name, model_class in models.items():
        for token_id, tokenizer in tokenizers.items():
            for opt_id, optimizer in optimizer_opts.items():
                # save each one
                filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                try:
                    print("About to train %s" % filename)
                    model_obj = model_class(filename, tokenizer, optimizer)
                    model_obj.train_save()
                except:
                    print("Error training model: " + filename)
                    traceback.print_exc()
                    pass


def summarize_tokenizers():
    # load full dataset
    dataset = load_clean_sentences('both')
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


if __name__ == '__main__':
    # Avoid memory errors on Mac
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # or `pip install nomkl`
    # summarize_tokenizers()
    # Start the training
    train_all()
