"""
This module helps us examine/debug the different tokenizers
"""

from helpers import *


def summarize_tokenizers():
    # load full dataset
    dataset = load_clean_sentences('both')
    for tokenizer_id, tokenizer_func in tokenizers.items():
        print('Summary of Tokenizer: %s' % tokenizer_id)
        # prepare english tokenizer
        dataset_lang1 = dataset[:, 0]
        eng_lines = tokenizer_func(dataset_lang1, 'en')
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
        other_tokenized = tokenizer_func(dataset_lang2, lang2)
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
    summarize_tokenizers()
