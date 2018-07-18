from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

from helpers import *

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    """

    :type tokenizer: Tokenizer
    """
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        if i < 20:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

def eval_model(model_name):
    # load datasets
    dataset = load_clean_sentences('eng-' + lang2 + '-both.pkl')
    train = load_clean_sentences('eng-' + lang2 + '-train.pkl')
    test = load_clean_sentences('eng-' + lang2 + '-test.pkl')
    # prepare english tokenizer
    eng_tokenizer = create_tokenizer(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(dataset[:, 0])
    # prepare german tokenizer
    other_tokenizer = create_tokenizer(dataset[:, 1])
    other_vocab_size = len(other_tokenizer.word_index) + 1
    other_length = max_length(dataset[:, 1])
    # prepare/encode/pad data (pad to length of target language) TODO: What if eng_length > other_length ??
    trainX = encode_sequences(other_tokenizer, other_length, train[:, 1])
    testX = encode_sequences(other_tokenizer, other_length, test[:, 1])
    # load model
    model = load_model(model_name)
    # test on some training sequences
    print('Evaluating training set')
    evaluate_model(model, eng_tokenizer, trainX, train)
    # test on some test sequences
    print('Evaluating testing set')
    evaluate_model(model, eng_tokenizer, testX, test)

eval_model('model.h5')