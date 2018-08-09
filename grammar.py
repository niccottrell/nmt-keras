from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

from helpers import *


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    """
    :type tokenizer: Tokenizer
    """
    translations = list()
    predictions = model.predict(source, verbose=0)
    for p in predictions:
        integers = [argmax(vector) for vector in p]
        target = list()
        for i in integers:
            word = word_for_id(i, tokenizer)
            if word is None:
                break
            target.append(word)
        translations.append(' '.join(target))
    print("Candidates=", translations)
    return translations[0]


def evaluate_model(model, tokenizer, sources, raw_dataset):
    """
    Evaluate a model on a small set of sentence pairs
    :param model: The model to decode/infer with
    :param tokenizer: The tokenizer used on the target language (to convert integer back to words)
    :param sources: The encoded source sentences
    :param raw_dataset: The original pairs to score against
    """
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


# load datasets
dataset = load_clean_sentences('both')
train = array([["I eat a sandwich.", "Jag äter en smörgås."],
               ["You eat a sandwich.", "Du äter en smörgås."],
               ["You eat the sandwich.", "Du äter smörgåsen."],
               ["You ate the sandwich.", "Du åt smörgåsen."]])

def sample_all():
    for token_id, tokenizer_func in tokenizers.items():
        # prepare english tokenizer
        dataset_lang1 = dataset[:, 0]
        eng_tokenized = tokenizer_func(dataset_lang1, 'en')
        eng_tokenizer = create_tokenizer(eng_tokenized)
        # prepare other tokenizer
        dataset_lang2 = dataset[:, 1]
        other_tokenized = tokenizer_func(dataset_lang2, lang2)
        other_tokenizer = create_tokenizer(other_tokenized)
        other_length = max_length(other_tokenized)
        # prepare data
        trainX = encode_sequences(other_tokenizer, other_length, tokenizer_func(train[:, 1], lang2))
        for model_name, model_func in models.items():
            for opt_id, optimizer in optimizers.items():
                # save each one
                filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                # load model
                model = load_model('checkpoints/' + filename + '.h5')
                # test on some training sequences
                print('Evaluating manual set: ' + model_name)
                evaluate_model(model, eng_tokenizer, trainX, train)


sample_all()