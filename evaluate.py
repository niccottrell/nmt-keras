from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

from helpers import *
from train import train_save


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    """
    generate target given source sequence (for all predictions)
    :type model: Model
    :type tokenizer: Tokenizer
    :type source: The input data
    :return the most likely prediction
    """
    translations = list()
    predictions = model.predict(source, verbose=0)
    for preduction in predictions:
        integers = [argmax(vector) for vector in preduction]
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
    evaluate the skill of the model
    :param model: Model the model with weights already trained
    :param tokenizer: Tokenizer run on the target language dataset (identical to when the model was trained)
    :param sources: The sequence in the other language (encoded as integers, but not yet 1-hot encoded)
    :param raw_dataset: The validation dataset language pairs prior to tokenizer (i.e. actual strings)
    """
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        if model != simple.simple_model:
            vocab_size = model.input_shape[0][2]
            # encode to one-hot ndarray (3-dimensions)
            source_encoded = encode_output(source, vocab_size)
            translation = attention.decode_sequence(source_encoded, tokenizer)
        else:
            translation = predict_sequence(model, tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        if i < 20:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
    # calculate BLEU score (at the corpus level)
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-4: %f' % bleu4)
    return bleu4


def eval_model(model_name, tokenizer_func):
    print('### About to evaluate model %s with tokenizer %s' % (model_name, tokenizer_func.__name__))
    # load datasets
    dataset = load_clean_sentences('both')
    train = load_clean_sentences('train')
    test = load_clean_sentences('test')
    # prepare english tokenizer
    eng_tokenized = tokenizer_func(dataset[:, 0], 'en')
    if model_name.startswith('dense_'): eng_tokenized = mark_ends(eng_tokenized)
    eng_tokenizer = create_tokenizer(eng_tokenized)
    # prepare other tokenizer
    dataset_lang2 = dataset[:, 1]
    other_tokenizer = create_tokenizer(tokenizer_func(dataset_lang2, lang2))
    other_tokenized = tokenizer_func(dataset_lang2, lang2)
    other_length = max_length(other_tokenized)
    # prepare/encode/pad data (pad to length of target language)
    trainX = encode_sequences(other_tokenizer, other_length, tokenizer_func(train[:, 1], lang2))
    testX = encode_sequences(other_tokenizer, other_length, tokenizer_func(test[:, 1], lang2))
    # load model
    model = load_model('checkpoints/' + model_name + '.h5')
    print(model.summary())
    # test on some training sequences
    print('Evaluating training set: train=%s, trainX=%s' % (str(train.shape), str(trainX.shape)))
    evaluate_model(model, eng_tokenizer, trainX, train)
    # test on some test sequences
    print('Evaluating testing set: test=%s, testX=%s' % (str(test.shape), str(testX.shape)))
    test_bleu4 = evaluate_model(model, eng_tokenizer, testX, test)
    return test_bleu4


def evaluate_all():
    summary = {}
    for model_name, model_func in models.items():
        for token_id, tokenizer in tokenizers.items():
            for opt_id, optimizer in optimizers.items():
                # prepare the attention decoder model (with a hack)
                train_save(model_func, tokenizer, 'blah', epochs=0)
                # save each one
                filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                try:
                    test_bleu4 = eval_model(filename, tokenizer)
                    summary[filename] = test_bleu4
                except:
                    traceback.print_exc()
                    pass
    # print out the summary test BLEU-4 scores
    for model_name, score in summary.items():
        print('%s=%f' % (model_name, score))


evaluate_all()
