import time

from nltk.translate.bleu_score import corpus_bleu

import config
import train
from helpers import *
from prepare import n_test


def evaluate_model(model_obj, raw_dataset, verbose=True):
    """
    evaluate the skill of the model
    :param verbose: if true, print out more debug information
    :param model_obj: models.base.BaseModel the model container
    :param raw_dataset: The validation dataset language pairs prior to tokenizer (i.e. actual strings)
    :return: BLEU-1 score
    """
    if verbose: print('About to evaluate test set of size %d' % len(raw_dataset))
    actual, predicted = list(), list()
    for i, pair in enumerate(raw_dataset):
        # translate encoded source text
        raw_target, raw_src = pair[0], pair[1]
        translation = model_obj.translate(raw_src, verbose)
        if i < 20:
            if verbose: print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        elif i % 80 == 0:
            print('.', flush=True)
        else:
            print('.', end='', flush=True)
        actual.append(raw_target.split())
        predicted.append(translation.split())
    # calculate BLEU score (at the corpus level)
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    if verbose:
        bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
        bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
        print('BLEU-1: %f' % bleu1)
        print('BLEU-2: %f' % bleu2)
        print('BLEU-3: %f' % bleu3)
        print('BLEU-4: %f' % bleu4)
    return bleu1


def eval_model(model_obj, verbose=True):
    """
    :param verbose: If true, print example texts etc
    :param model_obj: The model with weights set
    :type model_obj: models.base.BaseModel
    :return: BLEU-4 score for this model
    :rtype: float
    """
    print('### About to evaluate model %s with tokenizer %s' % (model_obj.name, model_obj.tokenizer.__class__.__name__))
    # load datasets
    # dataset = load_clean_sentences('both')
    # train = load_clean_sentences('train')
    test = config.data.load_clean_sentences('test')
    # prepare english tokenizer
    # eng_tokenized = tokenizer_func(dataset[:, 0], 'en')
    # if filename.startswith('dense'): eng_tokenized = mark_ends(eng_tokenized)
    # eng_tokenizer = create_tokenizer(eng_tokenized)
    # prepare other tokenizer
    # model_obj.update(dataset)
    # trainX = encode_sequences(other_tokenizer, tokenizer_func(train[:, 1], lang2), pad_length)
    # testX = encode_sequences(other_tokenizer, tokenizer_func(test[:, 1], lang2), pad_length)
    # load model
    # model = load_model('checkpoints/' + model_obj.name + '.h5')
    # print(model.summary())
    # test on some training sequences
    # print('Evaluating training set: train=%s' % (str(train)))
    # evaluate_model(model_obj, train)
    # test on some test sequences
    # print('Evaluating testing set: test=%s' % (str(test)))
    test_bleu = evaluate_model(model_obj, test, verbose)
    return test_bleu


def evaluate_all(model_filter=None, token_filter=None, opt_filter=None):
    summary = {}
    time_taken = {}
    test = config.data.load_clean_sentences('test')
    for model_name, model_class in train.models.items():
        if model_filter is None or model_filter == model_name:
            for token_id, tokenizer in train.tokenizers.items():
                if token_filter is None or token_filter == token_id:
                    for opt_id, optimizer in train.optimizer_opts.items():
                        if opt_filter is None or opt_filter == opt_id:
                            try:
                                # define a unique name for this combination
                                filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                                # prepare the attention decoder model (with a hack)
                                model_obj = model_class(filename, tokenizer, optimizer)
                                model_obj.train_save(epochs=0)  # load pre-trained weights
                                # evaluate each model (and time it)
                                start = time.clock()
                                test_bleu = evaluate_model(model_obj, test, False)
                                summary[filename] = test_bleu
                                elapsed = time.clock() - start
                                time_taken[filename] = elapsed
                            except:
                                traceback.print_exc()
                                pass
    # print out the summary test BLEU-4 scores
    for model_name, score in summary.items():
        total_time = time_taken[model_name]
        print('%s=%f (took %d s per sentence)' % (model_name, score, total_time/n_test))
    return summary


if __name__ == '__main__':
    evaluate_all(model_filter='att')
