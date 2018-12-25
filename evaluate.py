from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

import train
from helpers import *


def evaluate_model(model_obj, raw_dataset):
    """
    evaluate the skill of the model
    :param model_obj: models.base.BaseModel the model container
    :param raw_dataset: The validation dataset language pairs prior to tokenizer (i.e. actual strings)
    """
    actual, predicted = list(), list()
    for i, pair in enumerate(raw_dataset):
        # translate encoded source text
        raw_target, raw_src = pair[0], pair[1]
        translation = model_obj.translate(raw_src)
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


def eval_model(model_obj):
    """
    :param model_obj:
    :type model_obj: models.base.BaseModel
    :return: BLEU-4 score for this model
    :rtype: float
    """
    print('### About to evaluate model %s with tokenizer %s' % (model_obj.name, model_obj.tokenizer.__class__.__name__))
    # load datasets
    # dataset = load_clean_sentences('both')
    # train = load_clean_sentences('train')
    test = load_clean_sentences('test')
    # prepare english tokenizer
    # eng_tokenized = tokenizer_func(dataset[:, 0], 'en')
    # if filename.startswith('dense'): eng_tokenized = mark_ends(eng_tokenized)
    # eng_tokenizer = create_tokenizer(eng_tokenized)
    # prepare other tokenizer
    # model_obj.update(dataset)
    # trainX = encode_sequences(other_tokenizer, tokenizer_func(train[:, 1], lang2), pad_length)
    # testX = encode_sequences(other_tokenizer, tokenizer_func(test[:, 1], lang2), pad_length)
    # load model
    model = load_model('checkpoints/' + model_obj.name + '.h5')
    print(model.summary())
    # test on some training sequences
    # print('Evaluating training set: train=%s' % (str(train)))
    # evaluate_model(model_obj, train)
    # test on some test sequences
    print('Evaluating testing set: test=%s' % (str(test)))
    test_bleu4 = evaluate_model(model_obj, test)
    return test_bleu4


def evaluate_all():
    summary = {}
    for model_name, model_class in train.models.items():
        for token_id, tokenizer in train.tokenizers.items():
            for opt_id, optimizer in train.optimizer_opts.items():
                try:
                    # define a unique name for this combination
                    filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                    # prepare the attention decoder model (with a hack)
                    model_obj = model_class(filename, tokenizer, optimizer)
                    model_obj.train_save(epochs=0)
                    # save each one
                    test_bleu4 = eval_model(model_obj)
                    summary[filename] = test_bleu4
                except:
                    traceback.print_exc()
                    pass
    # print out the summary test BLEU-4 scores
    for model_name, score in summary.items():
        print('%s=%f' % (model_name, score))


if __name__ == '__main__':
    evaluate_all()
