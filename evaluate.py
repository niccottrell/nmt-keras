from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

import train
from helpers import *


def evaluate_model(model_obj, model, tokenizer, sources, raw_dataset):
    """
    evaluate the skill of the model
    :param model_obj: models.base.BaseModel the model container
    :param model: keras.models.Model the model with weights already trained
    :param tokenizer: Tokenizer run on the target language dataset (identical to when the model was trained)
    :param sources: list(list(int)): The sequence in the other language (encoded as integers, but not yet 1-hot encoded)
    :param raw_dataset: The validation dataset language pairs prior to tokenizer (i.e. actual strings)
    """
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        translation = model_obj.translate(model, tokenizer, source)
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


def eval_model(model_obj, filename, tokenizer_func):
    """
    :param model_obj:
    :type model_obj: models.base.BaseModel
    :param filename:
    :type filename: str
    :param tokenizer_func: function
    :return:
    """
    print('### About to evaluate mod el %s with tokenizer %s' % (filename, tokenizer_func.__name__))
    # load datasets
    dataset = load_clean_sentences('both')
    train = load_clean_sentences('train')
    test = load_clean_sentences('test')
    # prepare english tokenizer
    eng_tokenized = tokenizer_func(dataset[:, 0], 'en')
    if filename.startswith('dense'): eng_tokenized = mark_ends(eng_tokenized)
    eng_tokenizer = create_tokenizer(eng_tokenized)
    # prepare other tokenizer
    dataset_lang2 = dataset[:, 1]
    other_tokenizer = create_tokenizer(tokenizer_func(dataset_lang2, lang2))
    other_tokenized = tokenizer_func(dataset_lang2, lang2)
    other_length = max_length(other_tokenized)
    # prepare/encode/pad data (pad to length of target language)
    pad_length = other_length if filename.startswith('simple') else None
    trainX = encode_sequences(other_tokenizer, tokenizer_func(train[:, 1], lang2), pad_length)
    testX = encode_sequences(other_tokenizer, tokenizer_func(test[:, 1], lang2), pad_length)
    # load model
    model = load_model('checkpoints/' + filename + '.h5')
    print(model.summary())
    # test on some training sequences
    print('Evaluating training set: train=%s, trainX=%s' % (str(train), str(trainX)))
    evaluate_model(model_obj, model, eng_tokenizer, trainX, train)
    # test on some test sequences
    print('Evaluating testing set: test=%s, testX=%s' % (str(test), str(testX)))
    test_bleu4 = evaluate_model(model_obj, model, eng_tokenizer, testX, test)
    return test_bleu4


def evaluate_all():
    summary = {}
    for model_name, model_obj in train.models.items():
        for token_id, tokenizer in train.tokenizers.items():
            for opt_id, optimizer in train.optimizer_opts.items():
                try:
                    # prepare the attention decoder model (with a hack)
                    model_obj.train_save(tokenizer, model_name, optimizer, mode='readonly')
                    # save each one
                    filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                    test_bleu4 = eval_model(model_obj, filename, tokenizer)
                    summary[filename] = test_bleu4
                except:
                    traceback.print_exc()
                    pass
    # print out the summary test BLEU-4 scores
    for model_name, score in summary.items():
        print('%s=%f' % (model_name, score))


if __name__ == '__main__':
    evaluate_all()
