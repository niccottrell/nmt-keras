import train
from evaluate import evaluate_model
from grammar import grammar_dataset
from models import attention3


def print_vocab_summary():
    model_name = 'att512'
    model_class = attention3.Attention512
    opt_id = 'adam'
    for token_id, tokenizer_obj in train.tokenizers.items():
        # save each one
        filename = model_name + '_' + token_id + '_' + opt_id + '_' + train.version
        # load model
        model_obj = model_class(filename, tokenizer_obj, opt_id)
        model_obj.train_save(epochs=0)  # load weights
        # test on some training sequences
        print('Evaluating manual set: ' + model_name)
        evaluate_model(model_obj, grammar_dataset)


if __name__ == '__main__':
    print_vocab_summary()
