"""
Evaluate a small set of similar pairs for grammar handling
"""
from evaluate import *

# load datasets
full_dataset = load_clean_sentences('both')
grammar_dataset = array([["I eat a sandwich.", "Jag äter en smörgås."],
                         ["You eat a sandwich.", "Du äter en smörgås."],
                         ["You eat the sandwich.", "Du äter smörgåsen."],
                         ["You ate the sandwich.", "Du åt smörgåsen."]])


def sample_all():
    for token_id, tokenizer_func in train.tokenizers.items():
        for model_name, model_obj in train.models.items():
            # prepare english tokenizer
            dataset_lang1 = full_dataset[:, 0]
            eng_tokenized = tokenizer_func(dataset_lang1, 'en')
            if model_name.startswith('dense'): eng_tokenized = mark_ends(eng_tokenized)
            eng_tokenizer = create_tokenizer(eng_tokenized)
            # prepare other tokenizer
            dataset_lang2 = full_dataset[:, 1]
            other_tokenized = tokenizer_func(dataset_lang2, lang2)
            other_tokenizer = create_tokenizer(other_tokenized)
            other_length = max_length(other_tokenized)
            # prepare data
            pad_length = other_length if model_name == 'simple' else None
            trainX = encode_sequences(other_tokenizer, tokenizer_func(grammar_dataset[:, 1], lang2), pad_length)
            for opt_id, optimizer in train.optimizer_opts.items():
                # save each one
                filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                # load model
                model = load_model('checkpoints/' + filename + '.h5')
                model.name = model_name
                # test on some training sequences
                print('Evaluating manual set: ' + model_name)
                evaluate_model(model_obj, model, eng_tokenizer, trainX, grammar_dataset)


if __name__ == '__main__':
    sample_all()
