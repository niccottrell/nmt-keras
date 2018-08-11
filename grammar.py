"""
Evaluate a small set of similar pairs for grammar handling
"""
from evaluate import *

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

if __name__ == '__main__':
    sample_all()