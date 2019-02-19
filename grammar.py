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
    for model_name, model_class in train.models.items():
        for token_id, tokenizer_obj in train.tokenizers.items():
            for opt_id, optimizer in train.optimizer_opts.items():
                # save each one
                filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                # load model
                model_obj = model_class(filename, tokenizer_obj, optimizer)
                model_obj.train_save(epochs=0) # load weights
                # test on some training sequences
                print('Evaluating manual set: ' + model_name)
                evaluate_model(model_obj, grammar_dataset)


if __name__ == '__main__':
    sample_all()
