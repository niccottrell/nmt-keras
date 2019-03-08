"""
Evaluate a small set of similar pairs for grammar handling
"""
from evaluate import *

# load datasets
from train import matches

full_dataset = config.data.load_clean_sentences('both')
grammar_dataset = array([
    # The following examples do not exist in the training or test sets
    ["I eat a sandwich .", "Jag äter en smörgås ."],
    ["You eat a sandwich .", "Du äter en smörgås ."],
    ["You eat a cheese sandwich .", "Du äter en ostsmörgås ."],
    ["You eat the sandwich .", "Du äter smörgåsen ."],
    ["You ate the sandwich .", "Du åt smörgåsen ."],
    # This example exists in the data set
    ["Tom is looking at me .", "Tom tittar på mig ."]
])



def sample_all(model_filter=None, token_filter=None, opt_filter=None):
    for model_name, model_class in train.models.items():
        if matches(model_filter, model_name):
            for token_id, tokenizer_obj in train.tokenizers.items():
                if matches(token_filter, token_id):
                    for opt_id, optimizer in train.optimizer_opts.items():
                        if matches(opt_filter, opt_id):
                            # save each one
                            filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                            # load model
                            model_obj = model_class(filename, tokenizer_obj, optimizer)
                            model_obj.train_save(epochs=0)  # load weights
                            # test on some training sequences
                            print('Evaluating manual set: ' + model_name)
                            evaluate_model(model_obj, grammar_dataset)


if __name__ == '__main__':
    sample_all(model_filter="attbidi")
