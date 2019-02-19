# Plot val_loss progress vs epochs
#
# Source: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
#

import matplotlib.pyplot as plt
from numpy import genfromtxt

# Fit the model
# history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
from train import models, tokenizers, optimizer_opts, version

line_styles = {'simple': ':',
               'att': '-',
               'attdropout': '-.',
               'att512': '--'}

def plot_trainings(model_filter=None, token_filter=None, opt_filter=None):
    # record all the models found
    entries = []

    plt.figure()  # reset the plot
    for model_name, model_class in models.items():
        if model_filter is None or model_filter == model_name:
            for token_id, tokenizer in tokenizers.items():
                if token_filter is None or token_filter == token_id:
                    for opt_id, optimizer in optimizer_opts.items():
                        if opt_filter is None or opt_filter == opt_id:
                            # save each one
                            label = model_name + '_' + token_id + '_' + opt_id
                            filename = label + '_' + version
                            try:
                                history = genfromtxt('checkpoints/' + filename + '.csv', delimiter=',')
                                entries.append(label)
                                plt.plot(history[:, 2], linestyle=line_styles[model_name])
                                print("Plotted: " + filename)
                            except:
                                # No model trained yet
                                print('No model logs for: ' + filename)

    if not entries:
        print("No matching data for filter %s, %s, %s" % (model_filter, token_filter, opt_filter))
    else:
        # summarize history for loss
        plt.title('model loss')
        plt.ylabel('val_loss')
        plt.xlabel('epoch')
        plt.legend(entries, loc='upper right')
        # plt.show()
        # plt.ticklabel_formpltat(style='plain', axis='x', useOffset=False)
        # plt.axis([0, 1,  0, 1])
        plt.ylim(bottom=0, top=1)

        file_out = "plots/training_loss"
        if model_filter is not None:
            file_out += '_' + model_filter
        if token_filter is not None:
            file_out += '_' + token_filter
        if opt_filter is not None:
            file_out += '_' + opt_filter
        plt.savefig(file_out + '.png')
        print("Wrote plot to " + file_out)


def plot_trainings_all():
    for model_name in (list(models.keys()) + [None]):
        for token_id in (list(tokenizers.keys()) + [None]):
            for opt_id in (list(optimizer_opts.keys()) + [None]):
                plot_trainings(model_name, token_id, opt_id)


if __name__ == '__main__':
    # plot_trainings(None, None, 'sgd2')
    plot_trainings_all()
