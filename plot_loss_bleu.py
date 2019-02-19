# Plot val_loss progress vs BLEU scores
#
# Source: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
#

import csv
import traceback
import matplotlib.pyplot as plt
from numpy import genfromtxt
from evaluate import evaluate_all
from train import models, tokenizers, optimizer_opts, version


def plot_bleu(model_filter=None, token_filter=None, opt_filter=None):

    # Calculate all scores first
    bleu_scores = evaluate_all(model_filter, token_filter, opt_filter)
    print('Finished calculating BLEU scores: %s' % "\n".join(bleu_scores.keys()))

    file_out = "plots/bleu"
    if model_filter is not None:
        file_out += '_' + model_filter
    if token_filter is not None:
        file_out += '_' + token_filter
    if opt_filter is not None:
        file_out += '_' + opt_filter

    labels = []
    xs = []
    ys = []

    with open(file_out + '.csv', mode='w') as csv_file: # overwrite any existing file
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # For each, lookup the val_loss and plot them
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
                                    bleu = bleu_scores[filename]
                                    val_loss = history[:, 2][-1]
                                    csv_writer.writerow([filename, bleu, val_loss])
                                    print("bleu=%s, val_loss=%s" % (bleu, val_loss))
                                    labels.append(label)
                                    xs.append(bleu)
                                    ys.append(val_loss)
                                    # if isinstance(bleu, numbers.Number):
                                    #   plt.plot(bleu, val_loss, label=filename, markersize=12)
                                    print("Plotted: " + filename)
                                except UserWarning as uw:
                                    # print(uw)
                                    traceback.print_exc()
                                    # No model trained yet
                                    print('No val_los history: ' + filename)
                                except Exception as e:
                                    # print(e)
                                    traceback.print_exc()
                                    # No model trained yet
                                    print('No model logs for: ' + filename)

    if not labels:
        print("No matching data for filter %s, %s, %s" % (model_filter, token_filter, opt_filter))

    else:
        plt.scatter(xs, ys, marker='o')
        for label, x, y in zip(labels, xs, ys):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        # summarize history for loss
        plt.style.use('seaborn-whitegrid')
        plt.title('BLEU-1 vs val_loss')
        plt.ylabel('val_loss')
        plt.xlabel('BLEU')
        # plt.show()
        plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)

        plt.savefig(file_out + '.png')
        print("Wrote plot to " + file_out)


def plot_bleu_all():
    for model_name in (list(models.keys()) + [None]):
        for token_id in (list(tokenizers.keys()) + [None]):
            for opt_id in (list(optimizer_opts.keys()) + [None]):
                plot_bleu(model_name, token_id, opt_id)


if __name__ == '__main__':
    # plot_bleu(None, 'a', 'adam') # Plot all models, but only with SimpleLines and Adam optimizer
    plot_bleu_all()
