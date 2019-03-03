# Analyze average training times and output as markdown tables
#
from statistics import mean

from numpy import genfromtxt
from tabulate import tabulate

from train import models, tokenizers, optimizer_opts, version


def times_per_model():
    for model_name in list(models.keys()):
        headers = [""]
        for token_id in list(tokenizers.keys()):
            headers.append(token_id)
        table = []
        for opt_id in list(optimizer_opts.keys()):
            row = [opt_id]
            for token_id in list(tokenizers.keys()):
                # get the average training time per epoch
                label = model_name + '_' + token_id + '_' + opt_id
                filename = label + '_' + version
                try:
                    history = genfromtxt('checkpoints/' + filename + '-time.txt')
                    mean_times = mean(history)
                    row.append(int(round(mean_times)))
                except:
                    # No model trained yet
                    # print('No model logs for: ' + filename)
                    row.append("")
            table.append(row)
        # Print this model's data
        print("\n\n### Mean Training Times - Model %s:\n" % model_name)
        print(tabulate(table, headers, tablefmt="github"))


def times_per_optimizer():
    for opt_id in list(optimizer_opts.keys()):
        headers = [""]
        for token_id in list(tokenizers.keys()):
            headers.append(token_id)
        table = []
        for model_name in list(models.keys()):
            row = [model_name]
            for token_id in list(tokenizers.keys()):
                # get the average training time per epoch
                label = model_name + '_' + token_id + '_' + opt_id
                filename = label + '_' + version
                try:
                    history = genfromtxt('checkpoints/' + filename + '-time.txt')
                    mean_times = mean(history)
                    row.append(int(round(mean_times)))
                except:
                    # No model trained yet
                    # print('No model logs for: ' + filename)
                    row.append("")
            table.append(row)
        # Print this model's data
        print("\n\n### Mean Training Times - Optimizer %s:\n" % opt_id)
        print(tabulate(table, headers, tablefmt="github"))


def overview_epochs_per_model():
    for model_name in list(models.keys()):
        headers = [""]
        for token_id in list(tokenizers.keys()):
            headers.append(token_id)
        table = []
        for opt_id in list(optimizer_opts.keys()):
            row = [opt_id]
            for token_id in list(tokenizers.keys()):
                # get the average training time per epoch
                label = model_name + '_' + token_id + '_' + opt_id
                filename = label + '_' + version
                try:
                    history = genfromtxt('checkpoints/' + filename + '-time.txt')
                    row.append(len(history))
                except:
                    # No model trained yet
                    # print('No model logs for: ' + filename)
                    row.append("")
            table.append(row)
        # Print this model's data
        print("\n\n### Training Epochs - Model %s:\n" % model_name)
        print(tabulate(table, headers, tablefmt="github"))


def loss_per_model():
    """
    Table with the lowest val_loss achieved per mode, tokenizer and optimizer
    """
    for model_name in list(models.keys()):
        headers = [""]
        for token_id in list(tokenizers.keys()):
            headers.append(token_id)
        table = []
        for opt_id in list(optimizer_opts.keys()):
            row = [opt_id]
            for token_id in list(tokenizers.keys()):
                # get the average training time per epoch
                label = model_name + '_' + token_id + '_' + opt_id
                filename = label + '_' + version
                try:
                    history = genfromtxt('checkpoints/' + filename + '.csv', delimiter=",", skip_header=1)
                    min_loss = min(history[:, 2])
                    row.append(round(min_loss, 2))
                except:
                    # No model trained yet
                    # print('No model logs for: ' + filename)
                    row.append("")
            table.append(row)
        # Print this model's data
        print("\n\n### Min loss - Model %s:\n" % model_name)
        print(tabulate(table, headers, tablefmt="github"))

def loss_per_optimizer():
    for opt_id in list(optimizer_opts.keys()):
        headers = [""]
        for token_id in list(tokenizers.keys()):
            headers.append(token_id)
        table = []
        for model_name in list(models.keys()):
            row = [model_name]
            for token_id in list(tokenizers.keys()):
                # get the average training time per epoch
                label = model_name + '_' + token_id + '_' + opt_id
                filename = label + '_' + version
                try:
                    history = genfromtxt('checkpoints/' + filename + '.csv', delimiter=",", skip_header=1)
                    min_loss = min(history[:, 2])
                    row.append(round(min_loss, 2))
                except:
                    # No model trained yet
                    # print('No model logs for: ' + filename)
                    row.append("")
            table.append(row)
        # Print this model's data
        print("\n\n### Min Loss - Optimizer %s:\n" % opt_id)
        print(tabulate(table, headers, tablefmt="github"))


if __name__ == '__main__':
    overview_epochs_per_model()
    times_per_model()
    times_per_optimizer()
    loss_per_model()
    loss_per_optimizer()

