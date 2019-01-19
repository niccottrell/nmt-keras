# Source: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

import matplotlib.pyplot as plt
from numpy import genfromtxt

# Fit the model
# history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
from train import models, tokenizers, optimizer_opts, version

# record all the models found
entries = []

model_filter = None
token_filter = 'a'
opt_filter = None

for model_name, model_class in models.items():
    if model_filter is None or model_filter == model_name:
        for token_id, tokenizer in tokenizers.items():
            if token_filter is None or token_filter == token_id:
                for opt_id, optimizer in optimizer_opts.items():
                    if opt_filter is None or opt_filter == opt_id:
                        # save each one
                        filename = model_name + '_' + token_id + '_' + opt_id + '_' + version
                        try:
                            history = genfromtxt('checkpoints/' + filename + '.csv', delimiter=',')
                            entries.append(filename)
                            plt.plot(history[:, 2])
                            print("Plotted: " + filename)
                        except:
                            # No model trained yet
                            print('No model logs for: ' + filename)

# summarize history for loss
plt.title('model loss')
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(entries, loc='upper right')
plt.show()

file_out = "training_loss.png"
plt.savefig(file_out)
print("Wrote plot to " + file_out)
