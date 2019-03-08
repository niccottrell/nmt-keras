import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import config

# Load sentence lengths (characters)
raw_dataset = config.data.load_clean_sentences()
d = []
for pair in raw_dataset:
    d.append([len(pair[0]), len(pair[1])])

langs = ['English', 'Swedish']
data = pd.DataFrame(d, columns=langs)

plt.figure()  # reset the plot
for lang in langs:
    # Draw the density plot
    sns.distplot(data[lang], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3},
                 label=lang)

# Plot formatting
plt.legend(title='Sentence length')
plt.xlabel('Sentence length (characters)')
plt.ylabel('Density')
# plt.show()
plt.savefig("dataset.png")