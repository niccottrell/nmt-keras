import unittest

import tokenizers
from thirdparty.word2phrase import train_model


class Word2PhraseTests(unittest.TestCase):

    def test_simple1(self):
        iter = [['New', 'York', 'is', 'fun'],
                ['New', 'York', 'is', 'big'],
                ['New', 'York', 'can', 'be', 'scary'],
                ['New', 'York', 'used', 'to', 'be', 'called', 'New', 'Amsterdam']]
        out = train_model(iter, min_count=2, threshold=2.0)
        for row in out:
            print(row)
            self.assertIn('New_York', row)

    def test_real_en(self):
        """
        Test through the helper which will train on the dataset
        """
        lines = ['Take a few days off.']
        word2phrase = tokenizers.Word2Phrase()
        out = word2phrase.tokenize(lines, 'en')
        for row in out:
            print(row)
            self.assertIn('a_few', row)
