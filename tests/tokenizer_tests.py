import unittest
from helpers import *


class TokenizerTests(unittest.TestCase):

    def test_is_in_dict(self):
        self.assertTrue(is_in_dict('dog', 'en'))
        self.assertFalse(is_in_dict('doog', 'en'))
        self.assertTrue(is_in_dict('dog', 'sv'))
        self.assertTrue(is_in_dict('dög', 'sv'))
        self.assertFalse(is_in_dict('dug', 'sv'))

    def test_is_proper_sv(self):
        # Yakuhako is a made up string
        self.assertTrue(is_proper('Yakuhako', 'sv'))
        # dog (Swedish) == dead (English) so it exists but is not a noun
        self.assertFalse(is_noun('dog', 'sv'))
        self.assertFalse(is_proper('dog', 'sv'))
        self.assertFalse(is_proper('Dog', 'sv'))

    def test_is_proper_en(self):
        self.assertTrue(is_noun('dog', 'en'))
        self.assertFalse(is_proper('dog', 'en'))
        self.assertFalse(is_proper('Dog', 'en'))  # Even though it's capitalized

    def test_prepare1(self):
        self.assertEqual(['The dog barked .'], prepare_lines(['The dog barked.'], lang='en'))
        self.assertEqual(['The dog barked !'], prepare_lines(['The dog barked!'], lang='en'))
        self.assertEqual(['The dog barked ?'], prepare_lines(['The dog barked?'], lang='en'))
        self.assertEqual(['The dog , barked'], prepare_lines(['The dog, barked'], lang='en'))

    def test_prepare2(self):
        self.assertEqual(['the dog barked .'], prepare_lines(['The dog barked.'], lang='en', lc_first='lookup'))
        self.assertEqual(['the dog barked !'], prepare_lines(['The dog barked!'], lang='en', lc_first='lookup'))
        self.assertEqual(['the dog barked ?'], prepare_lines(['The dog barked?'], lang='en', lc_first='lookup'))
        self.assertEqual(['the dog , barked'], prepare_lines(['The dog, barked'], lang='en', lc_first='lookup'))

    def test_prepare3(self):
        self.assertEqual(['I am OK .'], prepare_lines(['I\'m OK.']))
        self.assertEqual(['He is fine !'], prepare_lines(['He\'s fine!']))
        self.assertEqual(['I bet it is good'], prepare_lines(['I bet it\'s good']))
        # Introduce a space at least when we can't easily detect
        self.assertEqual(['Jim \'s Restaurant'], prepare_lines(['Jim\'s Restaurant']))
        self.assertEqual(['Jim \'s fine .'], prepare_lines(['Jim\'s fine.']))

    def test_tokenize1(self):
        res_list = self.tokenize('The dog barked.')
        self.assertEqualSet(['The', 'dog', 'barked', '.'], res_list)

    def test_tokenize2(self):
        res_list = self.tokenize('The dog barked at the other dog.')
        self.assertEqualSet(['The', 'dog', 'barked', 'at', 'the', 'other', '.'], res_list)

    def test_tokenize3(self):
        res_list = self.tokenize('The dog, and the puppy, barked at the other dog.')
        self.assertEqualSet(['The', 'dog', ',', 'and', 'puppy', 'barked', 'at', 'the', 'other', '.'], res_list)

    def test_pos_tag_en(self):
        res_list = pos_tag(['The', 'dog', ',', 'and', 'the', 'puppy', 'barked', 'at', 'the', 'other', 'dog'])
        self.assertEqual(
            ['The.DT', 'dog.NN', ',', 'and.CC', 'the.DT', 'puppy.NN', 'barked.VB', 'at.IN', 'the.DT', 'other.JJ',
             'dog.NN'], res_list)

    def test_pos_tag_sv(self):
        res_list = pos_tag(['Hunden', 'och', 'valpen', 'barkade', 'på', 'den', 'andra', 'hunden', '.'], lang='sv')
        self.assertEqual(
            ['Hunden.NN', 'och.KN', 'valpen.NN', 'barkade.VB', 'på.PP', 'den.DT', 'andra.JJ', 'hunden.NN', '.'],
            res_list)

    def test_hyphenate_en(self):
        self.assertEqual(['dig'], hyphenate('dig', lang='en'))
        self.assertEqual(['dig', 'i', 'tal'], hyphenate('digital', lang='en'))
        self.assertEqual(['hos', 'pi', 'tal'], hyphenate('hospital', lang='en'))

    def test_hyphenate_sv(self):
        self.assertEqual(['hus'], hyphenate('hus', lang='sv'))
        self.assertEqual(['sjuk', 'hus'], hyphenate('sjukhus', lang='sv'))
        self.assertEqual(['sjuk', 'hus', 'et'], hyphenate('sjukhuset', lang='sv'))

    def test_hyphenate_lines(self):
        self.assertEqual([['hat']], hyphenate_lines(['hat'], lang='en'))
        self.assertEqual([['The', ' ', 'cat', ' ', 'in', ' ', 'the', ' ', 'hat']],
                         hyphenate_lines(['The cat in the hat'], lang='en'))
        self.assertEqual([['The', ' ', 'fe', 'line', ' ', 'in', ' ', 'the', ' ', 'fe', 'do', 'ra']],
                         hyphenate_lines(['The feline in the fedora'], lang='en'))

    def test_word2phrase_lines(self):
        self.assertEqual([['Hat']], word2phrase_lines(['Hat'], lang='en'))
        self.assertEqual([['The', 'cat', 'in', 'the', 'hat', '.']], word2phrase_lines(['The cat in the hat .'], lang='en'))

    def assertEqualSet(self, list1, list2):
        self.assertEqual(set(list1), set(list2))

    def tokenize(self, tr, lang='en', lc_first=None):
        prepared = prepare_lines([tr], lang, lc_first)
        res = create_tokenizer_simple(prepared)
        res_list = list(res.word_index.keys())
        if (lc_first == 'lookup'):
            # We should try and lowercase the first word if it's not proper
            for i, line in enumerate(res_list):
                fw = line[0]
                if not is_proper(fw, lang):
                    line[0] = line[0].lower()

        return res_list


if __name__ == '__main__':
    unittest.main()
