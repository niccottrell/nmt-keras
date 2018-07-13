import unittest
from helpers import *

class TokenizerTests(unittest.TestCase):

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

    def assertEqualSet(self, list1, list2):
        self.assertEqual(set(list1), set(list2))

    def tokenize(self, tr, lang='en', lc_first=None):
        prepared = prepare_lines([tr])
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
