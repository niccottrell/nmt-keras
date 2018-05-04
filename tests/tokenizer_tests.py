import unittest
from helpers import *


class TokenizerTests(unittest.TestCase):

    def test_prepare1(self):
       self.assertEqual(['The dog barked .'], prepare_lines(['The dog barked.']))
       self.assertEqual(['The dog barked !'], prepare_lines(['The dog barked!']))
       self.assertEqual(['The dog barked ?'], prepare_lines(['The dog barked?']))
       self.assertEqual(['The dog , barked'], prepare_lines(['The dog, barked']))

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

    def tokenize(self, tr):
        prepared  = prepare_lines([tr])
        res = create_tokenizer_simple(prepared)
        res_list = list(res.word_index.keys())
        return res_list

if __name__ == '__main__':
    unittest.main()
