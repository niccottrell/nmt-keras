import unittest

from helpers import *
import tokenizers

pos_tagged_en = ['The.DT', 'dog.NN', ',', 'and.CC', 'the.DT', 'puppy.NN', 'barked.VB', 'at.IN', 'the.DT',
                 'other.JJ', 'dog.NN']

simple_lines = tokenizers.SimpleLines()
pos_tag = tokenizers.PosTag()
replace_proper = tokenizers.ReplaceProper()
word2phrase = tokenizers.Word2Phrase()
letters = tokenizers.LetterByLetter()
hyphenate = tokenizers.Hyphenate()


class TokenizerTests(unittest.TestCase):

    def test_simple_lines(self):
        self.assertEqual([["Good", "muffins", "cost", "$", "3", ".", "88", "in", "New", "York", "."]],
                         simple_lines.tokenize(["Good muffins cost $3.88\nin New York."], 'en'))
        self.assertEqual([["Good", "muffins", "cost", "$", "3", ".", "88", "in", "New", "York", "!"]],
                         simple_lines.tokenize(["Good muffins cost $3.88 in New York !"], 'en'))

    def test_is_in_dict(self):
        self.assertTrue(is_in_dict('dog', 'en'))
        self.assertFalse(is_in_dict('doog', 'en'))
        self.assertTrue(is_in_dict('dog', 'sv'))
        self.assertTrue(is_in_dict('dög', 'sv'))
        # self.assertFalse(is_in_dict('dug', 'sv'))

    def test_hunpos_tagger_cache(self):
        self.assertTrue(pos_tag.tokenize(['the dog'], 'en'))
        self.assertTrue(pos_tag.tokenize(['a dog'], 'en'))

    def test_is_proper_sv(self):
        # Yakuhako is a made up string
        self.assertTrue(is_proper('Yakuhako', 'sv'))
        # dog (Swedish) == dead (English) so it exists but is not a noun
        self.assertFalse(tokenizers.is_noun('dog', 'sv'))
        self.assertFalse(is_proper('dog', 'sv'))
        self.assertFalse(is_proper('Dog', 'sv'))

    def test_is_proper_en(self):
        self.assertTrue(tokenizers.is_noun('dog', 'en'))
        self.assertTrue(is_in_dict('dog', 'en'))
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
        res_list = pos_tag.tag(['The', 'dog', ',', 'and', 'the', 'puppy', 'barked', 'at', 'the', 'other', 'dog'])
        self.assertEqual(pos_tagged_en, res_list)

    def test_pos_tag_sv(self):
        res_list = pos_tag.tag(['Hunden', 'och', 'valpen', 'barkade', 'på', 'den', 'andra', 'hunden', '.'], lang='sv')
        self.assertEqual(
            ['Hunden.NN', 'och.KN', 'valpen.NN', 'barkade.VB', 'på.PP', 'den.DT', 'andra.JJ', 'hunden.NN', '.'],
            res_list)

    def test_pos_tag_sve(self):
        res_list = pos_tag.tag(['Den', 'såg', 'billig', 'ut', '.'], lang='sve')
        self.assertEqual(
            ['Den.PN', 'såg.VB', 'billig.JJ', 'ut.AB', '.'],  # where AB=adverb
            res_list)

    def test_pos_tag_utf8(self):
        """
        Tests a non-latin1 character (the euro symbol) which causes problems since the POS taggers are trained on latin1 input
        """
        input = ['It', 'costs', '€', '2', '.', '35']
        target = ['It.PR', 'costs.VB', '€', '2', '.', '35']
        res_list = pos_tag.tag(input)
        self.assertEqual(target, res_list)

    def test_pos_tag_lines(self):
        res_list = pos_tag.tokenize(['The dog , and the puppy barked at the other dog'], 'en')
        self.assertEqual([pos_tagged_en], res_list)

    def test_hyphenate_en(self):
        self.assertEqual(['dig'], hyphenate.tokenize_one('dig', lang='en'))
        self.assertEqual(['dig', 'i', 'tal'], hyphenate.tokenize_one('digital', lang='en'))
        self.assertEqual(['hos', 'pi', 'tal'], hyphenate.tokenize_one('hospital', lang='en'))

    def test_hyphenate_sv(self):
        self.assertEqual(['hus'], hyphenate.tokenize_one('hus', lang='sv'))
        self.assertEqual(['sjuk', 'hus'], hyphenate.tokenize_one('sjukhus', lang='sv'))
        self.assertEqual(['sjuk', 'hus', 'et'], hyphenate.tokenize_one('sjukhuset', lang='sv'))

    def test_hyphenate_sv2(self):
        self.assertEqual(['gås'], hyphenate.tokenize_one('gås', lang='sv'))
        self.assertEqual(['smör', 'gås'], hyphenate.tokenize_one('smörgås', lang='sv'))
        self.assertEqual(['smör', 'gås', 'en'], hyphenate.tokenize_one('smörgåsen', lang='sv'))

    def test_hyphenate_lines(self):
        self.assertEqual([['hat']],
                         hyphenate.tokenize(['hat'], lang='en'))
        self.assertEqual([['The', ' ', 'cat', ' ', 'in', ' ', 'the', ' ', 'hat']],
                         hyphenate.tokenize(['The cat in the hat'], lang='en'))
        self.assertEqual([['The', ' ', 'fe', 'line', ' ', 'in', ' ', 'the', ' ', 'fe', 'do', 'ra']],
                         hyphenate.tokenize(['The feline in the fedora'], lang='en'))

    def test_replace_proper_en(self):
        self.assertEqual([['hat']],
                         replace_proper.tokenize(['hat'], lang='en'))
        self.assertEqual([['The', 'cat', 'in', 'the', 'hat']],
                         replace_proper.tokenize(['The cat in the hat'], lang='en'))
        self.assertEqual([['The', 'feline', 'in', 'the', 'fedora']],
                         replace_proper.tokenize(['The feline in the fedora'], lang='en'))
        self.assertEqual([['NP1', 'is', 'a', 'cat', ',', 'and', 'NP2', 'is', 'a', 'duck']],
                         replace_proper.tokenize(['Felix is a cat, and Daffy is a duck'], lang='en'))

    def test_replace_proper_sv(self):
        self.assertEqual([['katt']],
                         replace_proper.tokenize(['katt'], lang='sv'))
        self.assertEqual([['Han', 'kommer', 'från', 'NP1']],
                         replace_proper.tokenize(['Han kommer från Stockholm'], lang='sv'))
        self.assertEqual([['The', 'feline', 'in', 'the', 'fedora']],
                         replace_proper.tokenize(['The feline in the fedora'], lang='sv'))
        self.assertEqual([['NP1', 'är', 'en', 'katt', ',', 'och', 'NP2', 'är', 'en', 'anka']],
                         replace_proper.tokenize(['Felix är en katt, och Kalle Anka är en anka'], lang='sv'))

    def test_word2phrase_lines(self):
        self.assertEqual([['Hat']], word2phrase.tokenize(['Hat'], lang='en'))
        self.assertEqual([['The', 'cat', 'in', 'the', 'hat', '.']],
                         word2phrase.tokenize(['The cat in the hat .'], lang='en'))

    def test_letters1(self):
        self.assertEqual(['H', 'a', 't'], letters.tokenize_one('Hat', lang='en'))
        self.assertEqual([['H', 'a', 't']], letters.tokenize(['Hat'], lang='en'))
        self.assertEqual(
            [['T', 'h', 'e', ' ', 'c', 'a', 't', ' ', 'i', 'n', ' ', 't', 'h', 'e', ' ', 'h', 'a', 't', '.']],
            letters.tokenize(['The cat in the hat.'], lang='en'))

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

    def test_mark_ends(self):
        self.assertEqual([['\t', 'the', 'fat', 'cat', '.', '\n']], mark_ends([['the', 'fat', 'cat', '.']]))


if __name__ == '__main__':
    unittest.main()
