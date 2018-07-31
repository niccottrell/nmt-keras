import unittest
from clean import clean_line

class CleanTests(unittest.TestCase):

    # first word lowercased since it's not a proper noun
    def test_clean_line1(self):
        self.assertEqual('good muffins cost PRICE .',
                         clean_line('Good muffins cost $3.88.', 'en'))
        self.assertEqual('goda muffinsar kostar PRICE .',
                         clean_line('Goda muffinsar kostar $3,88.', 'sv'))

    def test_clean_line2(self):
        self.assertEqual('"Good muffins cost PRICE ."',
                         clean_line('”Good muffins cost $3.88.”', 'en'))
        self.assertEqual('"Goda muffinsar kostar PRICE ."',
                         clean_line('”Goda muffinsar kostar $3,88.”', 'sv'))


if __name__ == '__main__':
    unittest.main()
