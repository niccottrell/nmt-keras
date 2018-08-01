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

    def test_clean_line_euros(self):
        self.assertEqual('show me the $ !',
                         clean_line('Show me the €!', 'en'))

    def test_clean_line_sv(self):
        self.assertEqual('hjälp !',
                         clean_line('Hjälp!', 'sv'))


if __name__ == '__main__':
    unittest.main()
