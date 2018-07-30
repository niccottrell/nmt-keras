import unittest
from clean import clean_line

class CleanTests(unittest.TestCase):

    def test_clean_line(self):
        self.assertEqual('"Good muffins cost PRICE ."',
                         clean_line('”Good muffins cost $3.88.”', 'en'))
        self.assertEqual('"Goda muffinsar kostar PRICE ."',
                         clean_line('”Goda muffinsar kostar $3,88.”', 'sv'))


if __name__ == '__main__':
    unittest.main()
