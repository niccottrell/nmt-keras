import unittest

from nltk.translate.bleu_score import sentence_bleu, modified_precision, SmoothingFunction


class EvaluateTests(unittest.TestCase):

    def test_sentence_bleu(self):
        sm = SmoothingFunction()
        reference1 = 'the cat is on the mat'.split()
        reference2 = 'there is a cat on the mat'.split()
        references = [reference1, reference2]
        hypothesis1 = 'the the the the the the the'.split()  # known to 'trick' BLEU
        hypothesis2 = 'the cat the the the the mat'.split()  # should still score much higher for BLEU-1 (but not BLEU-4)
        score1_mod = float(modified_precision(references, hypothesis1, n=1))
        score1_bleu1 = sentence_bleu(references, hypothesis1, (1, 0, 0, 0))
        score1_bleu1_method5 = sentence_bleu(references, hypothesis1, (1, 0, 0, 0), smoothing_function=sm.method5)
        score1_bleu4 = sentence_bleu(references, hypothesis1)
        print("Scores for hypothesis1 modified=%f, BLEU-1=%f (method5 %f), BLEU-4=%f" % (score1_mod, score1_bleu1, score1_bleu1_method5, score1_bleu4))
        score2_mod = float(modified_precision(references, hypothesis2, n=1))
        score2_bleu1 = sentence_bleu(references, hypothesis2, (1, 0, 0, 0))
        score2_bleu1_method5 = sentence_bleu(references, hypothesis2, (1, 0, 0, 0), smoothing_function=sm.method5)
        score2_bleu4 = sentence_bleu(references, hypothesis2)
        print("Scores for hypothesis2 modified=%f, BLEU-1=%f (method5 %f), BLEU-4=%f" % (score2_mod, score2_bleu1, score2_bleu1_method5, score2_bleu4))
        self.assertLess(score1_mod, 0.3)
        self.assertGreater(score1_bleu1, 0.0)
        self.assertLess(score1_bleu1, 0.3)
        self.assertGreater(score1_bleu1, 0.0)
        self.assertLess(score1_bleu4, 0.1)
        self.assertLess(score2_mod, 0.6)
        self.assertGreater(score2_bleu1, 0.0)
        self.assertLess(score2_bleu1, 0.6)
        self.assertGreater(score2_bleu4, 0.0)
        self.assertLess(score2_bleu4, 0.1)

    def test_sentence_perfect1(self):
        reference1 = 'You eat a cheese sandwich'.split()
        reference2 = 'You\'re eating a cheese sandwich'.split()
        reference3 = 'You are eating a cheese sandwich'.split()
        references = [reference1, reference2, reference3]
        hypothesis1 = 'You eat a cheese sandwich'.split()
        score1_mod = float(modified_precision(references, hypothesis1, n=1))
        score1_bleu1 = sentence_bleu([reference1], hypothesis1, (1, 0, 0, 0))
        score1_bleu4 = sentence_bleu([reference1], hypothesis1)
        score_bleu4 = sentence_bleu(references, hypothesis1)
        print("Scores for hypothesis modified=%f, BLEU-1=%f, BLEU-4=%f (all %f)" % (score1_mod, score1_bleu1, score1_bleu4, score_bleu4))
        self.assertGreaterEqual(score1_bleu4, 1.0)
        self.assertGreaterEqual(score_bleu4, 1.0)



if __name__ == '__main__':
    unittest.main()
