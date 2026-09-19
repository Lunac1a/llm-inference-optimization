import unittest
from report import score, paired_interval

class ScoringTests(unittest.TestCase):
    def test_official_partial_and_any_reference_rules(self):
        self.assertEqual(score('Answer: ALPHA',['alpha','beta'],'niah'),.5)
        self.assertEqual(score('Answer: ALPHA',['alpha','beta'],'qa'),1)
        self.assertEqual(score('',['alpha'],'qa'),0)
        self.assertEqual(score('\x00alpha\x01beta',['alpha','beta'],'niah'),1)

    def test_paired_interval_preserves_identical_scores(self):
        self.assertEqual(paired_interval([[0,0],[0,0]]),[0,0])
        self.assertEqual(paired_interval([[-.5,-.5],[-.5,-.5]]),[-50,-50])

if __name__=='__main__':unittest.main()
