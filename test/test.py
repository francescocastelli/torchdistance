import torch
import unittest
from editdistance import editdistance

class EditDistanceTest(unittest.TestCase):
    def test_cer(self):

        # ref: [TEST]
        # hyp: [TESST]

        x = torch.tensor([1, 2, 3, 1], dtype=torch.int)
        y = torch.tensor([1, 2, 3, 3, 1], dtype=torch.int)

        r = editdistance(x, y)
        self.assertEqual(r, 1)

if __name__ == "__main__":
    unittest.main()
