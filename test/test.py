import torch
import unittest
from torch.nn.utils.rnn import pad_sequence
from parameterized import parameterized
from editdistance import editdistance
from utils import Tokenizer, padToken

class EditDistanceTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    @parameterized.expand([
        ['cpu'],
        ['cuda:0']
    ])
    def test_no_batch_ins(self, device):
        # ref: [TEST]
        # hyp: [TESST]

        x = self.tokenizer.tokenize('test', device)
        y = self.tokenizer.tokenize('tesst', device)

        r = editdistance(x, y, padToken)
        self.assertEqual(r.item(), 1)

    @parameterized.expand([
        ['cpu'],
        ['cuda:0']
    ])
    def test_no_batch_del(self, device):
        # ref: [TEST]
        # hyp: [TST]

        x = self.tokenizer.tokenize('test', device)
        y = self.tokenizer.tokenize('tst', device)

        r = editdistance(x, y, padToken)
        self.assertEqual(r.item(), 1)

    @parameterized.expand([
        ['cpu'],
        ['cuda:0']
    ])
    def test_no_batch_sub(self, device):
        # ref: [TEST]
        # hyp: [TAST]

        x = self.tokenizer.tokenize('test', device)
        y = self.tokenizer.tokenize('tast', device)

        r = editdistance(x, y, padToken)
        self.assertEqual(r.item(), 1)

    @parameterized.expand([
        ['cpu'],
        ['cuda:0']
    ])
    def test_batch_ins(self, device):
        # ref: [[TEST], [HELLO]]
        # hyp: [[TESTT], [HELLOO]]

        ref = [self.tokenizer.tokenize('test', device), self.tokenizer.tokenize('hello', device)]
        hyp = [self.tokenizer.tokenize('testt', device), self.tokenizer.tokenize('helloo', device)] 
        res = [1, 1]

        # padding
        x = pad_sequence(ref, batch_first=True, padding_value=padToken)
        y = pad_sequence(hyp, batch_first=True, padding_value=padToken)

        pred = editdistance(x, y, padToken).to('cpu')

        for r, p in zip(res, pred): 
            self.assertEqual(r, p.item())

    @parameterized.expand([
        ['cpu'],
        ['cuda:0']
    ])
    def test_batch_del(self, device):
        # ref: [[TEST], [HELLO]]
        # hyp: [[TET], [HELO]]

        ref = [self.tokenizer.tokenize('test', device), self.tokenizer.tokenize('hello', device)]
        hyp = [self.tokenizer.tokenize('tet', device), self.tokenizer.tokenize('helo', device)] 
        res = [1, 1]

        # padding
        x = pad_sequence(ref, batch_first=True, padding_value=padToken)
        y = pad_sequence(hyp, batch_first=True, padding_value=padToken)

        pred = editdistance(x, y, padToken).to('cpu')

        for r, p in zip(res, pred): 
            self.assertEqual(r, p.item())


if __name__ == "__main__":
    unittest.main()
