import torch
import unittest
import json
from torch.nn.utils.rnn import pad_sequence
from parameterized import parameterized
from torchdistance import editdistance
from utils import Tokenizer, padToken

_devices = ['cpu', 'cuda:0'] if torch.cuda.is_available() else ['cpu']
_error_types = ['ins', 'del', 'sub']

def _test_name(testcase_func, param_num, param):
    args = [param.args[2], param_num, param.args[-1]]
    return (
            f"{testcase_func.__name__}_" 
            f"{parameterized.to_safe_name('_'.join(str(x) for x in args))}"
    )

def _decode_json(word : dict) -> list:
    return [word['ref'], word['hyp'], word['type'], word['error']]

def _filter_words(words_list: list, batched: bool) -> list:
    word_type = list if batched else str 
    return [w for w in words_list if isinstance(w['ref'], word_type)]

def _make_input(devices, batched):
    with open('test/words.json') as f:
        words_list = json.load(f)

    words_list = _filter_words(words_list, batched)
    inputs = [(*_decode_json(w), d) for w in words_list for d in devices]
    return inputs

class EditDistanceTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    @parameterized.expand(
        _make_input(_devices, False), 
        name_func=_test_name
    )
    def test_no_batch(self, ref, hyp, err_type, error, device):
        x = self.tokenizer.tokenize(ref, device)
        y = self.tokenizer.tokenize(hyp, device)

        r = editdistance(x, y, padToken)
        self.assertEqual(r.item(), error)

    @parameterized.expand(
        _make_input(_devices, True), 
        name_func=_test_name
    )
    def test_batch(self, ref, hyp, err_type, error, device):
        ref = [self.tokenizer.tokenize(w) for w in ref]
        hyp = [self.tokenizer.tokenize(w) for w in hyp]

        # padding
        x = pad_sequence(ref, batch_first=True, padding_value=padToken)
        y = pad_sequence(hyp, batch_first=True, padding_value=padToken)

        x, y = x.to(device), y.to(device)
        pred = editdistance(x, y, padToken).to('cpu')

        self.assertEqual(error, [p.item() for p in pred])

if __name__ == "__main__":
    unittest.main(verbosity=2)
