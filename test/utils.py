import string
import torch
from collections import defaultdict

padToken = 0

class Tokenizer():
    r"""
    Tokenizer contains a dictionary of tokens, where each token is a character. In particular we consider:
        * digits
        * all letters of the english alphabet 
        * punction: dot, comma, colon, space
        * pad: special token used to pad sequences (set to 0)
        * unk: all the other characters
    """
    def __init__(self):
        # tokens = lower-case lettes, numbers and some punctuation
        tokens = [*string.ascii_lowercase, *range(0, 10), ' ', ',', '.', '\'', ':']

        self.token_dict = defaultdict(lambda: len(tokens), {c: k for k, c in enumerate(tokens, 1)})
        self.char_dict = {v: k for k, v in self.token_dict.items()}

        # char not in dict gets len(tokens) -> unk
        self.char_dict[len(tokens)] = 'unk'
        # set constant special tokens
        self.char_dict[padToken] = 'pad'

        self.vocabulary_len = len(self.char_dict)


    def tokenize(self, seq: str, device):
        r"""
           Used to tokenized the input sequence.

           Args:
                seq: list of char that compose the sequence

           Returns: 
                list of tokens corresponding to the input seq
        """
        token_seq = [self.token_dict[c.lower()] for c in seq]
        return torch.tensor(token_seq, dtype=torch.long, device=device)

    def decode_tokens(self, token_seq: list):
        r"""
           Reconstruct the seq starting from a list of tokens

           Args:
                token_seq: list of int that compose the token sequence

           Returns: 
                list of chars corresponding to the token seq
        """
        char_list = [self.char_dict[t] for t in token_seq] 
        string = "".join(char_list)

        return char_list, string
