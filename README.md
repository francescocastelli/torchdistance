# torchdistance

PyTorch implementation of the editdistance, supported both on cpu and nvidia gpu. \
Implemented as a PyTorch extension, written in C++ and Cuda.

## Build

```
git clone https://github.com/francescocastelli/torchdistance
cd torchdistance/
python3 setup.py install
```

If a nvidia gpu is present, this will automatically build both the cpu and gpu version of the library.
It requires PyTorch to be already installed.

## Usage

It's supposed to work with already tokenized char or words.
The editdistance method recive as input two tensors, which are either 1d or 2d tensors.
The two tensors contain the reference words and the predicted words, must have the same number of words.
It return a tensor containing the element-wise editdistance among the ref tensor and the predicted tensor.

```
import torch
from torchdistance import editdistance

# check tests/utils for an example of a tokenizer
tokenizer = Tokenizer()
padToken = 0
device = 'cuda:0'
ref = ['test', 'hello', 'ciao']
hyp = ['tcest', 'heo', 'ciaoo']

ref = [tokenizer.tokenize(w) for w in ref]
hyp = [tokenizer.tokenize(w) for w in hyp]

# padding
x = pad_sequence(ref, batch_first=True, padding_value=padToken)
y = pad_sequence(hyp, batch_first=True, padding_value=padToken)

x, y = x.to(device), y.to(device)
pred = editdistance(x, y, padToken).to('cpu')
# pred is a torch.Tensor([1, 2, 1]) containing the editdistance for each word in the batch
```
