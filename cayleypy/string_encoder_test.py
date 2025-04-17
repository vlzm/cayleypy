import math

import numpy as np
import pytest
import torch

from .string_encoder import StringEncoder


def _apply_permutation(x, p):
    return [x[p[i]] for i in range(len(p))]


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30), (10, 100)])
def test_encode_decode(code_width, n):
    num_states = 5
    s = torch.randint(0, 2 ** code_width, (num_states, n))
    enc = StringEncoder(code_width=code_width, n=n)
    s_encoded = enc.encode(s)
    assert s_encoded.shape == (num_states, int(math.ceil(code_width * n / 63)))
    assert torch.equal(s, enc.decode(s_encoded))


@pytest.mark.parametrize("code_width,n", [(1, 2), (1, 5), (2, 30), (10, 100)])
def test_permutation(code_width: int, n: int):
    num_states = 5
    s = torch.randint(0, 2 ** code_width, (num_states, n), dtype=torch.int64)
    perm = np.random.permutation(n)
    expected = torch.tensor([_apply_permutation(row, perm) for row in s.numpy()], dtype=torch.int64)
    enc = StringEncoder(code_width=code_width, n=n)
    s_encoded = enc.encode(s)
    result = torch.zeros_like(s_encoded)
    perm_func = enc.implement_permutation(perm)
    perm_func(s_encoded, result)
    ans = enc.decode(result)
    assert torch.equal(ans, expected)
