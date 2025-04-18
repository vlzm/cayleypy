"""Helper functions."""
from typing import Sequence


def inverse_permutation(p: Sequence[int]) -> list[int]:
    n = len(p)
    ans = [0] * n
    for i in range(n):
        ans[p[i]] = i
    return ans
