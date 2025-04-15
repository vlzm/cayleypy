# TODO: adds Rubik's cubes etc.; support specifying initial state.

def prepare_generators(name, n=0) -> list[list[int]]:
    """Returns pre-defined set of generating permutations.

    Args:
        name: name of pre-defined generators set.
        n: length of permutations (if applicable).

    Supported generator sets:
        "lrx" - shift left, shift right, swap first two elements (n>=2).
        "top_spin" - shift left, shift right, reverse first four elements (n>=4).
    """
    if name == "lrx":
        assert n >= 2
        return [list(range(1, n)) + [0], [n - 1] + list(range(0, n - 1)), [1, 0] + list(range(2, n))]
    elif name == "top_spin":
        assert n >= 4
        return [list(range(1, n)) + [0], [n - 1] + list(range(0, n - 1)), [3, 2, 1, 0] + list(range(4, n))]
    else:
        raise ValueError(f"Unknown generator set: {name}")
