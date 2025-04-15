### base functions
# 1. Not depending on cayley_group or predictor as a class
# 2. useful in many cases
import gc

import torch
import time
import numpy  as np


def setup_of_random(seed=None, border=2**32, verbose=0):
    """
    Explicit set of random seed. Works for CPU/GPU

    :param seed           : int or None
    :param border         : int or float

    :return: None, just random seed is set
    """
    if seed is None:
        seed = torch.randint(-border,border+1,(1,)).item()
    if verbose > 0:
        print(f'\nRandom seed used for experiments: {seed}\n')
    torch.manual_seed(seed)
    return seed

################################################################################################################################################################################################################################################################################
def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

################################################################################################################################################################################################################################################################################
def get_neighbors_plain(states, moves):
    """
    Some torch magic to calculate all new states which can be obtained from states by moves
    """
    return torch.gather(states.unsqueeze(1).expand(states.size(0), moves.shape[0], states.size(1)),
                        2,
                        moves .unsqueeze(0).expand(states.size(0), moves.shape[0], states.size(1))).flatten(end_dim=1) # added flatten to the end, because we always add it

################################################################################################################################################################################################################################################################################
def get_neighbors2(states, moves, chunking_thres=2**18):
    """
    Some torch magic to calculate all new states which can be obtained from states by moves
    """
    s_sh = states.shape[0]
    if s_sh > chunking_thres:
        result = torch.zeros(s_sh*moves.shape[0], states.shape[1], dtype=states.dtype, device=states.device)
        for i in range(0,moves.shape[0]):
            result[i*s_sh:(i+1)*s_sh, :] = get_neighbors_plain(states, torch.narrow(moves, 0, i, 1))
        return result
    return get_neighbors_plain(states, moves)

            
