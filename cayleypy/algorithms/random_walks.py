from typing import Union
import time
import numpy as np
import torch

from ..torch_utils import TorchHashSet


class RandomWalksGenerator:
    """Random walks generator for CayleyGraph."""
    
    def __init__(self, graph):
        self.graph = graph
    
    def _random_walks_classic_hash(
        self, width: int, length: int, start_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Allocate memory.
        x_shape = (width * length, self.graph.encoded_state_size)
        x = torch.zeros(x_shape, device=self.graph.device, dtype=torch.int64)
        y = torch.zeros(width * length, device=self.graph.device, dtype=torch.int32)

        # First state in each walk is the start state.
        x[:width, :] = start_state.reshape((-1,))
        y[:width] = 0

        # Main loop.
        for i_step in range(1, length):
            y[i_step * width : (i_step + 1) * width] = i_step
            gen_idx = torch.randint(0, self.graph.definition.n_generators, (width,), device=self.graph.device)
            src = x[(i_step - 1) * width : i_step * width, :]
            dst = x[i_step * width : (i_step + 1) * width, :]
            for j in range(self.graph.definition.n_generators):
                # Go to next state for walks where we chose to use j-th generator on this step.
                mask = gen_idx == j
                prev_states = src[mask, :]
                next_states = torch.zeros_like(prev_states)
                self.graph.state_ops._apply_generator_batched(j, prev_states, next_states)
                dst[mask, :] = next_states

        return self.graph.state_ops.decode_states(x), y

    def _random_walks_bfs_hash(
        self, width: int, length: int, start_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: change to int64
        start_state = start_state.to(torch.int64)
        x_hashes = TorchHashSet()
        x_hashes.add_sorted_hashes(self.graph.hasher.make_hashes(start_state))
        x = [start_state]
        y = [torch.full((1,), 0, device=self.graph.device, dtype=torch.int64)]

        for i_step in range(1, length):
            next_states = self.graph.get_neighbors(x[-1])
            next_states, next_states_hashes = self.graph.graph_utils._get_unique_states(next_states)
            mask = x_hashes.get_mask_to_remove_seen_hashes(next_states_hashes)
            next_states, next_states_hashes = next_states[mask], next_states_hashes[mask]
            layer_size = len(next_states)
            if layer_size == 0:
                break
            if layer_size > width:
                random_indices = torch.randperm(layer_size)[:width]
                layer_size = width
                next_states = next_states[random_indices]
                next_states_hashes = next_states_hashes[random_indices]
            x.append(next_states)
            x_hashes.add_sorted_hashes(next_states_hashes)
            y.append(torch.full((layer_size,), i_step, device=self.graph.device, dtype=torch.int64))
        return self.graph.state_ops.decode_states(torch.vstack(x)), torch.hstack(y)

    def _random_walks_classic_nbt(self, state_destination, generators , n_random_walk_length,  n_random_walks_to_generate,    state_rw_start = '01234...',
                    n_random_walks_steps_back_to_ban = 0, random_walks_type = 'non-backtracking-beam',
                    device='Auto', dtype = 'Auto' , vec_hasher = 'Auto', verbose = 0):
        t0 = time.time()
        ##########################################################################################
        # Processing/Reformating input params
        ##########################################################################################

        # device 
        if device == 'Auto':
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")      
        # Analyse input format of "generators"
        # can be list_generators, or tensor/np.array with rows - generators
        if isinstance(generators, list):
            list_generators = generators
        elif isinstance(generators, tuple):
            list_generators = list(generators)        
        elif isinstance(generators, torch.Tensor ):
            list_generators = [ list(generators[i,:]) for i in range(generators.shape[0] ) ]       
        elif isinstance(generators, np.ndarray ):
            list_generators = [list(generators[i,:]) for i in range(generators.shape[0] ) ]
        else:
            print('Unsupported format for "generators"', type(generators), generators)
            raise ValueError('Unsupported format for "generators" ' + str(type(generators)) )
        state_size = len(list_generators[0])
        n_generators = len( list_generators )

        # dtype 
        if (state_rw_start == '01234...') or (state_rw_start == 'Auto' ):
            n_unique_symbols_in_states = state_size
        else:
            tmp = set( [int(i) for i in state_rw_start ]  ) # Number of unique elements in any iterator
            n_unique_symbols_in_states = len(tmp)
        if dtype == 'Auto':
            if n_unique_symbols_in_states <= 256:
                dtype = torch.uint8
            else:
                dtype = torch.uint16

        # Destination state
        if (state_rw_start == '01234...') or (state_rw_start == 'Auto' ):
            state_rw_start = torch.arange( state_size, device=device, dtype = dtype).reshape(-1,state_size)
        elif isinstance(state_destination, torch.Tensor ):
            state_rw_start =  state_destination.to(device).to(dtype).reshape(-1,state_size)
        else:
            state_rw_start = torch.tensor( state_destination, device=device, dtype = dtype).reshape(-1,state_size)

        # Reformat generators in torch 2d array
        dtype_generators = torch.int64  
        tensor_generators = torch.tensor(np.array(list_generators), device=device, dtype=dtype_generators)
        #print('tensor_generators.shape:', tensor_generators.shape)

        # Preprare vector which will used for hashing - to avoid revisiting same states 
        max_int =  int( (2**62) )
        dtype_for_hash = torch.int64
        if vec_hasher == 'Auto':
            vec_hasher = torch.randint(-max_int, max_int+1, size=(state_size,), device=device, dtype=dtype_for_hash) 
            

        ##########################################################################################
        # Initializations
        ##########################################################################################

        array_current_states = state_rw_start.view(1, state_size  ).expand(n_random_walks_to_generate, state_size ).clone()
        
        X = torch.zeros( (n_random_walks_to_generate)*n_random_walk_length , state_size, device=device, dtype = dtype )
        y = torch.zeros( (n_random_walks_to_generate)*n_random_walk_length , device=device, dtype = torch.uint32 )
        X[:n_random_walks_to_generate,:] = array_current_states
        y[:n_random_walks_to_generate] = 0  

        # Hash initial states. 
        if n_random_walks_steps_back_to_ban > 0:
            hash_initial_state = torch.sum( state_rw_start.view(-1, state_size  ) * vec_hasher, dim=1) # Compute hashes 
            vec_hashes_current = hash_initial_state.expand( n_random_walks_to_generate * n_generators  , n_random_walks_steps_back_to_ban ).clone()
            i_cyclic_index_for_hash_storage = 0
            

        if verbose >= 100:
            print('X.shape:',X.shape, 'y.shape:',y.shape)
            print(array_current_states.shape)
            print( array_current_states[:3,:] )    

        i_step_corrected = 0
        for i_step in range(1,n_random_walk_length):
            t_moves = t_hash = t_isin =  0; t_full_step = time.time() # Time profiling
            t_unique_els = 0 # not used currently 

            # 1 Create new states: 
            # Apply all generators to all current states at once
            # array_new_states: 2d array (n_random_walks_to_generate * n_generators  ) x state_size
            t1 = time.time()
            array_new_states = self.graph.get_neighbors(array_current_states).flatten(end_dim=1) # Flatten converts 3d array to 2d
            t_moves += (time.time() - t1)
            
            # 2.1 Compute hashes
            # Compute hash. For non-backtracking - selection not visited states before. 
            t1 = time.time()
            vec_hashes_new = torch.sum(array_new_states * vec_hasher, dim=1) # Compute hashes 
            # That is equivalent to matmul, but matmul is not supported for ints in torch (on GPU <= 2018) - that is way round
            t_hash += (time.time() - t1)
            #  print('hashed', t_hash )

            if n_random_walks_steps_back_to_ban > 0:
                # 2.2 Select only states not seen before
                # Nonbacktracking - select states not visited before 
                t1 = time.time()
                mask_new = ~torch.isin(vec_hashes_new, vec_hashes_current.view(-1), assume_unique=False)
                t_isin += (time.time() - t1)
                mask_new_sum = mask_new.sum().item()
                if mask_new_sum >= n_random_walks_to_generate:
                    # Select only new states - not visited before
                    array_new_states = array_new_states[mask_new,:]
                    i_step_corrected += 1
                else:
                    # Exceptional case - should not happen for large group of interest
                    # The case: can not find enough new states - will take old also not to crash the code
                    if mask_new_sum > 0:
                        i_tmp0 = int( np.ceil( n_random_walks_to_generate/mask_new_sum ))
                        array_new_states = array_new_states[mask_new,:].repeat(i_tmp0, 1)[:n_random_walks_to_generate,:]
                        i_step_corrected += 1
                    else:
                        # do not move
                        array_new_states = array_current_states # 
                        i_step_corrected = i_step_corrected
                        
            # 3. Select only desired number of states
            # Select only n_random_walks_to_generate (with preliminary shuffling)
            # Update current states with them 
            perm = torch.randperm(array_new_states.size(0), device = device)
            array_current_states = array_new_states[perm][:n_random_walks_to_generate]

            # 4. Store results in final output
            y[ (i_step)*n_random_walks_to_generate : (i_step+1)*n_random_walks_to_generate  ] = i_step_corrected
            X[ (i_step)*n_random_walks_to_generate : (i_step+1)*n_random_walks_to_generate , : ] = array_current_states

            if n_random_walks_steps_back_to_ban>0:
                i_cyclic_index_for_hash_storage = (i_cyclic_index_for_hash_storage + 1 ) % n_random_walks_steps_back_to_ban
                vec_hashes_current[:, i_cyclic_index_for_hash_storage ] = vec_hashes_new   

            if verbose >= 10:
                t_full_step = time.time()-t_full_step
                print(i_step,'i_step', 'array_current_states.shape:',array_current_states.shape, 'Time %.3f'%(time.time()-t0),
                    't_moves  %.3f, t_hash  %.3f, t_isin %.3f, t_unique_els  %.3f, t_full_step %.3f'%(t_moves , 
                    t_hash , t_isin , t_unique_els, t_full_step) )
                
        return X,y


    def generate(
        self,
        *,
        width=5,
        length=10,
        mode="classic",
        start_state: Union[None, torch.Tensor, np.ndarray, list] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates random walks on this graph.

        The following modes of random walk generation are supported:

          * "classic" - random walk is a path in this graph starting from `start_state`, where on each step the next
            edge is chosen randomly with equal probability. We generate `width` such random walks independently.
            The output will have exactly ``width*length`` states.
            i-th random walk can be extracted as: ``[x[i+j*width] for j in range(length)]``.
            ``y[i]`` is equal to number of random steps it took to get to state ``x[i]``.
            Note that in this mode a lot of states will have overestimated distance (meaning ``y[i]`` may be larger than
            the length of the shortest path from ``x[i]`` to `start_state`).
            The same state may appear multiple times with different distance in ``y``.
          * "bfs" - we perform Breadth First Search starting from ``start_state`` with one modification: if size of
            next layer is larger than ``width``, only ``width`` states (chosen randomly) will be kept.
            We also remove states from current layer if they appeared on some previous layer (so this also can be
            called "non-backtracking random walk").
            All states in the output are unique. ``y`` still can be overestimated, but it will be closer to the true
            distance than in "classic" mode. Size of output is ``<= width*length``.
            If ``width`` and ``length`` are large enough (``width`` at least as large as largest BFS layer, and
            ``length >= diameter``), this will return all states and true distances to the start state.

        :param width: Number of random walks to generate.
        :param length: Length of each random walk.
        :param start_state: State from which to start random walk. Defaults to the central state.
        :param mode: Type of random walk (see above). Defaults to "classic".
        :return: Pair of tensors ``x, y``. ``x`` contains states. ``y[i]`` is the estimated distance from start state
          to state ``x[i]``.
        """
        start_state = self.graph.state_ops.encode_states(start_state or self.graph.central_state)
        if mode == "classic_hash":
            return self._random_walks_classic_hash(width, length, start_state)
        elif mode == "bfs_hash":
            return self._random_walks_bfs_hash(width, length, start_state)
        elif mode == "classic_nbt":
            return self._random_walks_classic_nbt(start_state, self.graph.generators, length, width, '01234...', 8, "non-backtracking-beam", "Auto", "Auto", "Auto", 0)
        else:
            raise ValueError("Unknown mode:", mode)