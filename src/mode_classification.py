import acoustic_inversion as ai
import numpy as np
import itertools as it
from multiprocessing import Pool

def fixed_weight_gray_code(n: int, k: int):
    """
    Generates all binary strings of length n with fixed Hamming weight k.
    Consecutive codewords differ by swapping a single '0' and '1' (Hamming distance 2).
    """
    if k < 0 or k > n:
        return []
    if k == 0:
        return [[False] * n]
    if k == n:
        return [[True] * n]

    # Recursive step:
    # 1. Prepend '0' to the Gray sequence for (n-1, k)
    # 2. Prepend '1' to the REVERSED Gray sequence for (n-1, k-1)
    part0 = [[False] + code for code in reversed(fixed_weight_gray_code(n - 1, k))]
    part1 = [[True] + code for code in fixed_weight_gray_code(n - 1, k - 1)]

    return part0 + part1



class ModeClassifier:
    def __init__(self, data_arrival_times, replica_arrival_times):
        self.data_arrival_times = data_arrival_times
        self.replica_arrival_times = replica_arrival_times

        self.nmodes = self.data_arrival_times.shape[0]
        self.nreplicas = self.replica_arrival_times.shape[-2]

        self.lk = ai.AcousticInversionLikelihood(self.data_arrival_times)

    def lm(self, m):
        return self.lk.likelihood_direct(self.replica_arrival_times[...,m,:])

    def classify_modes2(self):
        mode_combinations = np.fliplr(fixed_weight_gray_code(self.nreplicas, self.nmodes))
        #mode_combinations = list(it.combinations(range(self.nreplicas), self.nmodes))

        with Pool(4) as p:
            likelihoods = np.array(p.map(self.lm, mode_combinations))

        return likelihoods

    def classify_modes(self):
        mode_combinations = list(it.combinations(range(self.nreplicas), self.nmodes))

        #ind = np.ones(arrival_times_replica.ndim - 2)



        with Pool(4) as p:
            likelihoods = np.array(p.map(self.lm, mode_combinations))

        # likelihoods = np.array([self.lk.likelihood(self.replica_arrival_times[...,m,:])
                                 # for m in mode_combinations])

        ind_estimated = np.unravel_index(np.argmax(likelihoods), likelihoods.shape)[0]

        return mode_combinations[ind_estimated]


        

        
        
