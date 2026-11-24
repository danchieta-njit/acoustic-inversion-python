import numpy as np
from multiprocessing import Pool
from joblib import Parallel, delayed


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
    part0 = [code + [False]  for code in reversed(fixed_weight_gray_code(n - 1, k))]
    part1 = [code + [True] for code in fixed_weight_gray_code(n - 1, k - 1)]

    return part0 + part1



class AIEstimator:
    def __init__(self, data_arrival_times, replica_arrival_times):
        self.data_arrival_times = data_arrival_times
        self.replica_arrival_times = replica_arrival_times

        self.nmodes = self.data_arrival_times.shape[0]
        self.nreplicas = self.replica_arrival_times.shape[-2]

        self.lk = AcousticInversionLikelihood(self.data_arrival_times)
        self.mode_combinations = fixed_weight_gray_code(self.nreplicas, self.nmodes)

    def lm(self, m):
        return self.lk.likelihood_direct(self.replica_arrival_times[...,m,:])

    def likelihood(self, ind, fun=None):
        if fun is None:
            fun=self.lk.likelihood_direct()
        return fun(self.replica_arrival_times[...,self.mode_combinations[ind[0],:]])[ind[1:]]

    def estimate(self):
        #mode_combinations = list(it.combinations(range(self.nreplicas), self.nmodes))

        with Pool(4) as p:
            self.likelihoods = np.array(p.map(self.lm, self.mode_combinations))

        ind_modes, ind_soundspeed = np.unravel_index(
            np.argmax(self.likelihoods), self.likelihoods.shape)

        return self.mode_combinations[ind_modes], ind_soundspeed

def peaks_from_file(filename):
    """
    Loads the file containing peaks data (from which modal arrival times will
    be inferred) and stores it into a dictionary.

    """
    with np.load(filename, allow_pickle=True ) as file1:
        data = dict(mode_nos = file1['mode_nos'].squeeze(),
                    peaks_coordinates = file1['peaks_coordinates'])
    data['mode_nos'] = data['mode_nos'] - 1

    inds = np.argsort(data['mode_nos'])
    data['mode_nos'] = data['mode_nos'][inds]
    data['peaks_coordinates'] = data['peaks_coordinates'][inds]

    return data


def constrained_polyfit(x, y, deg, limit=np.inf, **kwargs):
    p = np.polyfit(x,y,deg, **kwargs)

    if np.abs(p[0]) <= limit:
        return p

    
    y = y - np.sign(p[0])*limit*x**deg

    p_low = np.polyfit(x,y,deg-1)

    return np.append(np.sign(p[0])*limit, p_low)


def peaks_to_arrivaltimes(peaks, deg = 2, limit=5e-6, freqs=None, method="all"):

    if freqs is None:
        freqs = np.unique(np.concatenate([p[:,0] for p in peaks]))

    coefs = np.array(
        [constrained_polyfit(d[:,0], d[:,1], deg, limit) for d in peaks])
    nmodes = len(coefs)
    nfreqs = len(freqs)

    if method == "all":
        out = np.array([np.polyval(c, freqs) for c in coefs])


    if method == "valid":
        out = np.empty((nmodes, nfreqs)) * np.nan
        for imode in range(nmodes):
            fnow = np.intersect1d(peaks[imode][:,0],freqs)
            out[imode, np.isin(freqs, fnow)] = np.polyval(coefs[imode], fnow)

    if method == "jagged":
        out = np.empty((nmodes, nfreqs)) * np.nan
        for imode in range(nmodes):
            (fmin, fmax) = (peaks[imode][:,0].min(), peaks[imode][:,0].max())
            mask = (freqs >= fmin) & (freqs <= fmax)
            out[imode, mask] = np.polyval(coefs[imode], freqs[mask])
    
    return out, freqs

def arrival_times_from_peaks(peaks, method="regression", f=None, **kwargs):

    coefs = np.array(
        [constrained_polyfit(d[:,0], d[:,1], deg, limit) for d in peaks])

    if f == None:
        freq = [np.unique(p[:,0]) for p in peaks]
        return freq, [np.polyval(c, f) for c,f in zip(coefs, freq)]
    else:
        return np.array([np.polyval(c, f) for c in coefs]) 

# class GetArrivalTimes:
    # def __init__(self, peaks):
        # self.coefs = np.array([np.polyfit(d[:,0], d[:,1], 2) for d in peaks])
    # def arrival_times(self, f):
        # return np.array([np.polyval(c, f) for c in self.coefs]) 
        
def nan_norm(x, axis):
    return np.sqrt(np.nansum(x**2, axis=axis))

def calculate_deltas(x):
    return x[..., np.newaxis, :] - x[..., :, np.newaxis]

def calculate_residual(data_arrival_times, replica_arrival_times):
    # data_arrival_times: 2d ndarray nmodes x nfreqs
    # replica_arrival_times: Nd ndarray bulk_parameters x nmodes x nfreqs
    replica_delta = calculate_deltas(replica_arrival_times)
    data_delta = calculate_deltas(data_arrival_times)

    return replica_delta - data_delta

def likelihood_direct(data, replica, sigma = 1):
    residual = data - replica

    return -np.nansum(residual**2,(-2,-1))/2/sigma**2

def likelihood_combined(data, replica, sigma = 1):
    residual_inter = calculate_residual(data.mT, replica.mT)
    residual_intra = calculate_residual(data, replica)
    
    return -(np.nansum(np.tril(residual_intra)**2,(-3, -2, -1)) +
        np.nansum(np.tril(residual_inter)**2,(-3, -2, -1)))/2/sigma**2

# def evaluate_likelihoods(data, replicas, mode_combinations, likelihood):
    # return np.array([
        # likelihood(data, replicas[..., modes, :])
        # for modes in mode_combinations
    # ])
def evaluate_likelihoods(data, replicas, mode_combinations, likelihood, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(likelihood)(data, replicas[..., modes, :])
        for modes in mode_combinations
    )

    return np.array(results)

def maximize_likelihood(likelihoods, mode_combinations):

    best_index = np.unravel_index(
        np.nanargmax(likelihoods),
        likelihoods.shape
    )

    return (
        mode_combinations[best_index[0]],
        best_index[1:]
    )

class AcousticInversionLikelihood:
    def __init__(self, data_arrival_times):
        self.data_arrival_times = data_arrival_times

    def likelihood_direct(self, replica_arrival_times, sigma = 1):
        residual = self.data_arrival_times - replica_arrival_times

        return -np.nansum(residual**2,(-2,-1))/2/sigma**2

    def likelihood_intermodal(self, replica_arrival_times):
        residual = calculate_residual(self.data_arrival_times.mT,
                                      replica_arrival_times.mT)
        
        return np.exp(-nan_norm(residual, -3)**2)

    def likelihood_intramodal(self, replica_arrival_times):
        residual = calculate_residual(self.data_arrival_times,
                                      replica_arrival_times)
        
        return np.exp(-nan_norm(residual, -3)**2)

    def likelihood(self, replica_arrival_times, sigma = 1):
        residual_inter = calculate_residual(self.data_arrival_times.mT,
                                            replica_arrival_times.mT)
        residual_intra = calculate_residual(self.data_arrival_times,
                                            replica_arrival_times)
        
        return -(np.nansum(np.tril(residual_intra)**2,(-3, -2, -1)) +
            np.nansum(np.tril(residual_inter)**2,(-3, -2, -1)))/2/sigma**2
