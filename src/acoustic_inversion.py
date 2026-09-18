import numpy as np

def peaks_from_file(filename):
    """
    Loads the file containing peaks data (from which modal arrival times will
    be inferred) and stores it into a dictionary.

    """
    with np.load(filename, allow_pickle=True ) as file1:
        data = dict(mode_nos = file1['mode_nos'].squeeze(),
                    peaks_coordinates = file1['peaks_coordinates'])
    data['mode_nos'] = data['mode_nos'] - 1

    return data

def peaks_to_arrivaltimes(peaks, deg = 2, freqs=None, method="valid"):

    if freqs is None:
        freqs = np.unique(np.concatenate([p[:,0] for p in peaks]))

    coefs = np.array([np.polyfit(d[:,0], d[:,1], deg) for d in peaks])
    nmodes = len(coefs)
    nfreqs = len(freqs)

    if method == "all":
        out = np.array([np.polyval(c, freqs) for c in coefs])


    if method == "valid":
        out = np.empty((nmodes, nfreqs)) * np.nan
        for imode in range(nmodes):
            fnow = np.intersect1d(peaks[imode][:,0],freqs)
            out[imode, np.isin(freqs, fnow)] = np.polyval(coefs[imode], fnow)
    
    #out = np.array([np.polyval(c, freqs) for c in coefs])


    return out, freqs

def arrival_times_from_peaks(peaks, method="regression", f=None, **kwargs):

    coefs = np.array([np.polyfit(d[:,0], d[:,1], deg) for d in peaks])

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
    #if x.ndim == 2:
    #    x = x[np.newaxis,...]
        
    delta = np.zeros(x.shape[:-1] + (x.shape[-1], x.shape[-1]))

    for ind in np.ndindex(x.shape[:-2]):
        for indx, m in enumerate(x[ind]):
            delta[ind+(indx,)] = m - m[:,np.newaxis]

    return delta

def calculate_residual(data_arrival_times, replica_arrival_times):
    # data_arrival_times: 2d ndarray nmodes x nfreqs
    # replica_arrival_times: Nd ndarray bulk_parameters x nmodes x nfreqs
    replica_delta = calculate_deltas(replica_arrival_times)
    data_delta = calculate_deltas(data_arrival_times)

    return replica_delta - data_delta


class AcousticInversionLikelihood:
    def __init__(self, data_arrival_times):
        self.data_arrival_times = data_arrival_times

    def likelihood_direct(self, replica_arrival_times, sigma = 1):
        residual = self.data_arrival_times - replica_arrival_times

        return np.exp(-np.nansum(nan_norm(residual, -1)**2,-1)/2/sigma**2)

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
        
        return np.exp(-(
            np.nansum(np.tril(
                nan_norm(residual_intra, -3)**2),(-2, -1)) +
            np.nansum(np.tril(
                nan_norm(residual_inter, -3)**2),(-2, -1)))
                    /2/sigma**2)


def sensitivity_maps(data_arrival_times, replica_arrival_times):
    # Intramodal: residual_intra shape (..., nmodes, nfreq, nfreq)
    # sum over the "other" frequency axis -> importance of freq i, within each mode
    residual_intra = calculate_residual(data_arrival_times, replica_arrival_times)
    intra_mode_freq = np.sum(residual_intra**2, axis=-1)          # (..., nmodes, nfreq)

    # Intermodal: residual_inter shape (..., nfreq, nmodes, nmodes)
    # sum over the "other" mode axis -> importance of mode m, at each frequency
    residual_inter = calculate_residual(data_arrival_times.mT, replica_arrival_times.mT)
    inter_freq_mode = np.sum(residual_inter**2, axis=-1)          # (..., nfreq, nmodes)
    inter_mode_freq = np.moveaxis(inter_freq_mode, -1, -2)        # (..., nmodes, nfreq)

    combined = intra_mode_freq + inter_mode_freq                  # (..., nmodes, nfreq)
    return combined, intra_mode_freq, inter_mode_freq
