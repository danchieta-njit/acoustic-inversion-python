import numpy as np


def calculate_deltas(x):
    #if x.ndim == 2:
    #    x = x[np.newaxis,...]
        
    delta = np.zeros(x.shape[:-1] + (x.shape[-1], x.shape[-1]))

    for ind in np.ndindex(x.shape[:-2]):
        for indx, m in enumerate(x[ind]):
            delta[ind+(indx,)] = m - m[:,np.newaxis]

    return delta

def calculate_residual(data_arrival_times, replica_arrival_times):
    replica_delta = calculate_deltas(replica_arrival_times)
    data_delta = calculate_deltas(data_arrival_times)

    return replica_delta - data_delta


class AcousticInversionLikelihood:
    def __init__(self, data_arrival_times):
        self.data_arrival_times = data_arrival_times

    def likelihood_direct(self, replica_arrival_times, sigma = 1):
        residual = self.data_arrival_times - replica_arrival_times

        return np.exp(-np.sum(np.linalg.norm(residual, 2,2)**2,1)/2/sigma**2)

    def likelihood_intramodal(self, replica_arrival_times):
        residual = calculate_residual(self.data_arrival_times.mT,
                                      replica_arrival_times.mT)
        
        return np.exp(-np.linalg.norm(residual, 2, -3)**2)

    def likelihood_intramodal(self, replica_arrival_times):
        residual = calculate_residual(self.data_arrival_times,
                                      replica_arrival_times)
        
        return np.exp(-np.linalg.norm(residual, 2, -3)**2)

    def likelihood(self, replica_arrival_times, sigma = 1):
        residual_intra = calculate_residual(self.data_arrival_times.mT,
                                            replica_arrival_times.mT)
        residual_inter = calculate_residual(self.data_arrival_times,
                                            replica_arrival_times)
        
        return np.exp(-(
            np.sum(np.tril(
                np.linalg.norm(residual_intra, 2, -3)**2),(-2, -1)) +
            np.sum(np.tril(
                np.linalg.norm(residual_inter, 2, -3)**2),(-2, -1)))
                    /2/sigma**2)
