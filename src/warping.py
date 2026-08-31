import numpy as np

warping_fun  = lambda t, tr: np.sqrt(t**2 + tr**2) 
warping_ifun = lambda t, tr: np.sqrt(t**2 - tr**2) 

def nonlinear_warping(y, fs, tr, t0 = None):
    N = len(y)

    if t0 == None:
        t0 = tr + 1/fs
    t = np.arange(N)/fs + t0

    tmin = t[0]
    tmax = t[-1]

    deltatn = tmax / fs / warping_ifun(tmax, tr)

    fsh = 2/deltatn
    
    # K = np.ceil((warping_ifun(tmax, tr) - warping_ifun(tmin, tr)) * fsh)
    K = np.ceil(warping_ifun(tmax, tr) * fsh)

    tw = np.arange(K)/fsh# + warping_ifun(tmin, tr)

    H  = np.sinc(fs*(warping_fun(tw, tr)[:, np.newaxis] - t))
    yw = np.sqrt(tw/warping_fun(tw, tr))[:, np.newaxis] * (H @ y[:,np.newaxis])

    return yw.squeeze(), tw, fsh

def inv_nonlinear_warping(y, fs, tr, fs_out, N):
    K = len(y)

    t_in = np.arange(K)/fs# + t0in

    t_out = (1 + np.arange(N))/fs_out + tr

    H  = np.sinc(fs*(warping_ifun(t_out, tr)[:, np.newaxis] - t_in))
    yw = np.sqrt(t_out/warping_ifun(t_out, tr))[:, np.newaxis] * (H @ y[:,np.newaxis])

    return yw.squeeze(), t_out
