import scipy.signal as sp
import numpy as np
from matplotlib.widgets import PolygonSelector
from matplotlib import path

def simulated_annealing(
    objective,
    domains,
    initial_solution=None,
    maximize=False,
    T0=100.0,
    Tmin=1e-5,
    cooling_rate=0.995,
    max_iter=10000,
    seed=None,
):
    """
    Minimize or maximize an objective function over discrete variables.

    Parameters
    ----------
    objective : callable
        Function that takes a solution (array) and returns a scalar.

    domains : list of array-like
        Allowed values for each variable.

    initial_solution : array-like, optional
        Initial solution. If None, sampled randomly.

    maximize : bool
        If True, maximize the objective. Otherwise minimize.

    T0 : float
        Initial temperature.

    Tmin : float
        Minimum temperature.

    cooling_rate : float
        Multiplicative cooling factor (0 < cooling_rate < 1).

    max_iter : int
        Maximum number of iterations.

    seed : int, optional
        Random seed.

    Returns
    -------
    best_solution : ndarray
        Best solution found.

    best_value : float
        Objective value of the best solution.

    history : ndarray
        Best objective value at each iteration.
    """

    rng = np.random.default_rng(seed)
    domains = [np.asarray(d) for d in domains]

    n_variables = len(domains)

    if n_variables == 0 or any(len(d) == 0 for d in domains):
        raise ValueError("Every variable must have a non-empty domain.")

    if not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate must be between 0 and 1.")

    # Convert maximization into minimization
    sign = -1 if maximize else 1

    # Generate initial solution
    if initial_solution is None:
        current = np.array([
            rng.choice(d) for d in domains
        ])
    else:
        current = np.asarray(initial_solution).copy()

        if len(current) != n_variables:
            raise ValueError("Initial solution has incorrect length.")

        for i, value in enumerate(current):
            if not np.any(domains[i] == value):
                raise ValueError(
                    f"Initial value {value} is not in domain {i}."
                )
                
    current_value = sign * objective(current)

    best = current.copy()
    best_value = current_value

    history = []

    # Annealing loop
    T = T0

    for iteration in range(max_iter):

        if T < Tmin:
            break

        # Generate a neighboring solution
        neighbor = current.copy()

        # Select one variable to modify
        i = rng.integers(n_variables)

        # Select a different value from its domain
        alternatives = domains[i][domains[i] != current[i]]

        if len(alternatives) > 0:
            neighbor[i] = rng.choice(alternatives)

            neighbor_value = sign * objective(neighbor)

            delta = neighbor_value - current_value

            # Metropolis criterion
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current = neighbor
                current_value = neighbor_value

            # Update global best
            if current_value < best_value:
                best = current.copy()
                best_value = current_value

        history.append(sign * best_value)

        # Cool down
        T *= cooling_rate

    return best, sign * best_value, np.asarray(history)

def field_to_timeseries(X, f, t0, N, fs):
    """
    Convert a narrowband complex frequency-domain field to a real time series.

    The one-sided spectrum X, sampled on a uniform frequency grid f, is
    time-shifted so the output starts at t0, inverse-transformed, and
    shifted from baseband up to its true frequency band, assuming the
    convention x(t) = sum_k X[k] exp(+j*2*pi*f[k]*t).

    Parameters
    ----------
    X : array_like, complex, shape (K,)
        Complex spectral values (e.g., a Fourier coefficient or acoustic
        field) at each frequency in `f`.
    f : array_like, real, shape (K,)
        Frequencies (Hz) corresponding to X. Must be uniformly spaced with
        spacing df = fs / N, so that each frequency falls on an FFT bin
        offset from f[0]. Only f[0] and the spacing are used to
        place the band.
    t0 : float
        Start time (s) of the output time series. The spectrum is phase-
        shifted by exp(j*2*pi*f*t0), so x[0] corresponds to time t0.
    N : int
        Number of samples in the output time series (also the IFFT length).
        If N < K, the spectrum is truncated; if N > K, it is zero-padded.
        The record length is N / fs.
    fs : float
        Sampling rate (Hz) of the output time series. Should satisfy
        fs > 2 * f[-1] to avoid aliasing.

    Returns
    -------
    x : ndarray, real, shape (N,)
        Time series sampled at times t0 + n/fs, n = 0..N-1.

    Notes
    -----
    Steps:
      1. Time shift: X <- X * exp(j*2*pi*f*t0).
      2. ifft(X, N) places X[k] in bin k, i.e., a baseband signal whose
         lowest frequency is 0 Hz rather than f[0].
      3. Multiplying by exp(j*2*pi*n*f[0]/fs) shifts the band up to its
         true frequencies, f[0]..f[-1].
      4. The real part is taken. No factor of 2 is applied for the
         discarded negative frequencies, and numpy's ifft includes a 1/N
         scaling, so the amplitude is X/N per component, not X.

    The IFFT is circular, so the signal is periodic with period N/fs;
    energy delayed beyond the record length wraps around to the start.

    Examples
    --------
    >>> fs, N, t0 = 1000.0, 1024, 0.0
    >>> f = 100.0 + np.arange(16) * fs / N
    >>> X = np.ones(16, dtype=complex)
    >>> x, tx = field_to_timeseries(X, f, t0, N, fs)
    """
    X = X * np.exp(1j * 2 * np.pi * f * t0)

    x = np.real(np.fft.ifft(X, int(N)) * np.exp(1j * 2 * np.pi * np.arange(N) * f[0] / fs))
    tx = np.arange(N) / fs + t0  # time axis; currently unused

    return x, tx


def find_peaks(Z, y=None, x=None, npeaks=None, min_height=-np.inf):
    peak_indices_x = list()
    peak_indices_y = list()
    
    
    for ind_y, z in enumerate(Z):
        # Find peaks and heihgts
        peak_indices_now, peak_properties = sp.find_peaks(z, height=min_height)
        # Sort peak indices by height
        if not npeaks == None:
            sorted_peak_indices = np.argsort(peak_properties['peak_heights'])
            # Grab the NPEAKS tallest peaks
            peak_indices_now = peak_indices_now[sorted_peak_indices[-npeaks:]]

        peak_indices_x.append(peak_indices_now)
        peak_indices_y.append(np.ones(len(peak_indices_now))*ind_y)


    peak_indices_x = np.concatenate(peak_indices_x)
    peak_indices_y = np.concatenate(peak_indices_y)

    peaks = np.array([peak_indices_x, peak_indices_y]).T

    if x is None and y is None:
        return peaks

    if not x is None:
        peaks[:,0] = x[peak_indices_x]

    if not y is None:
        peaks[:,1] = y[np.int32(peak_indices_y)]


    return peaks


def wideband_pulse(pulse_duration, fs, bw, fc, window=sp.windows.hamming):
    """
    Generate a windowed, wideband pulse (cosine-modulated sinc).

    Parameters
    ----------
    pulse_duration : float
        Pulse duration in seconds.
    fs : float
        Sampling frequency in Hz.
    bw : float
        Bandwidth in Hz.
    fc : float
        Center (carrier) frequency in Hz.
    window : callable, optional
        A window function taking an integer length N and returning
        an array of length N (e.g. scipy.signal.windows.hamming,
        scipy.signal.windows.hann, np.blackman, ...).
        Defaults to scipy.signal.windows.hamming.

    Returns
    -------
    signal : ndarray
        The generated pulse, length Npulse = ceil(pulse_duration * fs).
    """
    Npulse = int(np.ceil(pulse_duration * fs))

    w = window(Npulse)

    n = np.arange(Npulse) - np.floor(Npulse / 2)

    signal = (2 * bw / fs) * np.sinc(n * bw / fs) * \
             np.cos(n * 2 * np.pi * fc / fs) * w

    # signal = signal / np.abs(signal @ np.exp(1j * 2 * np.pi * n * fc / fs).conj())
    # signal = signal / np.sum(signal)

    return signal


def scramble_nums(x):
    
    c = np.array([-1, 0, 1])
    x_out = np.empty(x.shape, dtype=np.int64)

    for ind, num in enumerate(x):
        mask = np.array([True,]*3)
        if num == 0:
            mask[0] = False
        if ind > 0:
            if x_out[ind-1] == num-1:
                mask[0] = False
            if x_out[ind-1] == num:
                mask[1] = False
            if x_out[ind-1] == num + 1:
                mask[2] = False
        
        x_out[ind] = num + np.random.choice(c[mask])
    
    return x_out

class MyPolygonSelector:
    """Wrapper around matplotlib's PolygonSelector widget.

    Provides a simple interface for interactively drawing polygons on a
    matplotlib Axes and storing the resulting shapes as matplotlib.path.Path
    objects.

    Attributes:
        ax (matplotlib.axes.Axes): The Axes on which polygons are drawn.
        polygons (list[matplotlib.path.Path]): Collected polygons, one per
            completed selection.
        selector (matplotlib.widgets.PolygonSelector): The active selector
            widget, created when `select()` is called.
    """

    def __init__(self, ax):
        """Initialize the polygon selector.

        Args:
            ax (matplotlib.axes.Axes): The Axes to attach the selector to.
        """
        self.ax = ax
        self.polygons = list()

    def on_select(self, vertices):
        """Callback invoked by PolygonSelector when a polygon is completed.

        Converts the selected vertices into a matplotlib.path.Path and
        appends it to `self.polygons`.

        Args:
            vertices (list[tuple[float, float]]): The (x, y) vertices of the
                polygon drawn by the user.
        """
        self.polygons.append(path.Path(vertices))

    def select(self):
        """Create and activate the PolygonSelector widget on `self.ax`.

        Stores the widget in `self.selector`.
        User-drawn polygons are appended to `self.polygons` via `on_select`
        every time the user:
        1. Completes a polygon,
        2. Moves a vertex of a completed polygon.
        Knowing that, make sure you don't move any vertex once a polygon is
        completed, as it will generate extra (and likely unwanted) polygon
        entries.
        If you happen to place a vertex in the wrong place, press ESC
        to start the polygon over.
        """
        self.selector = PolygonSelector(self.ax, self.on_select)
