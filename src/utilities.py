import scipy.signal as sp
import numpy as np
from matplotlib.widgets import PolygonSelector
from matplotlib import path

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
