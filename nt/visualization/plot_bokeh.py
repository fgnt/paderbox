from nt.visualization.new_cm import viridis_hex
from bokeh.plotting import figure
import numpy as np

def plot_image_2d(signal, shared_range=None, plot_height=300, plot_width=900,
                  title=None):
    """ Generic function to plot a 2d matrix as an image

    :param signal: 2d matrix
    :param shared_range: Tuple of (x_range, y_range). Can be used to link
        different figures (e.g. zooming/panning will be mirrored)
    :param plot_height: Height of the figure
    :param plot_width: Width of the figure
    :return: Figure object
    """
    signal = signal.T
    if shared_range is None:
        p = figure(plot_height=plot_height, plot_width=plot_width,
                   x_range=(0, signal.shape[1]), y_range=(0, signal.shape[0]))
    else:
        x_range, y_range = shared_range
        p = figure(plot_height=plot_height, plot_width=plot_width,
                   x_range=x_range, y_range=y_range)
    p.image([signal], x=[0], y=[0], dw=[signal.shape[1]], dh=[signal.shape[0]],
            palette=viridis_hex)
    if title:
        p.title = title
    return p


def plot_spectrum(spectrum, **kwargs):
    """ Plots a spectrum

    Possible kwargs:
        - shared_range
        - plot_height
        - plot_width

    :param spectrum: Spectrum to plot
    :param kwargs: kwargs for `plot_image_2d`
    :return: Figure object
    """
    return plot_image_2d(spectrum, **kwargs)


def plot_stft(stft, **kwargs):
    """ Plots a stft signal

    Possible kwargs:
        - shared_range
        - plot_height
        - plot_width

    :param spectrum: Stft signal to plot
    :param kwargs: kwargs for `plot_image_2d`
    :return: Figure object
    """
    return plot_spectrum(np.log10(np.abs(stft)), **kwargs)