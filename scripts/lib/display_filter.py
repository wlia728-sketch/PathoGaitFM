"""Manuscript display-only Gaussian filtering and shape-preserving interpolation."""
import numpy as np
from scipy.interpolate import PchipInterpolator


def display_curve(values, x=None, sigma=1.0, radius=3, n=1001):
    raw=np.asarray(values,float)
    if raw.ndim!=1 or len(raw)<2 or not np.isfinite(raw).all():raise ValueError('Expected finite waveform')
    x=np.arange(len(raw),dtype=float) if x is None else np.asarray(x,float)
    kernel=np.exp(-.5*(np.arange(-radius,radius+1)/sigma)**2);kernel/=kernel.sum()
    smooth=np.convolve(np.pad(raw,(radius,radius),mode='edge'),kernel,mode='valid')
    dense_x=np.linspace(x[0],x[-1],n)
    return dense_x,PchipInterpolator(x,smooth)(dense_x),smooth
