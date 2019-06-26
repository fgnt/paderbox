"""
Source: https://www.audiolabs-erlangen.de/fau/professor/habets/software/noise-generators

Generating sensor signals for a 1D sensor array in a spherically
isotropic noise field [1,2]

   z = sinf_3D(P,len,params)

Input parameters:
   P            : sensor positions
   len          : desired data length
   params.fs    : sample frequency
   params.c     : sound velocity in m/s
   params.N_phi : number of cylindrical angles

Output parameters:
   z            : output signal

References:
   [1] E.A.P. Habets and S. Gannot, 'Generating sensor signals
       in isotropic noise fields', Submitted to the Journal
       of the Acoustical Society of America, May, 2007.
   [2] E.A.P. Habets and S. Gannot, 'Comments on Generating
       sensor signals in isotropic noise fields', Technical Report,
       Imperial College London, Sept. 2010.

Authors:  E.A.P. Habets and S. Gannot

History:  2004-11-02 - Initial version
          2010-09-16 - Near-uniformly distributed points over S^2

Copyright (C) 2007,2010 E.A.P. Habets

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You can obtain a copy of the GNU General Public License from
  http://www.gnu.org/copyleft/gpl.html or by writing to
  Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Spherical (isotropic) noise MATLAB implementation can be found on:
https://www.audiolabs-erlangen.de/fau/professor/habets/software/noise-generators
"""

from math import acos, sqrt
from math import pi, sin, cos, ceil, floor


import numpy as np
from numpy import zeros
from numpy.fft import irfft as ifft
from numpy.fft import fft as fft
from numpy.random import randn
from scipy.signal import detrend


def sinf_3d(positions, signal_length, sample_rate=8000, c=340, N=256):
    """

    :param positions: position of microphones in cartesian coordinates,
        shape 3 x #channels
    :param signal_length: signal length
    :param sample_rate: sample frequency
    :param c: sound velocity in m/s
    :param N: number of cylindrical angles
    :return:

    Example:

    >>> M = 4 # Number of sensors
    >>> P = randn(3, M)
    >>> N = 300
    >>> sinf_3d(P, N)

    """

    M = positions.shape[1]  # Number of sensors
    NFFT = 2 ** ceil(np.log2(signal_length))  # Number of frequency bins
    X = zeros((M, NFFT // 2 + 1), dtype=np.complex128)

    w = 2 * pi * sample_rate * np.arange(0, NFFT // 2 + 1) / NFFT

    # Generate N points that are near-uniformly distributed over S^2
    phi = zeros(N)
    theta = zeros(N)
    for k in range(N):
        h = -1 + 2 * (k) / (N - 1)
        phi[k] = acos(h)
        if k == 0 or k == N - 1:
            theta[k] = 0
        else:
            theta[k] = (theta[k - 1] + 3.6 / sqrt(N * (1 - h ** 2))) % 2 * pi

    # Calculate relative positions
    P_rel = zeros((3, M))
    for m in range(M):
        P_rel[:, m] = positions[:, m] - positions[:, 0]

    # Calculate sensor signals in the frequency domain
    for idx in range(N):
        X_prime = randn(NFFT // 2 + 1) + 1j * randn(NFFT // 2 + 1)
        X[0, :] = X[0, :] + X_prime
        for m in range(1, M):

            # It is unclear, why this differs from the implementation in
            # `paderbox.math.vector.sph2cart()`.
            v = np.array([
                cos(theta[idx]) * sin(phi[idx]),
                sin(theta[idx]) * sin(phi[idx]),
                cos(phi[idx])
            ])
            Delta = v @ P_rel[:, m]
            X[m, :] = X[m, :] + X_prime * np.exp(-1j * Delta * w / c)

    X = X / sqrt(N)

    # Transform to time domain
    X = [
        sqrt(NFFT) * np.real(X[:, 0]),
        sqrt(NFFT // 2) * X[:, 1:-1],
        sqrt(NFFT) * np.real(X[:, -1]),
        sqrt(NFFT / 2) * np.conj(X[:, -2:0:-1])
    ]
    X = [x if x.ndim is 2 else x[:, None] for x in X]
    X = np.concatenate(X, axis=1)
    z = np.real(ifft(X, NFFT, 1))

    # Truncate output signals
    return z[:, :signal_length]


def _mycohere(
        x,
        y,
        nfft=256,
        sample_rate=8000,
        window=None,
        noverlap=None,
        p=.95,
        dflag='none',
):
    """Coherence function estimate.

    Cxy = COHERE(X,Y,NFFT,Fs,WINDOW) estimates the coherence of X and Y
    using Welch's averaged periodogram method.  Coherence is a function
    of frequency with values between 0 and 1 that indicate how well the
    input X corresponds to the output Y at each frequency.  X and Y are
    divided into overlapping sections, each of which is detrended, then
    windowed by the WINDOW parameter, then zero-padded to length NFFT.
    The magnitude squared of the length NFFT DFTs of the sections of X and
    the sections of Y are averaged to form Pxx and Pyy, the Power Spectral
    Densities of X and Y respectively. The products of the length NFFT DFTs
    of the sections of X and Y are averaged to form Pxy, the Cross Spectral
    Density of X and Y. The coherence Cxy is given by
        Cxy = (abs(Pxy).^2)./(Pxx.*Pyy)
    Cxy has length NFFT/2+1 for NFFT even, (NFFT+1)/2 for NFFT odd, or NFFT
    if X or Y is complex. If you specify a scalar for WINDOW, a Hanning
    window of that length is used.  Fs is the sampling frequency which does
    not effect the cross spectrum estimate but is used for scaling of plots.

    [Cxy,F] = COHERE(X,Y,NFFT,Fs,WINDOW,NOVERLAP) returns a vector of freq-
    uencies the same size as Cxy at which the coherence is computed, and
    overlaps the sections of X and Y by NOVERLAP samples.

    COHERE(X,Y,...,DFLAG), where DFLAG can be 'linear', 'mean' or 'none',
    specifies a detrending mode for the prewindowed sections of X and Y.
    DFLAG can take the place of any parameter in the parameter list
    (besides X and Y) as long as it is last, e.g. COHERE(X,Y,'mean');

    COHERE with no output arguments plots the coherence in the current
    figure window.

    The default values for the parameters are NFFT = 256 (or LENGTH(X),
    whichever is smaller), NOVERLAP = 0, WINDOW = HANNING(NFFT), Fs = 2,
    P = .95, and DFLAG = 'none'.  You can obtain a default parameter by
    leaving it off or inserting an empty matrix [], e.g.
    COHERE(X,Y,[],10000).

    See also PSD, CSD, TFE.
    ETFE, SPA, and ARX in the Identification Toolbox.

    Author(s): T. Krauss, 3-31-93
    Copyright (c) 1988-98 by The MathWorks, Inc.
    $Revision: 1.1 $  $Date: 1998/06/03 14:42:19 $
    """

    #error(nargchk(2,7,nargin))
    #x = varargin{1};
    #y = varargin{2};
    #[msg,nfft,Fs,window,noverlap,p,dflag]=psdchk(varargin(3:end),x,y);
    #error(msg)
    if window is None:
        window = np.hanning(nfft)
    if not noverlap:
        noverlap = 0.75*nfft

    # compute PSD and CSD
    #window = window(:);
    n = len(x)		# Number of data points
    nwind = len(window) # length of window
    if n < nwind:    # zero-pad x , y if length is less than the window length
        x = np.pad(np.diag(x), (0, nwind - n), mode='constant')
        y = np.pad(np.diag(x), (0, nwind - n), mode='constant')
        # x(nwind)=0;
        # y(nwind)=0;
        n=nwind

    #x = x(:);		# Make sure x is a column vector
    #y = y(:);		# Make sure y is a column vector
    k = floor((n-noverlap) / (nwind-noverlap))	# Number of windows
                        # (k = fix(n/nwind) for noverlap=0)
    index = np.arange(nwind)

    Pxx = zeros(nfft)
    Pxx2 = zeros(nfft)
    Pyy = zeros(nfft)
    Pyy2 = zeros(nfft)
    Pxy = zeros(nfft, dtype=np.complex128)
    Pxy2 = zeros(nfft)
    for i in range(k): # 1:k
        if dflag == 'none':
            xw = window * x[index]
            yw = window * y[index]
        elif dflag == 'linear':
            xw = window * detrend(x[index])
            yw = window * detrend(y[index])
        else:
            xw = window * detrend(x[index], type='constant')
            yw = window * detrend(y[index], type='constant')

        index += int(nwind - noverlap)
        Xx = fft(xw,nfft)
        Yy = fft(yw,nfft)
        Xx2 = abs(Xx)**2
        Yy2 = abs(Yy)**2
        Xy2 = Yy * np.conj(Xx)
        Pxx += Xx2
        Pxx2 += abs(Xx2)**2
        Pyy += Yy2
        Pyy2 += abs(Yy2)**2
        Pxy += Xy2
        Pxy2 += abs(Xy2)**2

    # Select first half
    # if ~any(any(imag([x y])~=0)),   # if x and y are not complex
    select = np.arange(nfft) # 1:nfft;
    if (np.iscomplexobj(x) or np.iscomplexobj(y)) is False:
        if nfft % 2:     # nfft odd
            select = np.arange((nfft+1) // 2) # [1:(nfft+1)/2];
        else:
            select = np.arange((nfft) // 2 + 1) #  [1:nfft/2+1];   # include DC AND Nyquist

        Pxx = Pxx[select]
        Pxx2 = Pxx2[select]
        Pyy = Pyy[select]
        Pyy2 = Pyy2[select]
        Pxy = Pxy[select]
        Pxy2 = Pxy2[select]


    #Coh = (abs(Pxy).^2)./(Pxx.*Pyy);             # coherence function estimate
    Coh = Pxy / np.sqrt(Pxx * Pyy)
    freq_vector = (select - 1) * sample_rate / nfft

    # set up output parameters
    #if (nargout == 2):
    #Cxy = Coh
    #f = freq_vector

    return Coh, freq_vector
    #elif (nargout == 1):
    #   Cxy = Coh
    #elif (nargout == 0):   # do a plot
       #newplot;
       #plot(freq_vector,Coh), grid on
       #xlabel('Frequency'), ylabel('Coherence Function Estimate')

