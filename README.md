# Paderbox: A collection of utilities for audio / speech processing

[![Build Status](https://dev.azure.com/fgnt/fgnt/_apis/build/status/fgnt.paderbox?branchName=master)](https://dev.azure.com/fgnt/fgnt/_build/latest?definitionId=2&branchName=master)
[![Azure DevOps tests](https://img.shields.io/azure-devops/tests/fgnt/fgnt/2/master)](https://dev.azure.com/fgnt/fgnt/_build/latest?definitionId=2&branchName=master)
[![Azure DevOps coverage](https://img.shields.io/azure-devops/coverage/fgnt/fgnt/2/master)](https://dev.azure.com/fgnt/fgnt/_build/latest?definitionId=2&branchName=master)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/fgnt/paderbox/master/LICENSE)

This repository started in late 2014 as an internal development repository for the [Communications Engineering Group](https://ei.uni-paderborn.de/nt/) at Paderborn University, Germany.
Over the years it emerged to a collection of [IO helper](https://github.com/fgnt/paderbox/tree/master/paderbox/io), [feature extraction](https://github.com/fgnt/paderbox/tree/master/paderbox/transform) modules and numerous smaller tools adding functionality to Numpy, Pandas, and others.

The main purpose here is to limit code duplication across our [other public repositories](https://github.com/fgnt).

We ensured that most functions/ classes contain Python Docstrings such that automatic tooltips for most functions are supported.
It was deliberately decided against a lengthy documentation: most emphasis is put on the Python Docstrings and code readability itself.


# Examples
Without going through all functions, we here select two examples which demonstrate why we rely on this very implementation.


## Short-time Fourier transform

The Short-time Fourier transform (STFT) is a widely used feature extraction method when dealing with time series such as audio/ speech.
Most repositories, including Deep Learning frameworks such as TensorFlow, provide an STFT implementation.
However, it is rarely seen, that these implementations allow an exact reconstruction when applying the STFT followed by an inverse STFT.

Two important issues often overseen are:
- How do I need to calculate the biorthogonal reconstruction window when using *any* STFT window function?
- How much padding depeding on STFT window length, DFT length, and shift is needed to compensate for fade-in, fade-out, and uneven signal length?

Our [STFT implementation](https://github.com/fgnt/paderbox/blob/master/paderbox/transform/module_stft.py) addresses aforementioned issues, can operate on any number of independent dimensions and is already battle tested in our publications on audio/ speech since 2015.
Numerous [STFT tests](https://github.com/fgnt/paderbox/blob/master/tests/transform_tests/test_stft.py) ensure that the code remains stable and in particular test for the aforementioned problems.

## Fast access to the IPython audio player

The function `paderbox.play.play()` is a somewhat elaborated wrapper around `IPython.display.Audio`.
A single function allows to play audio from the waveform, from the STFT signal, and from file.
It therefore serves as a great tool within Jupyter Notebooks and helps for quick inspection of simulation results.

# Installation
Install it from PyPI with pip
```bash
pip install paderbox[all]
```
The `[all]` flag is optional and indicates to install all dependencies.
Remove it, when you want to have the minimal dependencies.

Alternatively, you can clone this repository and install it as follows
```bash
git clone https://github.com/fgnt/paderbox.git
cd paderbox
pip install --editable .[all]
```

# How to cite?

There is no clear way how to cite this repository for research.
However, we would be grateful for direct imports from this repository if you use, e.g., the STFT.
We are also fine when you copy the code as long as it remains visible where you copied the code from.

If you use one of our other repositories relying on this work we would be thankful if you respect citation hints for that repository.
