# pyADfit
A nested sampling approach to quasi-stellar object (QSO) accretion disc fitting.

This repository contains a Python module for modelling accretion discs around astrophysical objects. The module provides functions to calculate physical quantities related to accretion disks and perform parameter estimation using observational data.
The accretion disc model is the alpha-disc model (see [Shakura & Sunyaev 1976](https://ui.adsabs.harvard.edu/abs/1976MNRAS.175..613S/abstract)), while the parameter estimation can be performed either with [Nessai](https://nessai.readthedocs.io/en/latest/), [Raynest](https://pypi.org/project/raynest/) or [CPnest](https://pypi.org/project/cpnest/). 

## Dependencies
- numpy
- scipy
- matplotlib
- raynest
- CPNest
- nessai
- h5py
- pandas

## Installation
Clone this repository to your local machine:

```bash
git clone https://github.com/FabioRigamonti/pyADfit.git

Install the required dependencies using pip:

```bash
pip install -r requirements.txt

Or install it directly with pip:

```bash
pip install pyADfit
