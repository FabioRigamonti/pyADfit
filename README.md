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
```

Move into the directory where you have downloaded the repository and install the required dependencies using pip:

```bash
pip install -r requirements.txt
```
Star your fitting process by importing the proper libraries (see the example below or run the provided test)

Or install it directly with pip:

```bash
pip install pyADfit
```

## Usage

To fit quasar accretion disc data, follow these steps:

1. Define your input data in a text file with three columns: x-data [nu, i.e. frequency], y-data [log10 nu*Lnu], and y-errors.
2. Create a YAML configuration file specifying the hyperparameters, see "config.yaml" in the example directory, fitting parameters, and other settings.
3. Define your own "read_data" function to read and the path to the configuration file
4. Import the "read_config_and_launch" function from "disc_launch"
5. Run the parameter estimation by calling the "read_config_and_launch"

## Example

```python

#from disc_launch import read_config_and_launch # if installed via github
from pyADfit import read_config_and_launch      # if installed via pip
import matplotlib.pyplot as plt 

def read_data(file_path):
  your function here

  return xdata,ydata,yerr

if __name__=='__main__':
  config_path = './config.yaml'

  post_df,best = read_config_and_launch(config_path,read_data)
  plt.show()
```

  
