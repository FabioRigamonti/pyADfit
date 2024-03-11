
import numpy as np
import matplotlib.pyplot as plt
from pyADfit.disc_launch import read_config_and_launch
# from disc_launch import read_config_and_lauch  # if pyADfit was not installed via pip 

def read_data(input_path):
    """
    Reads in data from a text file. USER DEFINED.

    Parameters
    ----------
    input_path : str
        Path to the input file.

    Returns
    -------
    xdata : np.ndarray
        Array of x-data.
    ydata : np.ndarray
        Array of y-data.
    yerr : np.ndarray
        Array of y-errors.

    """
    xdata,ydata,yerr = np.loadtxt(input_path,unpack=True,usecols=[0,1,2])

    indsort = np.argsort(xdata)
    xdata = xdata[indsort]
    ydata = ydata[indsort]
    yerr = yerr[indsort]

    xdata = 1e15*xdata 
    ydata = 1e45*ydata
    yerr = 1e45*yerr

    return xdata,ydata,yerr

if __name__=='__main__':
    config_path = './config.yaml'

    post_df,best = read_config_and_launch(config_path,read_data)

    plt.show()
