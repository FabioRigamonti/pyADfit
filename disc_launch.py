import numpy as np
import yaml
import pandas as pd
from disc_fit import analysis,calling_all
import os 
import matplotlib.pyplot as plt

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


def read_yaml(file_path):
    ''' 
    Function
    ----------------------------------------------------------------------
    ----------------------------------------------------------------------
    read yaml (i.e. configuration) file
        
        Parameters:
            file_path: str
                absolute path to read
        Returns:
    '''
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    

def read_config_and_launch(config_path):
    """
    This function reads in a YAML configuration file and launches the 
    parameter estimation process.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    post_df : pandas.DataFrame
        Posterior samples from the MCMC sampler.
    best : dict
        Best-fit parameters and their uncertainties.

    """

    config_path = './config.yaml'

    config = read_yaml(config_path)

    if config['Hyperparameter_estimator']['raynest']['use'] == True:
        hyperpar_dict = config['Hyperparameter_estimator']['raynest']
        which_sampler ='raynest'

    elif config['Hyperparameter_estimator']['CPnest']['use'] == True:
        hyperpar_dict = config['Hyperparameter_estimator']['CPnest']
        which_sampler ='CPnest'
    elif config['Hyperparameter_estimator']['nessai']['use'] == True:
        hyperpar_dict = config['Hyperparameter_estimator']['nessai']
        which_sampler ='nessai'
    else:
        print('Not using Nested Sampling, no other estimator is available, exiting')
        exit()

    fitting_parameters = config['Parameters']
    hyperparameter_disc_model = config['Hyperparameter_disc_model']

    names = []
    LaTex_names = []
    bounds = []
    vary = []
    value = []

    for k in fitting_parameters:
        names.append(fitting_parameters[k]['name'])
        LaTex_names.append(fitting_parameters[k]["LaTex_name"])
        bounds.append(fitting_parameters[k]["bounds"])
        vary.append(fitting_parameters[k]["vary"])
        value.append(fitting_parameters[k]["value"])  


    variable_quantities = {names[i]:bounds[i] for i in range(0,len(names)) if vary[i]==True}
    fixed_quantities = {names[i]:value[i] for i in range(0,len(names)) if vary[i]==False}
    variable_LaTex = {names[i]:LaTex_names[i] for i in range(0,len(names)) if vary[i]==True}
    fixed_LaTex = {names[i]:LaTex_names[i] for i in range(0,len(names)) if vary[i]==False}

    names_hyp = []
    value_hyp = []

    for k in hyperparameter_disc_model:
        names_hyp.append(hyperparameter_disc_model[k]['name'])
        value_hyp.append(hyperparameter_disc_model[k]["value"]) 

    hyperpar_disc_dict = {names_hyp[i]:value_hyp[i] for i in range(0,len(names_hyp))}

    input_path = config['Settings']['input_path']
    output_path = config['Settings']['output_path']

    xdata,ydata,yerr = read_data(input_path)

    if os.path.exists(output_path+"posterior.csv"):
        print('Parameter estimation file already exists, plotting diagnostics')

        post_df = pd.read_csv(output_path+"posterior.csv")
        best = analysis(post_df,xdata,ydata,yerr,output_path,fixed_quantities,\
                        variable_LaTex,hyperpar_disc_dict)

    else:    
        post_df,best = calling_all(xdata,ydata,yerr,output_path,hyperpar_dict,\
                    variable_quantities,fixed_quantities,variable_LaTex,\
                    hyperpar_disc_dict,which_sampler)
        
    return post_df,best


if __name__=='__main__':
    config_path = './config.yaml'

    post_df,best = read_config_and_launch(config_path)

    plt.show()