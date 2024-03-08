import numpy as np
import raynest.model
from disc_model import disc_luminosity #,disco_bol_luminosity,disco_peak_nuLnu
import matplotlib.pyplot as plt
import yaml
import os
import pandas as pd
import cpnest.model
import nessai.model
from nessai.flowsampler import FlowSampler
import h5py

def likelihood(ydata,ymodel,yerr):
    """
    Calculates the likelihood function for a set of data and a model.

    Parameters
    ----------
    ydata : array_like
        The observed data values.
    ymodel : array_like
        The model predictions.
    yerr : array_like
        The uncertainty in the observed data.

    Returns
    -------
    lk : float
        The calculated likelihood value.

    """
    chi2 = (ydata-ymodel)/yerr
    lk   = chi2*chi2 + np.log(2*np.pi*yerr*yerr)

    return -0.5*np.sum(lk)

class Quasar_rayCP(raynest.model.Model):
    """
    Quasar class for accretion disc model raynest or CPnest.

    Parameters
    ----------
    freq : array_like
        Frequency array.
    lum : array_like
        Observed luminosity array. In solar luminosities.
    lum_err : array_like
        Uncertainty in the observed luminosity array. In solar luminosities.

    Attributes
    ----------
    freq : array_like
        Frequency array.
    lum : array_like
        Observed luminosity array. In solar luminosities.
    lum_err : array_like
        Uncertainty in the observed luminosity array. In solar luminosities.

    """

    names = []
    bounds = []

    def __init__(self,
                 freq,
                 lum,
                 lum_err,
                 variable_quantities,
                 fixed_quantities,
                 hyperpar_disc_dict,
                 *args,**kwargs):

        super(Quasar_rayCP,self).__init__()

        self.fixed_quantities    = fixed_quantities
        self.variable_quantities = variable_quantities
        self.bounds      = np.array(list(self.variable_quantities.values()))
        self.names       = list(self.variable_quantities.keys())
        self.names_fixed = list(fixed_quantities.keys())
        self.value       = list(fixed_quantities.values())     


        self.freq = freq 
        self.lum = np.log10(lum) 
        self.lum_err = lum_err/(lum*np.log(10)) 

        self.Rin = hyperpar_disc_dict['Rin']
        self.Rfin = hyperpar_disc_dict['Rfin']
        self.nu_min = hyperpar_disc_dict['nu_min']
        self.nu_max = hyperpar_disc_dict['nu_max']
   

    def log_likelihood(self,p):
        """
        Calculates the log-likelihood function for a set of data and a model.

        Parameters
        ----------
        p : dict
            The parameter values.

        Returns
        -------
        lk : float
            The calculated log-likelihood value.

        """
        if 'log10Mbh' in self.variable_quantities:
            m = p['log10Mbh']
        else:
            m = self.fixed_quantities['log10Mbh']

        if 'log10Mdot' in self.variable_quantities:
            mdot = p['log10Mdot']
        else:
            mdot = self.fixed_quantities['log10Mdot']

        if 'theta' in self.variable_quantities:
            theta = p['theta']
        else:
            theta = self.fixed_quantities['theta']

        theta = theta*np.pi/180

        ymodel = disc_luminosity(self.freq,m,mdot,theta,r_in=self.Rin,r_fin=self.Rfin)
        lk = likelihood(self.lum,ymodel,self.lum_err)
        return  lk


class Quasar_nessai(nessai.model.Model):
    """
    Quasar class for accretion disc model nessai.

    Parameters
    ----------
    freq : array_like
        Frequency array.
    lum : array_like
        Observed luminosity array. In solar luminosities.
    lum_err : array_like
        Uncertainty in the observed luminosity array. In solar luminosities.

    Attributes
    ----------
    freq : array_like
        Frequency array.
    lum : array_like
        Observed luminosity array. In solar luminosities.
    lum_err : array_like
        Uncertainty in the observed luminosity array. In solar luminosities.

    """

    names = []
    bounds = []

    def __init__(self,
                 freq,
                 lum,
                 lum_err,
                 variable_quantities,
                 fixed_quantities,
                 hyperpar_disc_dict,
                 *args,**kwargs):

        super(Quasar_nessai,self).__init__()

        self.fixed_quantities    = fixed_quantities
        self.variable_quantities = variable_quantities
        self.bounds = self.variable_quantities
        self.names       = list(self.variable_quantities.keys())
        self.names_fixed = list(fixed_quantities.keys())
        self.value       = list(fixed_quantities.values())     

        

        self.freq = freq 
        self.lum = np.log10(lum) 
        self.lum_err = lum_err/(lum*np.log(10)) 

        self.Rin = hyperpar_disc_dict['Rin']
        self.Rfin = hyperpar_disc_dict['Rfin']
        self.nu_min = hyperpar_disc_dict['nu_min']
        self.nu_max = hyperpar_disc_dict['nu_max']
   
    def log_prior(self, p):
        """
        Calculates the log-prior probability for a set of parameters.

        Parameters
        ----------
        p : dict
            The parameter values.

        Returns
        -------
        lprior : float
            The calculated log-prior value.

        """
        if not self.in_bounds(p).any():
            return -np.inf
    
        lp = np.log(self.in_bounds(p))
        
        return lp

    def log_likelihood(self,p):
        """
        Calculates the log-likelihood function for a set of data and a model.

        Parameters
        ----------
        p : dict
            The parameter values.

        Returns
        -------
        lk : float
            The calculated log-likelihood value.

        """
        if 'log10Mbh' in self.variable_quantities:
            m = p['log10Mbh']
        else:
            m = self.fixed_quantities['log10Mbh']

        if 'log10Mdot' in self.variable_quantities:
            mdot = p['log10Mdot']
        else:
            mdot = self.fixed_quantities['log10Mdot']

        if 'theta' in self.variable_quantities:
            theta = p['theta']
        else:
            theta = self.fixed_quantities['theta']

        theta = theta*np.pi/180

        ymodel = disc_luminosity(self.freq,m,mdot,theta,r_in=self.Rin,r_fin=self.Rfin)
        lk = likelihood(self.lum,ymodel,self.lum_err)
        return  lk

def run_nessai(freq,lum,lum_err,outdir,hyperpar_dict,variable_quantities,\
            fixed_quantities,hyperpar_disc_dict):
    """
    Runs RayNest to estimate the parameters of an accretion disc model.

    Parameters
    ----------
    freq : array_like
        Frequency array.
    lum : array_like
        Observed luminosity array [nu * Lnu].
    lum_err : array_like
        Uncertainty in the observed luminosity array.
    outdir : str
        Output directory.
    hyperpar_dict : dict
        RayNest Hyperparameter dictionary.
    variable_quantities : dict
        Dictionary of variable quantities.
    fixed_quantities : dict
        Dictionary of fixed quantities.
    hyperpar_disc_dict : dict
        Hyperparameter dictionary for the disc model.

    Returns
    -------
    df : pandas.DataFrame
        Posterior samples as a pandas DataFrame.

    """
    model = Quasar_nessai(freq,lum,lum_err,variable_quantities,fixed_quantities,hyperpar_disc_dict)
    names = list(model.names)
    names.append('logL')
    names.append('logP')


    work=FlowSampler(model,
                    output=outdir,
                    nlive= hyperpar_dict['nlive'],
                    prior_sampling= hyperpar_dict['prior_sampling'],
                    n_pool= hyperpar_dict['n_pool'],
                    resume_file= hyperpar_dict['resume_file'],
                    checkpointing= hyperpar_dict['checkpointing'],
                    checkpoint_on_training= hyperpar_dict['checkpoint_on_training'],
                    checkpoint_interval= hyperpar_dict['checkpoint_interval'])
    work.run()

    posterior = work.posterior_samples

    df = pd.DataFrame(posterior,columns=list(posterior.dtype.names))

    df.to_csv(outdir+'/posterior.csv',index=False)

    return df

def run_raynest(freq,lum,lum_err,outdir,hyperpar_dict,variable_quantities,\
        fixed_quantities,hyperpar_disc_dict):
    """
    Runs RayNest to estimate the parameters of an accretion disc model.

    Parameters
    ----------
    freq : array_like
        Frequency array.
    lum : array_like
        Observed luminosity array [nu * Lnu].
    lum_err : array_like
        Uncertainty in the observed luminosity array.
    outdir : str
        Output directory.
    hyperpar_dict : dict
        RayNest Hyperparameter dictionary.
    variable_quantities : dict
        Dictionary of variable quantities.
    fixed_quantities : dict
        Dictionary of fixed quantities.
    hyperpar_disc_dict : dict
        Hyperparameter dictionary for the disc model.

    Returns
    -------
    df : pandas.DataFrame
        Posterior samples as a pandas DataFrame.

    """
    model = Quasar_rayCP(freq,lum,lum_err,variable_quantities,fixed_quantities,hyperpar_disc_dict)
    names = list(model.names)
    names.append('logL')
    names.append('logP')

    work=raynest.raynest(model,
                         verbose=hyperpar_dict['verbose'],
                         nnest=hyperpar_dict['nnest'],
                         nensemble=hyperpar_dict['nensemble'],
                         nlive=hyperpar_dict['nlive'],
                         maxmcmc=hyperpar_dict['maxmcmc'],
                         nslice=hyperpar_dict['nslice'],
                         nhamiltonian=hyperpar_dict['nhamiltonian'],
                         resume=hyperpar_dict['resume'],
                         output=outdir,
                         periodic_checkpoint_interval=hyperpar_dict['periodic_checkpoint_interval'])
    work.run(corner = False)

    posterior = work.posterior_samples

    df = pd.DataFrame(posterior,columns=list(posterior.dtype.names))

    df.to_csv(outdir+'/posterior.csv',index=False)

    return df


def run_CPnest(freq,lum,lum_err,outdir,hyperpar_dict,variable_quantities,\
               fixed_quantities,hyperpar_disc_dict):
    """
    Runs CPNest to estimate the parameters of an accretion disc model.

    Parameters
    ----------
    freq : array_like
        Frequency array.
    lum : array_like
        Observed luminosity array [nu * Lnu].
    lum_err : array_like
        Uncertainty in the observed luminosity array.
    outdir : str
        Output directory.
    hyperpar_dict : dict
        CPnest Hyperparameter dictionary.
    variable_quantities : dict
        Dictionary of variable quantities.
    fixed_quantities : dict
        Dictionary of fixed quantities.
    hyperpar_disc_dict : dict
        Hyperparameter dictionary for the disc model.

    Returns
    -------
    df : pandas.DataFrame
        Posterior samples as a pandas DataFrame.

    """

    model = Quasar_rayCP(freq,lum,lum_err,variable_quantities,fixed_quantities,hyperpar_disc_dict)
    names = list(model.names)
    names.append('logL')
    names.append('logP')

    work=cpnest.CPNest(model,
                       verbose=hyperpar_dict['verbose'],
                       nlive=hyperpar_dict['nlive'],
                       maxmcmc=hyperpar_dict['maxmcmc'],
                       nslice=hyperpar_dict['nslice'],
                       nhamiltonian=hyperpar_dict['nhamiltonian'],
                       resume=hyperpar_dict['resume'],
                       output=outdir,
                       nthreads=hyperpar_dict['nthreads'],
                       periodic_checkpoint_interval=hyperpar_dict['periodic_checkpoint_interval'])
    work.run()

    posterior = work.get_posterior_samples()

    df = pd.DataFrame(posterior,columns=list(posterior.dtype.names))

    df.to_csv(outdir+'/posterior.csv',index=False)
    
    return df


def evidence(file):
    """
    Reads the evidence from an output file generated by Nessai.

    Parameters
    ----------
    file : str
        Path to the output file.

    Returns
    -------
    evi : float
        The evidence value.

    """
    data = h5py.File(file,mode='r')
    evi = data['log_evidence'][()]

    return evi

def analysis(post_df,xdata,ydata,yerr,outdir,fixed_quantities,variable_LaTex,hyperpar_disc_dict):
    """
    Function to analyze the posterior samples and plot the results.

    Parameters
    ----------
    post_df : pandas.DataFrame
        Posterior samples as a pandas DataFrame.
    xdata : np.ndarray
        Array of x-data. Frequency.
    ydata : np.ndarray
        Array of y-data. nu*Lnu
    yerr : np.ndarray
        Array of y-errors.
    outdir : str
        Output directory.
    fixed_quantities : dict
        Dictionary of fixed quantities.
    variable_LaTex : dict
        Dictionary of variable quantities with LaTeX labels.
    hyperpar_disc_dict : dict
        Hyperparameter dictionary for the disc model.

    Returns
    -------
    best : list
        Best-fit values for the parameters.

    """

    if os.path.exists(outdir+'/result.hdf5'):
        evi = evidence(outdir+'/result.hdf5')
        np.savetxt(outdir+'/evidence.txt',np.array([evi]),header='log_evidence')
    else:
        pass

    eta = 1/4./hyperpar_disc_dict['Rin']

    xplot = 10**np.linspace(14.5,16.,300)

    if 'log10Mbh' in fixed_quantities:
        Mbh = fixed_quantities['log10Mbh']
    else:
        Mbh = post_df['log10Mbh'].values
        Nsample = Mbh.shape[0]
    if 'log10Mdot' in fixed_quantities:
        Mdot = fixed_quantities['log10Mdot']
    else:
        Mdot = post_df['log10Mdot'].values
        Nsample = Mdot.shape[0]
    if 'theta' in fixed_quantities:
        theta = fixed_quantities['theta']
    else:
        theta = post_df['theta'].values
        Nsample = theta.shape[0]

    Mbh50,Mbh16,Mbh84 = np.percentile(Mbh,[50,16,84])
    Mdot50,Mdot16,Mdot84 = np.percentile(Mdot,[50,16,84])
    lambda_edd = np.log10(10**Mdot*eta)
    lambda_edd50,lambda_edd16,lambda_edd84 = np.percentile(lambda_edd,[50,16,84])
    theta50,theta16,theta84 = np.percentile(theta,[50,16,84])

    print('log10 Mbh: {:.4f} + {:.4f} -{:.4f}'.format(Mbh50,Mbh84-Mbh50,Mbh50-Mbh16))
    print('log10 Mdot/(Ledd/c^2): {:.4f} + {:.4f} -{:.4f}'.format(Mdot50,Mdot84-Mdot50,Mdot50-Mdot16))
    print('log10 lambda_edd (Lbol/Ledd): {:.4f} + {:.4f} -{:.4f}'.format(lambda_edd50,lambda_edd84-lambda_edd50,lambda_edd50-lambda_edd16))
    print('theta: {:.4f} + {:.4f} -{:.4f}'.format(theta50,theta84-theta50,theta50-theta16))

    

    import corner 
    samples = post_df[['log10Mbh',"log10Mdot","logL"]].values

    samples = np.concatenate((samples,lambda_edd.reshape(-1,1)), axis=1) 

    variable_LaTex = {**variable_LaTex,**{'logL':'logL', 'lambdaEdd':r'$\lambda_{Edd}$'}} 
    
    figure = corner.corner(samples,labels=list(variable_LaTex.values()),show_titles=True,quantiles=[0.16,0.50,0.84])
    figure.savefig(outdir+'corner.png')
    figure.show()
    

    bf_all = np.zeros((Nsample,xplot.shape[0]))

    for i in range(Nsample):
        if isinstance(Mbh,np.ndarray) and Mbh.shape[0] > 1:
            Mtmp = Mbh[i]
        else:
            Mtmp = Mbh
        if isinstance(Mdot,np.ndarray) and Mdot.shape[0] > 1:
            Mdottmp = Mdot[i]
        else:
            Mdottmp = Mdot
        if isinstance(theta,np.ndarray) and theta.shape[0] > 1:
            thetatmp = theta[i]
        else:
            thetatmp = theta
        bf = disc_luminosity(xplot,Mtmp,Mdottmp,thetatmp*np.pi/180)
        bf_all[i,:] = bf

    bf_median,bf_16,bf_84 = np.percentile(bf_all,[50,16,84],axis=0)

    fig = plt.figure()
    ax  = fig.add_subplot()
    ax.errorbar(np.log10(xdata),np.log10(ydata),yerr=yerr/(ydata*np.log(10)),fmt='.c',ms=10)
    ax.plot(np.log10(xplot),bf_median,'k',lw=3)
    #ax.plot(np.log10(xplot),bf_fix,'m',lw=3)
    ax.fill_between(np.log10(xplot),bf_84,y2=bf_median,color='k',alpha=0.25)
    ax.fill_between(np.log10(xplot),bf_median,y2=bf_16,color='k',alpha=0.25)
    
    ax.set_xlabel(r'$\nu~[Hz]$')
    ax.set_ylabel(r'$\nu L_{\nu}~[erg~Hz/s]$')

    #ax.set_ylim(44.5,47)

    data = np.concatenate((np.log10(xplot).reshape(-1,1),
                           bf_median.reshape(-1,1),
                           bf_84.reshape(-1,1),
                           bf_16.reshape(-1,1)),axis=1)

    np.savetxt(outdir+'best_fit_data.txt',data)

    fig.show()
    fig.savefig(outdir+'bestfit.png')


    return [Mbh50,Mdot50,lambda_edd50,theta50]


def calling_all(xdata,ydata,yerr,outdir,hyperpar_dict,variable_quantities,\
                fixed_quantities,variable_LaTex,hyperpar_disc_dict,which_sampler):
    """
    This function runs the parameter estimation and analysis for the accretion disc model.

    Parameters
    ----------
    xdata : np.ndarray
        Array of x-data. Frequency.
    ydata : np.ndarray
        Array of y-data. nu*Lnu
    yerr : np.ndarray
        Array of y-errors.
    outdir : str
        Output directory.
    hyperpar_dict : dict
        Hyperparameter dictionary for the sampler.
    variable_quantities : dict
        Dictionary of variable quantities.
    fixed_quantities : dict
        Dictionary of fixed quantities.
    variable_LaTex : dict
        Dictionary of variable quantities with LaTeX labels.
    hyperpar_disc_dict : dict
        Hyperparameter dictionary for the disc model.
    which_sampler : str
        Which sampler to use.

    Returns
    -------
    post_df : pandas.DataFrame
        Posterior samples as a pandas DataFrame.
    best : list
        Best-fit values for the parameters.

    """
    if which_sampler == 'CPnest':
        post_df = run_CPnest(xdata,ydata,yerr,outdir,hyperpar_dict,variable_quantities,\
                    fixed_quantities,hyperpar_disc_dict)
    elif which_sampler =='raynest':
        post_df = run_raynest(xdata,ydata,yerr,outdir,hyperpar_dict,variable_quantities,\
                    fixed_quantities,hyperpar_disc_dict)
    elif which_sampler =='nessai':
        post_df = run_nessai(xdata,ydata,yerr,outdir,hyperpar_dict,variable_quantities,\
                    fixed_quantities,hyperpar_disc_dict)
    else:
        raise ValueError('Unknown sampler {}'.format(which_sampler))


    best = analysis(post_df,xdata,ydata,yerr,outdir,fixed_quantities,variable_LaTex,hyperpar_disc_dict)
    return post_df,best



#if __name__=='__main__':




