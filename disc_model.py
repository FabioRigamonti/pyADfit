import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad,quad_vec,dblquad


G =  6.67408e-8                 # gravitational constant in cm^3 g^-1 s^-2
m_p = 1.6726219e-24             # proton mass in g
c = 2.99792458e10               # light speed of light in cm s^-1
L_sun = 3.828e33                # solar luminosity in erg s^-1
R_sun = 6.95700000e10           # solar radius in cm
sigma = 5.670367e-5             # stefan-boltzmann constant in erg cm^−2 s^−1 K^−4
h = 6.62607004e-27              # planck constant in erg s
k = 1.38064852e-16              # Boltzman const erg K−1
M_sun = 1.98847542e33           # mass of sun in g
sigma_tmp = 6.65245872e-25      # thompson cross section in cm^2


def r_swarz(m):
    """
    Calculates the Schwarzschild radius of a given mass.

    Parameters
    ----------
    m : float
        The mass of the object.

    Returns
    -------
    r_schw : float
        The Schwarzschild radius of the object.

    """
    return (2 * G * m) / (c ** 2)

def l_edd (m):
    """
    Calculates the luminosity of an Eddington-Rutherford disk at a given mass.

    Parameters
    ----------
    m : float
        The mass of the disk in solar masses.

    Returns
    -------
    lum : float
        The luminosity of the disk in erg/s.

    """
    return ( ( 4 * np.pi * c * G * m * m_p )/ sigma_tmp )




def temperatura (r, m,m_dot,r_in):
    """
    Calculates the temperature of alpha-disc.

    Parameters
    ----------
    r : float / numpy array
        The distance from the center.
    m : float
        The mass of the object.
    m_dot : float
        Accretion rate.
    r_in : float
        The inner radius of the potential well.

    Returns
    -------
    T : float / numpy array
        The temperature of the object.

    """
    a = 3 * G * m * m_dot
    b = 8 * np.pi * sigma * r**3
    d = (1 - (r_in/r)**0.5)
    
    return ((a/b)*(d))**0.25 


def brightness(nu,T):
    """
    Calculates the blackbody radiation intensity at a given frequency and temperature.

    Parameters
    ----------
    nu: float / numpy array
        Frequency in Hz.
    T: float 
        Temperature in Kelvin, see temperatura(r,m,m_dot,r_in)

    Returns
    -------
    br: float
        Blackbody radiation intensity at the given frequency and temperature.

    """
    return ((2*h/(c**2))*(nu**3/(np.exp(h*nu/(k*T))-1)))


def integrand (r, nu, m, m_dot, r_in) :
    """
    Calculates the integral of the radiation intensity over the radius.

    Parameters
    ----------
    r : float / numpy array
        Distance from the center.
    nu : float
        Frequency in Hz.
    m : float
        The mass of the object.
    m_dot : float
        Accretion rate.
    r_in : float
        The inner radius of the potential well.

    Returns
    -------
    I : float
        The integral of the radiation intensity over the radius.

    """
    return  4*np.pi*np.pi*r*brightness(nu,temperatura(r,m,m_dot,r_in))


def disc_luminosity(freq,m,m_dot,theta, r_in=3,r_fin=1e4) : 
    """
    Calculates the luminosity of an accretion disk at a given frequency, mass, accretion rate, and viewing angle.

    Parameters
    ----------
    freq : float or numpy.ndarray
        Frequency of the radiation in Hz.
    m : float
        The mass of the disk in log10 solar masses.
    m_dot : float
        The accretion rate of the disk in log10 Eddington luminosity.
    theta : float
        The angle between the line of sight and the major axis of the disk.
    r_in : float, optional
        Inner radius of the accretion disk in unit of Schwarzschild radii. The default is 3. 
    r_fin : float, optional
        Outer radius of the accretion disk in unit of Schwarzschild radii. The default is 1e4.
    Returns
    -------
    lum : float or numpy.ndarray
        The log10 luminosity of the disk multiplied by the frequency.

    """
    m = 10**m * M_sun
   
    m_dot =  10**m_dot*l_edd(m)/(c**2)

    rs = r_swarz(m)

    r_in *= rs
    r_fin *= rs 

    integrand2 = lambda r : integrand(r,freq,m,m_dot,r_in)

    lumx,_ = quad_vec(integrand2,r_in,r_fin)
    
    return np.log10(freq*lumx*2*np.cos(theta))


def disc_bol_luminosity(m,m_dot, r_in=3,r_fin=1e4,nu_min=10**10,nu_max=10**16.5): 
    """
    Calculates the bolometric luminosity of an accretion disk at a given mass and accretion rate.

    Parameters
    ----------
    m : float
        The mass of the disk in log10 solar masses.
    m_dot : float
        The accretion rate of the disk in log10 Eddington luminosities.
    r_in : float, optional
        Inner radius of the accretion disk in unit of Schwarzschild radii. The default is 3. 
    r_fin : float, optional
        Outer radius of the accretion disk in unit of Schwarzschild radii. The default is 1e4.
    nu_min : float, optional
        Minimum frequency for integrating the bolometric luminosity. The default is 10**10.
    nu_max : float, optional
        Maximum frequency for integrating the bolometric luminosity. The default is 10**16.5.

    Returns
    -------
    lum : float
        The bolometric luminosity of the disk in erg/s integrated over radius and frequency.
    bol_lum : float
        The bolometric luminosity of the disk in erg/s computed as eta * mdot* c^2 [eta 0.083, i.e. Rin = 3Rs]
    eta : float
        The efficiency of the disk, defined as the ratio of the bolometric luminosity to the Eddington luminosity.

    """
    m = 10**m * M_sun

    m_dot =  10**m_dot*l_edd(m)/(c**2)


    rs = r_swarz(m)
    r_in *= rs
    r_fin *= rs

    integrand2 = lambda r,freq : integrand(r,freq,m,m_dot,r_in)

    integrand3 = lambda freq : quad(integrand2,r_in,r_fin,args=(freq))[0]

    lumbol = quad(integrand3,nu_min,nu_max)

    lumx = 0.083*m_dot*c*c
    eta = lumbol[0]/m_dot/c/c

    return lumbol[0],lumx,eta




if __name__=='__main__':


    freq = 10**np.linspace(12,16,150)
    
    m = np.log10(1e9)
    m_dot =  np.log10(0.1)

    lum1  = disc_luminosity(freq,8,np.log10(1),0)
    lum2  = disc_luminosity(freq,9,np.log10(0.1),0)
    lum3  = disc_luminosity(freq,10,np.log10(0.01),0)
    lum5  = disc_luminosity(freq,11,np.log10(0.001),0)
    lum4  = disc_luminosity(freq,np.log10(3e9),np.log10(0.1/3),0)

    y = np.linspace(40,45,150)

    fig = plt.figure()
    ax  = fig.add_subplot()
    ax.plot(np.log10(freq),lum1,'r')
    ax.plot(np.log10(freq),lum2,'b')
    ax.plot(np.log10(freq),lum3,'k')
    ax.plot(np.log10(freq),lum4,'m')
    ax.plot(np.log10(freq),lum5,'c')
    #ax.plot(np.log10(5.88e14)*np.ones_like(y),y,'c')

    plt.ylim(40,46)
    plt.show()


    lum  = disc_luminosity(freq,m,m_dot,0)
    lum_bol,bol_lum,eta  = disc_bol_luminosity(m,m_dot)

    