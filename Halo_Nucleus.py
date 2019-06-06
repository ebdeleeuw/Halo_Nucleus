import matplotlib.pyplot as plt
import numpy as np
import math as m
import cmath as cm
import abc
import mpmath as mp

class Potential(abc.ABC):
    """Abstract base class for subclassing potential types from"""
    
    @abc.abstractmethod
    def evaluate(self,r):
        """ Evaluate's the potential at position r
        Returns
        _______
        potential at position r: float, units of MeV
            Potential as evaluated at position r
        """

    def plot(self,r,ax):
        """ Plot the potential for a NumPy array r
        Parameters
        __________
        r: ndarray
            Array containing the radial values for plotting the potential
        ax: axis object from matplotlib.pyplot.subplots()
            Axis object to plot the potential to 

        """
        if not isinstance(r,np.ndarray):
            raise TypeError('Expected a NumPy array')
        lns=ax.plot(r,self.evaluate(r),'-k',label='Potential')
        return lns

class WoodsSaxon(Potential):
    """Inherits from Potential for defining a Woods Saxon potential for the core"""
    
    def __init__(self,V0,A,aws):
        """Initializes variables needed for the form of the potential
        Parameters
        __________
        V0: float, units in MeV
            Depth parameter of the Woods Saxon potential
        A: int
            Number of nucleons in the core nucleus
        aws: float, units in fm
            Diffusness parameter of the Woods Saxon potential
        
        """
        self.V0=V0   
        self.A=A
        self.aws=aws
        
    def evaluate(self,r):
        Rws=1.2*self.A**(1.0/3.0) #units in fm
        return self.V0/(1+np.e**((r-Rws)/self.aws))
    
class HaloInteraction:
    """Uses Runge Kutta 4 algorithm to evaluate specified scatering states
       Uses S and R matricies to calculate the phase shifts of each scattering state"""
    
    def __init__(self,potential_type,mu):
        """Instantiates a potential for the core nucleus and defines the halo mass
        Parameters
        __________
        potential_type: class of Potential type 
            Used as the potential for the core nucleus
        mu: float
            Mass (in MeV/c^2) of the halo
        
        """
        self.Const = 2*mu / 197.32705**2 #2mu/hbar**2 hbar*c = 197.32705 MeV fm
        self.potential=potential_type

    def hplus(self,L,r,E,prime=0):
        """Calculates the Hankel Function H+ in terms of the regular (coulombf) and irregular (coulombg) Coulomb functions: H(+)=G+iF

        Parameters
        __________
        L: int
            Angular momentum of the scattering state
        r: float
            Position of the boundry match condition
        E: int or float
            Energy of the scattering wavefunction
        prime: int 0 or 1
            Whether to calculate the derivative
        Returns
        _______
        H+: float
            The Hankel function H+ or its derivative
        
        """


        if prime==0:
            return mp.coulombg(L,0,cm.sqrt(self.Const*E)*r) +  complex(0,1)*mp.coulombf(L,0,cm.sqrt(self.Const*E)*r)
        elif prime==1:
            return mp.diff(lambda x: mp.coulombg(L,0,cm.sqrt(self.Const*E)*x),r) +  complex(0,1)*mp.diff(lambda x: mp.coulombf(L,0,cm.sqrt(self.Const*E)*x),r)

    def hminus(self,L,r,E,prime=0):
        """Calculates the Hankel Function H- in terms of the regular (coulombf) and irregular (coulombg) Coulomb functions: H(-)=G-iF

        Parameters
        __________
        L: int
            Angular momentum of the scattering state
        r: float
            Position of the boundry match condition
        E: int or float
            Energy of the scattering wavefunction
        prime: int 0 or 1
            Whether to calculate the derivative
        Returns
        _______
        H-: float
            The Hankel function H- or its derivative
        
        """

        if prime==0:
            return mp.coulombg(L,0,cm.sqrt(self.Const*E)*r) -  complex(0,1)*mp.coulombf(L,0,cm.sqrt(self.Const*E)*r)
        elif prime==1:
            return mp.diff(lambda x: mp.coulombg(L,0,cm.sqrt(self.Const*E)*x),r) -  complex(0,1)*mp.diff(lambda x: mp.coulombf(L,0,cm.sqrt(self.Const*E)*x),r)

    def r_matrix(self,boundry_match):
        """Calculates the R Matrix required for boundry matching

        Parameters
        __________
        boundry_match: int
            Index at which boundry matching occurs (currently calculated by taking 80% of the max radial distance)

        Returns
        _______
        R: float
           R Matrix

        """
        
        R=1/self.r[boundry_match]*(self.u[boundry_match]/self.z[boundry_match])
        return R

    def s_matrix(self,boundry_match,R,E,L):
        """Calculates the S Matrix required for calculating the phase shift
           S = [H(-) - a*R*H(-)] / [H(+) - a*R*H(+)]  where (a) is the radius at which boundry matching occurs

        Parameters
        __________
        boundry_match: int
            Index at which boundry matching occurs (currently calculated by taking 80% of the max radial distance)
        R: float
            R Matrix calculated from r_matrix()
        E: int or float
            Energy of the scattering wavefunction
        L: int
            Angular momentum of the scattering state

        Returns
        _______
        S: float
           S Matrix

        """

        S=(self.hminus(L,self.r[boundry_match],E) - self.r[boundry_match]*R*self.hminus(L,self.r[boundry_match],E,prime=1)) / (self.hplus(L,self.r[boundry_match],E) - self.r[boundry_match]*R*self.hplus(L,self.r[boundry_match],E,prime=1))
        return S
    
    def phase_shift(self,boundry_match,R,E,L):
        """Calculates the phase shift of the scattering state at energy E

        Parameters
        __________
        boundry_match: int
            Index at which boundry matching occurs (currently calculated by taking 80% of the max radial distance)
        R: float
            R Matrix calculated from r_matrix()
        E: int or float
            Energy of the scattering wavefunction
        L: int
            Angular momentum of the scattering state

        Returns
        _______
        phase: float
            Phase shift

        """
        phase=(cm.log(self.s_matrix(boundry_match,R,E,L))).imag /2 *180/np.pi
        return phase
        
    def radial_eq(self,E,L,R,wave_function):
        """Radial scattering equation

        Parameters
        __________
        E: int or float
            Energy of the scattering wavefunction
        L: int
            Angular momentum of the scattering state
        R: float
            Radius from the origin at which to evaluate the scattering equation
        wavefunction: float
            Current value of the scattering wavefunction

        Returns
        _______
        radial equation: float
        
        """


        return (L*(L+1)/(R)**2 + self.Const*(self.potential.evaluate(R)-E))*wave_function

    def RK4(self,radius,E,L,ini_derivative):
        """Runge Kutta Algorithm
        Splits 2nd order ODE into dy/dr=z and dz/dr = radial_eq()

        Parameters
        __________
        radius: tuple of len 2
            start and end radial positions to calculate the wavefunction
        E: int or float
            Energy of the scattering wavefunction
        L: int
            Angular momentum of the scattering state
        ini_derivative: float
            Initial derivative (known or guessed) at which to start the Runge Kutta alogrithm

        """

        step=0.1
        ARRAY_SIZE=int((radius[1]-radius[0])/0.1)
        
        #weights for slopes: (i.e. l[0]+2*l[1]+2*l[2]+l[3])
        weights=np.array([1,2,2,1])
        
        #k0,k1,k2,k3 needed for calculation of y
        k=np.zeros(4)
        
        #l0,l1,l2,l3 needed for calculation of z
        l=np.zeros(4)
        
        self.r=np.arange(*radius,step)  
        self.u=np.zeros(ARRAY_SIZE)     
        self.z=np.zeros(ARRAY_SIZE)     

        #initial conditions
        self.u[0],self.z[0]=(0,ini_derivative)

        #Runge-Kutta routine
        for i in range(self.r.size-1):
            for j in range(4):
                k[0]=(self.z[i])
                if i==0:
                    l[0]=(self.u[i])
                else:
                    l[0]=(self.radial_eq(E,L,self.r[i],self.u[i]))
                if j==1 or j==2:
                    k[j]=(self.z[i]+step/2*l[-1])
                    l[j]=(self.radial_eq(E,L,self.r[i]+step/2,self.u[i]+step/2*k[-2]))

                k[3]=(self.z[i]+step*l[-1])
                l[3]=(self.radial_eq(E,L,self.r[i]+step,self.u[i]+step*k[-2]))

            self.u[i+1]=(self.u[i]+step/6*np.dot(k,weights))
            self.z[i+1]=(self.z[i]+step/6*np.dot(l,weights))

    def plot_wf(self,E,L,ini_d,radius=(0,100),color=['-b'],label=None):
        """Plots the radial behavior of the scattering wavefunctions

        Parameters
        __________
        E: itterable 
            Energy of the scattering wavefunction
        L: itterable 
            Angular momentum of the scattering state
        ini_derivative: float
            Initial derivative (known or guessed) at which to start the Runge Kutta alogrithm
        radius: tuple of len 2, optional
            Start and end radial positions to calculate the wavefunction
        color: itterable, optional
            Color of plotted lines
        label: itterable, optional
            Label of plotted lines

        """
        plt.style.use('ggplot')
        fig1,ax1=plt.subplots()
        ax2=ax1.twinx()
        for i,j in enumerate(E):
            self.RK4(radius,j,L[i],ini_d[i])
            if i==0:
                #lns stores label names for later use
                lns=ax1.plot(self.r,self.u,color[i],label=label[i])
            else:
                lns+=ax1.plot(self.r,self.u,color[i],label=label[i])
        lns+=self.potential.plot(self.r,ax2)
        labs=[l.get_label() for l in lns]
        ax1.legend(lns,labs,loc="lower right",fancybox=True)
        ax1.set(xlim=(self.r.min(),self.r.max()), xlabel="R (fm)", ylabel="u(x)")
        ax2.set(ylabel="MeV")

    def plot_phases(self,E,L,ini_d,radius=(0,100),color=['-b'],label=None):
        """Plots the phase shifts of the scattering wavefunctions and regularizes them.

        Parameters
        __________
        E: itterable 
            Energy of the scattering wavefunction
        L: int
            Angular momentum of the scattering state
        ini_derivative: float
            Initial derivative (known or guessed) at which to start the Runge Kutta alogrithm
        radius: tuple of len 2, optional
            Start and end radial positions to calculate the wavefunction
        color: itterable, optional
            Color of plotted lines
        label: itterable, optional
            Label of plotted lines

        """
        plt.style.use('ggplot')
        fig1,ax1=plt.subplots()

        phase=np.zeros(len(E))
        
        
        for i,j in enumerate(E):
            self.RK4(radius,j,L,ini_d)
            if i==0:
                #boundry match at 80% of the endpoint
                boundry_match=int(self.r.size*0.8) 
            phase[i]=self.phase_shift(boundry_match,self.r_matrix(boundry_match),j,L)
        ax1.plot(E,phase,color[0],label=label[0])


if __name__ == "__main__":    

    #Woods Saxon potential parameters for 11Be
    V0=-61.1 #MeV
    A=10
    aws=0.65 #fm

    energies=[0.1]*3
    L_values=range(3)
    initial_derivatives=[2,0.05,0.00005]

    #energies=np.arange(0.1,3,0.1)
    #L=1
    #ini_d=1
    
    mu=931.494 #MeV/c^2 (1amu why not 1 neutron mass??)
    Be_11=HaloInteraction(WoodsSaxon(V0,A,aws),mu)
    
    Be_11.plot_wf(energies,L_values,initial_derivatives,color=['-b','-r','-g'],label=['L=0','L=1','L=2'])
    #Be_11.plot_phases(energies,L,ini_d,label=["L=0"])
    plt.show()


