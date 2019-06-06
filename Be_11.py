from Halo_Nucleus import *

if __name__ == "__main__":    

    #Woods Saxon potential parameters for 11Be
    V0=-61.1 #MeV
    A=10
    aws=0.65 #fm
    mu=931.494 #MeV/c^2 (1 amu)
    
    #Finds the scatering wavefunciton for L values of [0,1,2] at energies of 0.1 MeV. The initial derivatives are set so that the scales are similar.
    energies=[0.1]*3
    L_values=range(3)
    initial_derivatives=[2,0.05,0.00005]

    Be_11=HaloInteraction(WoodsSaxon(V0,A,aws),mu)
    Be_11.plot_wf(energies,L_values,initial_derivatives,color=['-b','-r','-g'],label=['L=0','L=1','L=2'])
        
    #Uncomment to plot the phase shifts for energies between 0.1 and 3 MeV for L=1
    #energies=np.arange(0.1,3,0.1)
    #L=1
    #ini_d=1
    

    #Be_11=HaloInteraction(WoodsSaxon(V0,A,aws),mu)
    #Be_11.plot_phases(energies,L,ini_d,label=["L=0"])

    plt.show()
