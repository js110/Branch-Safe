#Packages
from pyemd import emd
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import time
import pandas as pd 
import csv


from privic_functions import X_paris_lat, X_paris_long, X_paris_lat_wide, X_paris_long_wide, X_paris, X_paris_wide, N_X_paris_lat, N_X_paris_long, X_paris_possible, N_X_paris, p0_paris, C0_paris, CLap_paris
from privic_functions import X_sf_lat, X_sf_long, X_sf_lat_wide, X_sf_long_wide, X_sf, X_sf_wide, N_X_sf_lat, N_X_sf_long, X_sf_possible, N_X_sf, p0_sf, C0_sf, CLap_sf
from privic_functions import ‎eps_tight_paris, eps_tight_sf‎
from privic_functions import ‎PosToValParis, ValToPosParis, PosToValSF, ValToPosSF, BlahutArimotoParis, BlahutArimotoSF, obfuscate_paris, obfuscate_sf, IBU, LaplaceBetter, KWdist



################# START PARIS

#Data is sampled following a certain fraction every round
frac=0.1 #Fraction of data sampled each round
beta=1 #Slope parameter for Blahut-Arimoto
R_paris=15 #Number of rounds of data collection for Paris dataset
collection_paris=np.array([]) #Cumulative collection of all Paris data points collected till a certain step 
size_paris=int(N_paris*frac) #Collection size from Paris 
index_paris=list(range(len(paris_loc)))
##Big cycle for Paris
#Estimate the original distribution given the noisy data with respect to the optimal mechanism
    ##Inputs:
        #B = rate-distortion parameter (high B implies low distortion or high MI= low privacy/high QoS) for BA
        #R_BA = Number of iterations of BA
        #R_IBU = Number of iterationss' of IBU
    ##Returns :
        #Approximate original distribution of Paris locations
Dyn_est_paris=[] #Estimated PDF at each round
KW_paris=[] #K-W distance bw the estimated and the real PDFs at each round
collected_dist_paris=[] #PDF of real collected data at each round
for c in range(R_paris): #Collection round
    collected=np.random.choice(np.array(index_paris),size_paris, replace=False) #Indices of Paris locations collected in this round
    for i in collected:
        index_paris.remove(i) #Removing the already collected elements
    collection_paris=np.append(collection_paris,[paris_loc[collected]])#Updating the list of collected indices
    collected_values_paris=paris_loc[collected] #Locations collected in this round
    p_collected_paris=np.array([]) #Updated original PDF of the collected data
    for i in range(N_X_paris):
        count_x=0
        x=np.array(X_paris[i])
        check=(collected_values_paris==x) #matching the values of the space of locations with the collected dataset
        for tv in (check):
            count_x=count_x+(np.all(tv))
        p_collected_paris=np.append(p_collected_paris,[count_x/(size_paris)]) #Empirical PDF of noisy locations

    #First round of data collection
    if c==0:
        #Laplace obfuscation
        obs_paris=obfuscate_paris(X_samp=collected_values_paris, C=CLap_paris, Y=X_paris) #Collected data obfuscated with Laplace mechanism in the first round
        #Appriximated original PDF 
        #Iterative Bayesian Update
        p_init_paris=IBU(p0=np.array([1/N_X_paris]*N_X_paris),q=obs_paris[1],C=CLap_paris,R_IBU=10,X=X_paris,Y=X_paris)[1]
        Dyn_est_paris.append(p_init_paris) 
        collected_dist_paris.append(p_collected_paris)
        #Distance between estimated and original PDF of the collected data
        KW_paris.append(KWdist(X=np.array(X_paris), Y=np.array(X_paris), a=p_collected_paris, b=p_init_paris)) #KW Distance
    else:
        pBA_paris=Dyn_est_paris[c-1] #Last approximated PDF
        #Blahut Arimito
        CBA_paris=BlahutArimotoParis(C0=C0_paris, p0=pBA_paris, B=beta, R_BA=8) #BA channel generated with last approximated PDF
        obs_paris=obfuscate_paris(X_samp=collected_values_paris, C=CBA_paris, Y=X_paris) #Data obfuscated with BA mechanism
        #Iterative Bayesian Update
        #Appriximated original PDF
        p_present=IBU(p0=np.array([1/(N_X_paris)]*(N_X_paris)),q=obs_paris[1],C=CBA_paris,R_IBU=10,X=X_paris,Y=X_paris)[1]
        Dyn_est_paris.append(p_present)
        collected_dist_paris.append(p_collected_paris)
        #Distance between estimated and original PDF of the collected data
        KW_paris.append(KWdist(X=np.array(X_paris), Y=np.array(X_paris), a=p_collected_paris, b=p_present)) #KW distance
        
paris_final_collected=np.reshape(collection_paris,(size_paris*(c+1),2))

###################### END OF PARIS


######################  START SAN FRANCISCO 

#Data is sampled following a certain fraction every round
frac=0.1 #Fraction of data sampled each round
beta=1 #Slope parameter for Blahut-Arimoto
R_sf=8 #Number of rounds of data collection for SF dataset
collection_sf=np.array([]) #Cumulative collection of all SF data points collected till a certain step 
size_sf=int(N_sf*frac) #Collection size from SF
index_sf=list(range(len(sf_loc)))
##Big cycle for SF
#Estimate the original distribution given the noisy data with respect to the optimal mechanism
    ##Inputs:
        #B = rate-distortion parameter (high B implies low distortion or high MI= low privacy/high QoS) for BA
        #R_BA = Number of iterations of BA
        #R_IBU = Number of iterationss' of IBU
    ##Returns :
        #Approximate original distribution of SF locations
Dyn_est_sf=[] #Estimated PDF at each round
KW_sf=[] #K-W distance bw the estimated and the real PDFs at each round
collected_dist_sf=[] #PDF of real collected data at each round
for c in range(R_sf): #Collection round
    collected=np.random.choice(np.array(index_sf),size_sf, replace=False) #Indices of SF locations collected in this round
    for i in collected:
        index_sf.remove(i) #Removing the already collected elements
    collection_sf=np.append(collection_sf,[sf_loc[collected]])#Updating the list of collected locations
    collected_values_sf=sf_loc[collected] #Locations collected in this round
    p_collected_sf=np.array([]) #Updated original PDF of the collected data
    for i in range(N_X_sf):
        count_x=0
        x=np.array(X_sf[i])
        check=(collected_values_sf==x) #matching the values of the space of locations with the collected dataset
        for tv in (check):
            count_x=count_x+(np.all(tv))
        p_collected_sf=np.append(p_collected_sf,[count_x/(size_sf)]) #Empirical PDF of noisy locations

    #First round of data collection
    if c==0:
        #Laplace mechanism
        obs_sf=obfuscate_sf(X_samp=collected_values_sf, C=CLap_sf, Y=X_sf) #Data obfuscated with Laplace mechanism
        #Appriximated original PDF 
        #Iterative Bayesian Update
        p_init_sf =IBU(p0=np.array([1/N_X_sf]*N_X_sf),q=obs_sf[1],C=CLap_sf,R_IBU=5,X=X_sf,Y=X_sf)[1]
        Dyn_est_sf.append(p_init_sf)
        collected_dist_sf.append(p_collected_sf)
        KW_sf.append(KWdist(X=np.array(X_sf), Y=np.array(X_sf), a=p_collected_sf, b=p_init_sf)) #KW Distance
    else:
        pBA_sf=Dyn_est_sf[c-1] #Last approximated PDF
        #Blahut Arimito
        CBA_sf=BlahutArimotoSF(C0=C0_sf, p0=pBA_sf, B=beta, R_BA=5) #BA channel generated with last approximated PDF
        obs_sf=obfuscate_sf(X_samp=collected_values_sf, C=CBA_sf, Y=X_sf) #Data obfuscated with BA mechanism
        #Iterative Bayesian Update
        p_present=IBU(p0=np.array([1/(N_X_sf)]*(N_X_sf)),q=obs_sf[1],C=CBA_sf,R_IBU=5,X=X_sf,Y=X_sf)[1]#Approximated original PDF
        Dyn_est_sf.append(p_present)
        collected_dist_sf.append(p_collected_sf)
        KW_sf.append(KWdist(X=np.array(X_sf), Y=np.array(X_sf), a=p_collected_sf, b=p_present))

sf_final_collected=np.reshape(collection_sf,(size_sf*(c+1),2))
        
#############################END OF SAN FRANCISCO
