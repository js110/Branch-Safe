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



#Importing and setting up
import pandas as pd 

#Paris dataset
gowalla_paris_all = pd.read_csv (r'Gowalla_Paris_grid.csv') #Importing Gowalla Paris (grid aggregated) dataset
gowalla_paris_values=gowalla_paris_all.values #Converting gowalla dataset to an array
N_paris=len(gowalla_paris_values) #Number of data points in Paris dataset

paris_loc=np.rint(gowalla_paris_values) #Rounding off latitudes and longitudes
paris_latitudes=np.rint(gowalla_paris_values[:,0]) #Latitudes rounded to nearest integer78
paris_longitudes=np.rint(gowalla_paris_values[:,1]) #Longitudes rounded to nearest integer

#Range of Paris dataset
paris_lat_max=np.max(paris_latitudes) #Maximum latitude
paris_lat_min=np.min(paris_latitudes) #Minimum latitude
paris_long_max=np.max(paris_longitudes) #Maximum longitude
paris_long_min=np.min(paris_longitudes) #Minimum longitude
paris_lat_range=np.array([paris_lat_min,paris_lat_max+1]) #Range of latitude
paris_long_range=np.array([paris_long_min,paris_long_max+1]) #Range of longitude

#San Francisco dataset
gowalla_sf_all = pd.read_csv (r'Gowalla_SF_grid.csv') #Importing Gowalla SF (grid aggregated) dataset
gowalla_sf_values=gowalla_sf_all.values #Converting gowalla dataset to an array
N_sf=len(gowalla_sf_values) #Number of data points in SF dataset

sf_loc=np.rint(gowalla_sf_values) #Rounding off latitudes and longitudes
sf_latitudes=np.rint(gowalla_sf_values[:,0]) #Latitudes rounded to nearest integer78
sf_longitudes=np.rint(gowalla_sf_values[:,1]) #Longitudes rounded to nearest integer

#Range of SF dataset
sf_lat_max=np.max(sf_latitudes) #Maximum latitude
sf_lat_min=np.min(sf_latitudes) #Minimum latitude
sf_long_max=np.max(sf_longitudes) #Maximum longitude
sf_long_min=np.min(sf_longitudes) #Minimum longitude
sf_lat_range=np.array([sf_lat_min,sf_lat_max+1]) #Range of latitude
sf_long_range=np.array([sf_long_min,sf_long_max+1]) #Range of longitude


#Setup for Paris experiment

X_paris_lat=np.array(range(int(paris_lat_min),int(paris_lat_max+1)))    #Space of latitudes of original data in Paris dataset
X_paris_long=np.array(range(int(paris_long_min),int(paris_long_max+1))) #Space of longitudes of original data in Paris dataset
Y_paris_lat=np.array(range(int(paris_lat_min),int(paris_lat_max+1)))    #Space of latitudes of sanitized data in Paris dataset
Y_paris_long=np.array(range(int(paris_long_min),int(paris_long_max+1))) #Space of longitudes of sanitized data in Paris dataset
N_X_paris_lat=len(X_paris_lat) #Total length of the space of source latitudes of Paris dataset
N_X_paris_long=len(X_paris_long) #Total length of the space of source longitudes of Paris dataset
from itertools import product
X_paris=list(product(X_paris_lat, X_paris_long)) #all possible inpput pairs of (lat,long) of Paris datset
Y_paris=list(product(Y_paris_lat, Y_paris_long)) #all possible output pairs of (lat,long) of Paris dataset

N_X_paris=len(X_paris) #Total length of the space of Paris data

#Entire domain for Paris locations
X_paris_possible=len(X_paris_lat)*len(X_paris_long) #Total possible input locations of Paris dataset
Y_paris_possible=len(Y_paris_lat)*len(Y_paris_long) #Total possible output locations of Paris dataset

#Setup for SF experiment

X_sf_lat=np.array(range(int(sf_lat_min),int(sf_lat_max+1)))    #Space of latitudes of original data in SF dataset
X_sf_long=np.array(range(int(sf_long_min),int(sf_long_max+1))) #Space of longitudes of original data in SF dataset
Y_sf_lat=np.array(range(int(sf_lat_min),int(sf_lat_max+1)))    #Space of latitudes of sanitized data in SF dataset
Y_sf_long=np.array(range(int(sf_long_min),int(sf_long_max+1))) #Space of longitudes of sanitized data in SF dataset
N_X_sf_lat=len(X_sf_lat) #Total length of the space of source latitudes of SF dataset
N_X_sf_long=len(X_sf_long) #Total length of the space of source longitudes of SF dataset
from itertools import product
X_sf=list(product(X_sf_lat, X_sf_long)) #all possible inpput pairs of (lat,long) of SF datset
Y_sf=list(product(Y_sf_lat, Y_sf_long)) #all possible output pairs of (lat,long) of SF dataset


N_X_sf=len(X_sf) #Total length of the space of SF data

#Dummy channel to test the mechanisms on SF dataset
X_sf_possible=len(X_sf_lat)*len(X_sf_long) #Total possible input locations of SF dataset
Y_sf_possible=len(Y_sf_lat)*len(Y_sf_long) #Total possible output locations of SF dataset

########################################################
###Expanding the co-domain (to implement Laplace)
#Paris
X_paris_lat_wide=np.array(range(int(paris_lat_min)-5,int(paris_lat_max+1)+5))    #Wider space of latitudes of original data in Paris dataset
X_paris_long_wide=np.array(range(int(paris_long_min)-5,int(paris_long_max+1)+5)) #Wider space of longitudes of original data in Paris dataset
X_paris_wide=list(product(X_paris_lat_wide, X_paris_long_wide)) #Wider space of all possible inpput pairs of (lat,long) of SF datset

#SF
X_sf_lat_wide=np.array(range(int(sf_lat_min)-5,int(sf_lat_max+1)+5))    #Wider space of latitudes of original data in SF dataset
X_sf_long_wide=np.array(range(int(sf_long_min)-5,int(sf_long_max+1)+5)) #Wider space of longitudes of original data in SF dataset
X_sf_wide=list(product(X_sf_lat_wide, X_sf_long_wide)) #Wider space of all possible inpput pairs of (lat,long) of SF datset

########################################################

#Assigning each (latitude,longitude) pair it's positional index
#Converting between 1D pos to 2D val 
#FOR PARIS
def PosToValParis(t,XLat=X_paris_lat,XLong=X_paris_long):
    n=len(XLong)
    #First giving positions of corresponding latitude, longitude
    pos_lat=int((t-(t%n))/n)
    pos_long=int(t%n)
    #Note: (pos_lat,pos_long) is NOT (lat,long)
    lat=int(XLat[pos_lat])
    long=int(XLong[pos_long])
    return lat, long

def ValToPosParis(a,b,XLong=X_paris_long):
    n=len(XLong)
    return (n*a+b)


#FOR SF
def PosToValSF(t,XLat=X_sf_lat,XLong=X_sf_long):
    n=len(XLong)
    #First giving positions of corresponding latitude, longitude
    pos_lat=int((t-(t%n))/n)
    pos_long=int(t%n)
    #Note: (pos_lat,pos_long) is NOT (lat,long)
    lat=int(XLat[pos_lat])
    long=int(XLong[pos_long])
    return lat, long

def ValToPosSF(a,b,XLong=X_sf_long):
    n=len(XLong)
    return (n*a+b)

#######################################################################
#Blahut-Arimoto Algorithm
def BlahutArimotoParis(C0,p0,B,R_BA,Num_X=N_X_paris,Num_Y=N_X_paris):
    ##Inputs:
        #C0=Starting Channel 
        #p0=PDF of source data
        #B=rate-distortion parameter (high values of B implies low avg distortion=QoS and high MI=privacy)
        #R_BA=Number of iterations 
        #Num_X=length of source alphabet
        #Num_Y=length of image alphabet
    ##Output:
        #Returns a channel that computes the minimum MI for a given average distortion

    start_time = time.time()
    #Intermediate storage
    Channels=[] #Storing all the channels till convergence 
    Channels.append(C0) #The first element is the initial matrix

    exp_mat=np.zeros((Num_X,Num_Y)) #Exponential matrix pre-computed 
    for x in range(Num_X):
        x_val=np.array([PosToValParis(x)[0],PosToValParis(x)[1]]) #(lat,long) value at index x
        for y in range(Num_Y):
            y_val=np.array([PosToValParis(y)[0],PosToValParis(y)[1]]) #(lat,long) value at index y
            exp_mat[x,y]=np.exp(-B*(np.linalg.norm(x_val-y_val)))

    for i in range(R_BA): #starting the iteration of BA algorithm
        #print("BA iteration:", i)
        c=np.matmul(p0,Channels[i]) #Computing the output probabilities
        C_next=np.zeros((Num_X,Num_Y)) #Creating a dummy matrix at each step
        for x in range(Num_X): #Rows of the channel
            #print("BA Channel row: ",x)
            x_val=np.array([PosToValParis(x)[0],PosToValParis(x)[1]]) #(lat,long) value at index x
            for y in range(Num_Y): #Columns of the channel
                #print("BA Channel column: ",y)
                y_val=np.array([PosToValParis(y)[0],PosToValParis(y)[1]]) #(lat,long) value at index y
                C_next_num=(c[y])*(np.exp(-B*(np.linalg.norm(x_val-y_val)))) #Numerator of the next channel
                C_next_denom=np.dot(c,(exp_mat)[x,:]) #Denominator of the next channel
                C_next[x,y]=(C_next_num)/(C_next_denom) #Next channel
        Channels.append(C_next)
    print("Run time of BA for Paris: %s seconds ---" % (time.time() - start_time))
    return(Channels[R_BA])



#######################################################################

#Blahut-Arimoto Algorithm
def BlahutArimotoSF(C0,p0,B,R_BA,Num_X=N_X_sf,Num_Y=N_X_sf):
    ##Inputs:
        #C0=Starting Channel 
        #p0=PDF of source data
        #B=rate-distortion parameter (high values of B implies low avg distortion=QoS and high MI=privacy)
        #R_BA=Number of iterations 
        #Num_X=length of source alphabet
        #Num_Y=length of image alphabet
    ##Output:
        #Returns a channel that computes the minimum MI for a given average distortion

    start_time = time.time()
    #Intermediate storage
    Channels=[] #Storing all the channels till convergence 
    Channels.append(C0) #The first element is the initial matrix

    exp_mat=np.zeros((Num_X,Num_Y)) #Exponential matrix pre-computed 
    for x in range(Num_X):
        x_val=np.array([PosToValSF(x)[0],PosToValSF(x)[1]]) #(lat,long) value at index x
        for y in range(Num_Y):
            y_val=np.array([PosToValSF(y)[0],PosToValSF(y)[1]]) #(lat,long) value at index y
            exp_mat[x,y]=np.exp(-B*(np.linalg.norm(x_val-y_val)))

    for i in range(R_BA): #starting the iteration of BA algorithm
        #print("BA iteration:", i)
        c=np.matmul(p0,Channels[i]) #Computing the output probabilities
        C_next=np.zeros((Num_X,Num_Y)) #Creating a dummy matrix at each step
        for x in range(Num_X): #Rows of the channel
            #print("BA Channel row: ",x)
            x_val=np.array([PosToValSF(x)[0],PosToValSF(x)[1]]) #(lat,long) value at index x
            for y in range(Num_Y): #Columns of the channel
                #print("BA Channel column: ",y)
                y_val=np.array([PosToValSF(y)[0],PosToValSF(y)[1]]) #(lat,long) value at index y
                C_next_num=(c[y])*(np.exp(-B*(np.linalg.norm(x_val-y_val)))) #Numerator of the next channel
                C_next_denom=np.dot(c,(exp_mat)[x,:]) #Denominator of the next channel
                C_next[x,y]=(C_next_num)/(C_next_denom) #Next channel
        Channels.append(C_next)
    print("Run time of BA for SF: %s seconds ---" % (time.time() - start_time))
    return(Channels[R_BA])


###################################################################
#Adding noise to data
#PARIS
def obfuscate_paris(X_samp,C,Y,XLat=X_paris_lat,XLong=X_paris_long,YLat=X_paris_lat,YLong=X_paris_long): #Sanitization of data
    ##Inputs :
        #XLat
        #X_samp = sample of original data
        #C = privacy channel used
        #Y = Codomain of the mechanism/Space of noisy data
    Y_samp_paris=[]#Noisy samples
    for pos_x in range(len(X_samp)):
        x=X_samp[pos_x] #Getting the location data
        lat_x=int(x[0]) #Latitude component of x
        long_x=int(x[1]) #Longitude component of x
        #Finding the position of latitude of the data in the source alphabet for latitudes
        pos_lat_x=int(np.where(X_paris_lat==lat_x)[0][0]) 
        #Finding the position of longitude of the data in the source alphabet for longitudes
        pos_long_x=int(np.where(X_paris_long==long_x)[0][0]) 
        pos_x_1D=ValToPosParis(pos_lat_x,pos_long_x,XLong) #Linear index of the (lat,long) pair #!!!!!!!!! NEEDS TO BE CHAMGED FOR PARIS AND SF DATASETS
        #Adding noise
        prob_x=C[pos_x_1D,:] #x is sanitized wrt the PDF of x'th (in linear index) row of the channel 
        #Note Y is a m*n x 2 array, each row having a (lat,long) pair
        y_1D=int(random.choices(list(range(len(Y))),prob_x,k=1)[0]) #!!!!!!!!! NEEDS TO BE CHAMGED FOR PARIS AND SF DATASETS
        #Extracting latitude, longitude components from y_1D
        y_lat=PosToValParis(y_1D,YLat,YLong)[0]
        y_long=PosToValParis(y_1D,YLat,YLong)[1]
        Y_samp_paris.append(np.array([y_lat,y_long]))
    Y_samp_paris=np.array(Y_samp_paris)
    
    #Computing empirical probabilities of the noisy data
    q_freq_paris=np.array([]) #Empirical probability of noisy_samp
    N_Y_paris=len(Y)
    for pos_y in range(N_Y_paris):
        count_y=0 #Count of element y in noisy locations
        y=Y[pos_y]
        print("Obfuscating: Progress {a} out of {b}".format(a=pos_y, b=N_Y_paris))
        Y_truth=(Y_samp_paris==y)
        for nv in (Y_truth):
            count_y=count_y+(np.all(nv))
        #empirical probability of y in noisy_samp
        q_freq_paris=np.append(q_freq_paris,[count_y/(len(Y_samp_paris))])
    return Y_samp_paris, q_freq_paris #Returning the noisy data (in 2D form) and their empirical probabilities 

#SF
def obfuscate_sf(X_samp,C,Y,XLat=X_sf_lat,XLong=X_sf_long,YLat=X_sf_lat,YLong=X_sf_long): #Sanitization of data
    ##Inputs :
        #X_samp = sample of original data
        #C = privacy channel used
        #Y = Codomain of the mechanism/Space of noisy data
    Y_samp_sf=[] #Noisy samples
    for pos_x in range(len(X_samp)):
        x=X_samp[pos_x] #Getting the location data
        lat_x=int(x[0]) #Latitude component of x
        long_x=int(x[1]) #Longitude component of x
        #Finding the position of latitude of the data in the source alphabet for latitudes
        pos_lat_x=int(np.where(XLat==lat_x)[0][0]) 
        #Finding the position of longitude of the data in the source alphabet for longitudes
        pos_long_x=int(np.where(XLong==long_x)[0][0]) 
        pos_x_1D=ValToPosSF(pos_lat_x,pos_long_x,XLong) #Linear index of the (lat,long) pair #!!!!!!!!! NEEDS TO BE CHAMGED FOR PARIS AND SF DATASETS
        #Adding noise
        prob_x=C[pos_x_1D,:] #x is sanitized wrt the PDF of x'th (in linear index) row of the channel 
        #Note Y is a m*n x 2 array, each row having a (lat,long) pair
        y_1D=int(random.choices(list(range(len(Y))),prob_x,k=1)[0]) #!!!!!!!!! NEEDS TO BE CHAMGED FOR PARIS AND SF DATASETS
        #Extracting latitude, longitude components from y_1D
        y_lat=PosToValSF(y_1D,YLat,YLong)[0]
        y_long=PosToValSF(y_1D,YLat,YLong)[1]
        Y_samp_sf.append(np.array([y_lat,y_long]))
    Y_samp_sf=np.array(Y_samp_sf)
    
    #Computing empirical probabilities of the noisy data
    q_freq_sf=np.array([]) #Empirical probability of noisy_samp
    N_Y_sf=len(Y)
    for pos_y in range(N_Y_sf):
        y=Y[pos_y]
        count_y=0 #Count of element y in noisy locations
        print("Obfuscating: Progress {a} out of {b}".format(a=pos_y, b=N_Y_sf))
        Y_truth=(Y_samp_sf==y)
        for nv in (Y_truth):
            count_y=count_y+(np.all(nv))
        #empirical probability of y in noisy_samp
        q_freq_sf=np.append(q_freq_sf,[count_y/(len(Y_samp_sf))])
    return Y_samp_sf, q_freq_sf #Returning the noisy data (in 2D form) and their empirical probabilities 


##################################################################

#Iterative Bayesian Update
def IBU(p0,q,C,R_IBU,X,Y): #Estimating original distribution
    ##Inputs :
        #p0 = initial guess for the original PDF (full support)
        #q = the empirical PDF of the observed (noisy) data
        #C = the privacy mechanism used
        #R_IBU = number of iterationss' between consequtive iteration
        #X = domain of original data
        #Y = domain of noisy data
    ##Output :
        #Estimate of the original PDF
    start_time = time.time()
    Num_X=len(X)
    Num_Y=len(Y)
    PDFs=[p0] #List of estimated distributions
    for i in range(R_IBU):
        print ("IBU iteration:", i)
        p=PDFs[i]
        p_next=np.zeros(Num_X)
        for x in range(len(X)): #computing the x'th component of the estimated PDF
            sum_ibu=0
            for y in range(len(Y)):
                num=q[y]*C[x,y]*p[x]
                denom=0
                for z in range(len(X)):
                    denom+=(p[z]*C[z,y])
                sum_ibu=(sum_ibu)+(num/denom)
            p_next[x]=sum_ibu
        PDFs.append(p_next)
    print("Run time of IBU: %s seconds ---" % (time.time() - start_time))
    return PDFs, PDFs[R_IBU]
    
    
####################################################
#CREATING A LINEAR PDF FOR 2D DATA for Paris
p0_paris=np.array([]) #Empirical probability of paris data
N_X_paris=len(X_paris)
for pos_x in range(N_X_paris):
    x=X_paris[pos_x]
    print("Progress {a} out of {b}".format(a=pos_x, b=N_X_paris))
    freq_x=0 #empirical probability of x in Paris data
    for y in paris_loc:
        comp=(x==y)
        if np.all(comp): #Testing if both elements of x and y are the same
            freq_x+=1
    p0_paris=np.append(p0_paris,[freq_x])
p0_paris=p0_paris/(N_paris) #Normalising frequencies to give empirical probabilities

#CREATING A LINEAR PDF FOR 2D DATA for SF
p0_sf=np.array([]) #Empirical probability of paris data
N_X_sf=len(X_sf)
for pos_x in range(N_X_sf):
    x=X_sf[pos_x]
    print("Progress {a} out of {b}".format(a=pos_x, b=N_X_sf))
    freq_x=0 #empirical probability of x in Paris data
    for y in sf_loc:
        comp=(x==y)
        if np.all(comp): #Testing if both elements of x and y are the same
            freq_x+=1
    p0_sf=np.append(p0_sf,[freq_x])
p0_sf=p0_sf/(N_sf) #Normalising frequencies to give empirical probabilities

##############################################

#Reshaping and initualizing uniform channels
C0_paris=np.reshape(np.array([1/N_X_paris]*(N_X_paris**2)),(N_X_paris,N_X_paris)) #Initial channel to start BA for Paris
p0_paris_unif=np.array([1/N_X_paris]*N_X_paris) #Initial guess of original distribution of Paris
C0_sf=np.reshape(np.array([1/N_X_sf]*(N_X_sf**2)),(N_X_sf,N_X_sf)) #Initial channel to start BA for SF
p0_sf_unif=np.array([1/N_X_sf]*N_X_sf) #Initial guess of original distribution of SF


#######################################################
#Testing Blahut-Arimoto Channels
#For slope=infinity
beta=1 #Slope parameter: almost infinity

#BA channel for Paris
C_paris=BlahutArimotoParis(C0_paris,p0_paris_unif,B=beta,R_BA=10,Num_X=N_X_paris,Num_Y=N_X_paris)
#BA channel for SF
C_sf=BlahutArimotoSF(C0_sf,p0_sf_unif,B=beta,R_BA=10,Num_X=N_X_sf,Num_Y=N_X_sf)



 
#################################################################################
#Laplace mechanism for the first round of data collection

#Laplace mechanism

def LaplaceBetter(eps,X,Y):        #eps= privacy parameter
        #X= domain of original data (lat,long)
        #Y= domain of noisy data (lat,long)
    C=np.zeros((len(X),len(Y)))
    #float((eps**2)/(2*(np.pi))) #Normalising constant
    Normalizer=[] #Normalizing constant for each value of X
    for i in range(len(X)):
        n=0
        for j in range(len(Y)):
            n+=float(np.exp(-eps*np.linalg.norm(np.array(X[i])-np.array(Y[j]))))
        Normalizer.append(1/n)
    for i in range(len(X)):
        norm_const=Normalizer[i]
        for j in range(len(Y)):
            C[i,j]=float((norm_const)*np.exp(-eps*np.linalg.norm(np.array(X[i])-np.array(Y[j]))))
    return (C)


#####################################################################################

CLap_paris=LaplaceBetter(eps=2, X=X_paris, Y=X_paris) #Laplace mechanism for Paris locations 

CLap_sf=LaplaceBetter(eps=2, X=X_sf, Y=X_sf) #Laplace mechanism for SF locations 


##Evaluation metrics

##Kantorowich-Wasserstein Distance

def KWdist(X,Y,a,b):
    ##Inputs:
        #X = space of original data
        #Y = space of noisy data
        #a = original PDF on X
        #b = estimated PDF on Y
    dist=np.reshape(np.zeros(len(a)*len(b)),(len(a),len(b))) #Distance matrix between X and Y
    for i in range(len(a)):
        for j in range(len(b)):#Finding 2D (lat,long) from 1D index for estimated dist
            dist[i,j]=np.linalg.norm(X[i]-Y[j])
    return (emd(a,b,dist))

#Empirical tight epsilons for Paris 
eps_tight_paris=np.array([0.392818028068501,
0.7813925715019457,
1.1605111783813877,
1.522997813452586,
1.9201735805617113,
2.318468681063899,
2.7235710043448096,
3.131913318373869,
3.539017362258331,
3.946218013335979,
4.353319497725823,
4.7597883617851595,
5.1650376600730725,
5.568215644792336,
5.967803359717294,
6.3610873573101845,
6.741703436218012,
7.094334323347863,
7.398355371524807,
7.706459118837066,
8.109656883411978,
8.512103108219597,
8.914469010971294,
9.316911956393886,
9.719415985071155
])
#Empirical tight epsilons for SF
eps_tight_sf=np.array([0.3986617268975043,
0.7883813903959236,
1.1750263394594984,
1.5677798334718995,
1.9816354129563862,
2.3881286826627117,
2.7901762206493075,
3.18914410820109,
3.587243587425301,
3.986583121714903,
4.3873489652913085,
4.78887068465851,
5.1906269545833394,
5.59231245598343,
5.993791505639457,
6.39501755552611,
6.795978710824691,
7.196667510002639,
7.597048656480873,
7.997072913065153,
8.3968159232846,
8.796488394614757,
9.19621125393365,
9.595978444142562,
9.99579069328416
])

