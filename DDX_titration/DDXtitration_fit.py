# -*- coding: utf-8 -*-
"""
# Fitting of 19F signal intensities to the binding model
# where DDX3X binds exclusively to ssRNA in the presence of the dsRNA-ssRNA equilibrium.

# numpy ver. 1.24.3
# matplotlib ver. 3.7.1
# pandas ver. 1.5.3
# lmfit ver. 1.2.2
"""

import numpy as np
from scipy import optimize as opt
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt

from lmfit import minimize, Parameters, fit_report

############################
## Import data
############################

data = pd.read_csv("input.csv")
LT = 100E-6 # Total RNA concentration

conc = np.array(data.conc*1E-6)
ds = np.array(data.Int_d)
ss = np.array(data.Int_s)
b = np.array(data.Int_b)


##############################
# Functions
##############################

def ffs_complex(q,p):
    
    # Unpack variables and constants
    # S: ssRNA, D: dsRNA, P: free DDX, PS: ssRNA complexed DDX
    # CT: total protein concentration, LT: total RNA concentration
    # K1: ssRNA binding affinity of DDX3X, K2: equilibrium constant for ssRNA hybridization
    
    S, D, P, PS = p # Variables
    CT, LT, K1, K2 = q # Constants
    
    # Equations have to be set up as equal to 0
    eq1 = -CT + P + PS        # Protein equation
    eq2 = -LT + 2*D + S + PS  # Ligand equation
    eq3 = K1*P*S - PS
    eq4 = K2*S**2 - D
    return [eq1, eq2, eq3, eq4]

# Function to calculate the molar concentration terms.
# conc needs to be an array.
def modelcalc(conc,LT,K1,K2):
    
    S = np.zeros(len(conc))
    D = np.zeros(len(conc))
    P = np.zeros(len(conc))
    PS = np.zeros(len(conc))
    
    for i in range(len(conc)):
        p = [1e-6,LT/2,conc[i],1e-6]
        q = [conc[i],LT,K1,K2]
        ffs_partial = partial(ffs_complex,q)
        # Solutions are ordered according to how the initial guess vector is arranged
        S[i],D[i],P[i],PS[i] = opt.root(ffs_partial,p,method='lm').x
    return S,D,P,PS

# Function to convert the molar concentration into the NMR signal intensity
def getmag(S,D,PS,Mag_d,Mag_s,Mag_b):
    LT =  2*D + S + PS 
    Int_d = Mag_d*2*D/LT
    Int_s = Mag_s*S/LT
    Int_b = Mag_b*PS/LT
    return Int_d, Int_s, Int_b
    

####################################
# Output data frame
####################################
   
col1 = ['K1','K2','Mag_d','Mag_s','Mag_b',
        'K1_std','K2_std','Mag_d_std','Mag_s_std','Mag_b_std']

params_df = pd.DataFrame(columns=col1)
plotname = "plot"


############################
## Fitting
############################

fit_params = Parameters()
fit_params.add('Mag_d',value=1E8,vary=True)
fit_params.add('Mag_s',expr='Mag_d')
fit_params.add('Mag_b',expr='Mag_d')

fit_params.add('K1',value=70000,vary=True)
fit_params.add('K2',value=24270,vary=True)

def objective(fit_params):
    
    Mag_d = fit_params['Mag_d']
    Mag_s = fit_params['Mag_s']
    Mag_b = fit_params['Mag_b']
    
    K1 = fit_params['K1']
    K2 = fit_params['K2']
    
    S = np.zeros(len(conc))
    D = np.zeros(len(conc))
    P = np.zeros(len(conc))
    PS = np.zeros(len(conc))
    
    S,D,P,PS = modelcalc(conc,LT,K1,K2)

    Int_d,Int_s,Int_b = getmag(S,D,PS,Mag_d,Mag_s,Mag_b)
    
    return np.concatenate([ds-Int_d,ss-Int_s,b-Int_b])

result = minimize(objective,fit_params,method="leastsq")   
print(fit_report(result))

with open("result.txt", 'w') as fh:
    fh.write(fit_report(result))

RMSD = np.sqrt(result.chisqr/result.ndata)

###############################
#  Calculate the best fit value
###############################

opt_params = result.params

opt_Mag_d = opt_params["Mag_d"].value 
opt_Mag_s = opt_params["Mag_s"].value 
opt_Mag_b = opt_params["Mag_b"].value 

opt_K1 = opt_params["K1"].value 
opt_K2 = opt_params["K2"].value 


Ssim,Dsim,Psim,PSsim = modelcalc(conc,LT,opt_K1,opt_K2)
Int_d_sim,Int_s_sim,Int_b_sim = getmag(Ssim,Dsim,PSsim,opt_Mag_d,opt_Mag_s,opt_Mag_b)


####################################
# Update the initial condition
####################################

fit_params = Parameters()
fit_params.add('Mag_d',value=opt_Mag_d,vary=True)
fit_params.add('Mag_s',expr='Mag_d')
fit_params.add('Mag_b',expr='Mag_d')

fit_params.add('K1',value=opt_K1,vary=True)
fit_params.add('K2',value=opt_K2,vary=True)


####################################
# Error analysis
####################################

cycle = 1000

MC_Mag_d = np.zeros(cycle) 
MC_Mag_s = np.zeros(cycle) 
MC_Mag_b = np.zeros(cycle) 

MC_K1 = np.zeros(cycle) 
MC_K2 = np.zeros(cycle) 

concsim=np.linspace(0,np.max(conc)*1.1,100)
MC_Id_sim = np.zeros([cycle,len(concsim)])
MC_Is_sim = np.zeros([cycle,len(concsim)])
MC_Ib_sim = np.zeros([cycle,len(concsim)])

for k in range(cycle):
    
    ds_sim = Int_d_sim  + np.random.normal(0,RMSD,len(conc))
    ss_sim = Int_s_sim  + np.random.normal(0,RMSD,len(conc))
    b_sim = Int_b_sim + np.random.normal(0,RMSD,len(conc))

    def MC(fit_params):
        
        Mag_d = fit_params['Mag_d']
        Mag_s = fit_params['Mag_s']
        Mag_b = fit_params['Mag_b']
        
        K1 = fit_params['K1']
        K2 = fit_params['K2']

        S = np.zeros(len(conc))
        D = np.zeros(len(conc))
        P = np.zeros(len(conc))
        PS = np.zeros(len(conc))
        
        S, D, P, PS = modelcalc(conc,LT,K1,K2)
    
        Int_d,Int_s,Int_b = getmag(S,D,PS,Mag_d,Mag_s,Mag_b)
        
        return np.concatenate([ds_sim-Int_d,ss_sim-Int_s,b_sim-Int_b])
    
    result_temp = minimize(MC,fit_params,method="leastsq")
    with open("result.txt", 'a') as fh:
        fh.write("\n\n#######################\n")
        fh.write("MC iterlation " + str(k))
        fh.write("\n#######################\n\n")
        fh.write(fit_report(result_temp))
    temp_params = result_temp.params
    
    MC_Mag_d[k] = temp_params["Mag_d"].value 
    MC_Mag_s[k] = temp_params["Mag_s"].value 
    MC_Mag_b[k] = temp_params["Mag_b"].value 

    MC_K1[k] = temp_params["K1"].value 
    MC_K2[k] = temp_params["K2"].value 
    
    temp_S,temp_D,temp_P,temp_PS = modelcalc(concsim,LT,MC_K1[k],MC_K2[k])
    
    MC_Id_sim[k],MC_Is_sim[k],MC_Ib_sim[k] = getmag(temp_S,temp_D,temp_PS,MC_Mag_d[k],MC_Mag_s[k],MC_Mag_b[k])
   
# Get smooth profiles for plotting

Ssim2,Dsim2,Psim2,PSsim2 = modelcalc(concsim,LT,opt_K1,opt_K2)
Int_d_sim2,Int_s_sim2,Int_b_sim2 = getmag(Ssim2,Dsim2,PSsim2,opt_Mag_d,opt_Mag_s,opt_Mag_b)

Int_s_top = Int_s_sim2 + np.std(MC_Is_sim, axis=0)
Int_s_bottom = Int_s_sim2 - np.std(MC_Is_sim, axis=0)

Int_d_top = Int_d_sim2 + np.std(MC_Id_sim, axis=0)
Int_d_bottom = Int_d_sim2 - np.std(MC_Id_sim, axis=0)

Int_b_top = Int_b_sim2 + np.std(MC_Ib_sim, axis=0)
Int_b_bottom = Int_b_sim2 - np.std(MC_Ib_sim, axis=0)

tmp_se = pd.Series([opt_K1,opt_K2,opt_Mag_d,opt_Mag_s,opt_Mag_b,
                    np.std(MC_K1),np.std(MC_K2),
                    np.std(MC_Mag_d),np.std(MC_Mag_s),np.std(MC_Mag_b)],
                    index=params_df.columns)
params_df = params_df.append(tmp_se, ignore_index=True)


####################################
# Plot
####################################

fig1 = plt.figure(figsize=(4.5,3.5))
ax1 = fig1.add_subplot(111)

ax1.plot(concsim*1E6,Int_s_sim2/opt_Mag_s,label="ssRNA fit",linewidth=1.5,color="turquoise")
ax1.plot(concsim*1E6,Int_d_sim2/opt_Mag_d,label="dsRNA fit",linewidth=1.5,color="dodgerblue")
ax1.plot(concsim*1E6,Int_b_sim2/opt_Mag_b,label="Bound fit",linewidth=1.5,color="tomato")

ax1.fill_between(concsim*1E6,Int_s_top/opt_Mag_s,Int_s_bottom/opt_Mag_s,color="turquoise",alpha=0.3)
ax1.fill_between(concsim*1E6,Int_d_top/opt_Mag_d,Int_d_bottom/opt_Mag_d,color="dodgerblue",alpha=0.3)
ax1.fill_between(concsim*1E6,Int_b_top/opt_Mag_b,Int_b_bottom/opt_Mag_b,color="tomato",alpha=0.3)

ax1.plot(conc*1E6,ss/opt_Mag_s,label="ssRNA exp",markeredgewidth=2, color='white',linewidth=0.,
         markeredgecolor="turquoise", marker='o', markersize=6)
ax1.plot(conc*1E6,ds/opt_Mag_d,label="dsRNA exp",markeredgewidth=2, color='white',linewidth=0.,
         markeredgecolor="dodgerblue", marker='o', markersize=6)
ax1.plot(conc*1E6,b/opt_Mag_b,label="bound exp",markeredgewidth=2, color='white',linewidth=0.,
         markeredgecolor="tomato", marker='o', markersize=6)

ax1.set_ylabel("Fractional population",fontsize=12)
ax1.set_xlabel("DDX3X E348Q [uM]",fontsize=12)
ax1.yaxis.major.formatter._useMathText = True
ax1.set_title("DDX3X titration",fontsize=14) 
ax1.legend()
#ax1.set_ylim(0,1E13)
plt.tight_layout()
plt.legend()

plt.savefig(plotname+".pdf")
params_df.to_csv("params.csv")