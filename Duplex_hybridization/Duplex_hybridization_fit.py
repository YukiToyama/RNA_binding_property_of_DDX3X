# -*- coding: utf-8 -*-
"""
# Fitting of UA-12mer dsRNA and ssRNA signal intensities as a function of temperature
# to obtain deltaH and deltaS values.

# numpy ver. 1.24.3
# matplotlib ver. 3.7.1
# pandas ver. 1.5.3
# lmfit ver. 1.2.2
"""


import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
import pandas as pd


#########################
## Functions
#########################

R = 1.99  #cal/(K*mol)

def DM(LT,Keq):
  
    a = 2*Keq
    b = 1
    c = -1*LT
    
    roots = np.roots([a,b,c])
    # get real and positive root
    roots = np.real(roots[np.where((np.imag(roots) == 0) & (np.real(roots) > 0))])[0]
    Monomer = roots
    Dimer = Keq*Monomer**2
    P_Monomer = Monomer/LT
    P_Dimer = 2*Dimer/LT

    return P_Monomer,P_Dimer

def Keq(T,dH,dS):
    return np.exp((-dH+T*dS)/(R*T))

#########################
## Data import
#########################
# Total RNA concentration [M]

LT = 100E-6

# Input data file
data = pd.read_csv("input.csv")

# Temperature in degree
# Datapoints from 22.5 to 40 deg were used for fitting.
# Three spectra were recorded at each temperature.

temp = np.array([22.5,25,27.5,30,32.5,35,37.5,40])

ds = np.zeros(len(temp))
ss = np.zeros(len(temp))
ds_e = np.zeros(len(temp))
ss_e = np.zeros(len(temp))

for i in range(len(temp)):
    ds[i]=np.average(np.array(data[(data['Temp']==temp[i])].Vol_ds))
    ds_e[i]=np.std(np.array(data[(data['Temp']==temp[i])].Vol_ds))
    ss[i]=np.average(np.array(data[(data['Temp']==temp[i])].Vol_ss))
    ss_e[i]=np.std(np.array(data[(data['Temp']==temp[i])].Vol_ss))

ds_pop = ds/(ds+ss)
ss_pop = 1-ds_pop
ds_pop_e = np.sqrt((ds_e/(ds+ss))**2+(ds*ds_e/(ds+ss)**2)**2+(ds*ss_e/(ds+ss)**2)**2) 
ss_pop_e = np.sqrt((ss_e/(ds+ss))**2+(ss*ds_e/(ds+ss)**2)**2+(ss*ss_e/(ds+ss)**2)**2) 

#########################
## Fitting
#########################

fit_params = Parameters()
fit_params.add('dH',value=60000,vary=True)
fit_params.add('dS',value=212,vary=True)

def objective(fit_params):
    M = np.zeros(len(temp))
    D = np.zeros(len(temp))
    dH = fit_params['dH']
    dS = fit_params['dS']
    for i in range(len(temp)):
        K = Keq(temp[i]+273.15,dH,dS)
        M[i],D[i] = DM(LT,K)
    
    return (D-ds_pop)


result = minimize(objective,fit_params)
print(fit_report(result))
with open("report.txt", 'w') as fh:
    fh.write(fit_report(result))

opt_params = result.params
opt_dH = opt_params["dH"].value 
std_dH = opt_params["dH"].stderr
opt_dS = opt_params["dS"].value 
std_dS = opt_params["dS"].stderr

#########################
## Plotting
#########################

# Get smoothlines for plotting
simtemp=np.linspace(np.min(temp)-5,np.max(temp)+5,100)
Msim = np.zeros(len(simtemp))
Dsim = np.zeros(len(simtemp))
for i in range(len(simtemp)):
    K = Keq(simtemp[i]+273.15,opt_dH,opt_dS)
    Msim[i],Dsim[i] = DM(LT,K)
    
fig1 = plt.figure(figsize=(4,3))
ax1 = fig1.add_subplot(111)
ax1.errorbar(temp,ds_pop,yerr=ds_pop_e,fmt='o', capsize=4,markersize=5,label="dsRNA",color="skyblue")
ax1.errorbar(temp,ss_pop,yerr=ss_pop_e,fmt='o', capsize=4, markersize=5,label="ssRNA",color="tomato")
ax1.plot(simtemp,Dsim,color="skyblue")
ax1.plot(simtemp,Msim,color="tomato")
ax1.set_title("$\\Delta$H = " + str(round(opt_dH/1000,2)) + " kcal/mol, "+
          "$\\Delta$S = " + str(round(opt_dS,2)) + " cal/(K*mol)")

ax1.set_xlabel("Temperature [deg]")
ax1.set_ylabel("Fraction")

ax1.spines['top'].set_linewidth(0.0)
ax1.spines['right'].set_linewidth(0.0)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)
plt.legend()
plt.tight_layout()

plt.savefig("plot.pdf")
