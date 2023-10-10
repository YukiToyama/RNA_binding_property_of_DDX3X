# -*- coding: utf-8 -*-
"""
# 3Q forbidden coherence transfer experiment
# This script allows you to extract eta and delta from a peakipy peaklist.
# Ref: Sun et al., J. Phys. Chem. B 2011, 115, 14878 – 14884

# numpy ver. 1.24.3
# matplotlib ver. 3.7.1
# pandas ver. 1.5.3
# lmfit ver. 1.2.2

"""

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import minimize, Parameters, fit_report
from matplotlib.backends.backend_pdf import PdfPages


############################
# Input files
############################
# Delay list
t = np.loadtxt("vdlist")

# Peak intensities 
# Note that the odd planes are from the forbidden datasets (Ia) 
# and the even planes are from the allowed datasets (Ib). 
data = pd.read_csv("input.csv")

# NS of the forbidden experiment/NS of the allowed experiment (L6/L7)
# (must be set properly)
scale=1.5 

# Output name
outname = "D2"

############################
# Functions and constants
############################
def function(Eta,Delta,T):
    sq=(Eta**2 + Delta**2)**0.5
    return 0.75*Eta*np.tanh(sq*T)/(sq-Delta*np.tanh(sq*T))

hbar=6.626E-34/2/np.pi
P2x=-0.5
gh=2.67E8
rH=1.813E-10

############################
# Fitting
############################

groups = data.groupby("assignment")
col1 = ['assign','res','eta','eta_error','delta','deleta_error',
        'S2tc','S2tc_error']
df = pd.DataFrame(columns=col1)


fit_params = Parameters()
fit_params.add('eta',value=50)
fit_params.add('delta',value=-10)
pdf = PdfPages('plot.pdf')
fig1 = plt.figure(figsize=(6,4))

for ind,group in groups:
   # data import
   name=group.assignment.iloc[0]
   res=re.sub(r"\D", "",name[1:4])
   print (name)
   
   # amplitude
   amp=np.array(group.amp)
   # amplitude error
   error=np.array(group.amp_err)
   # Ia
   Ia=amp[1::2]/scale
   Ia_error=error[1::2]/scale
   # Ib   
   Ib=amp[::2]
   Ib_error=error[::2]
   
   # Ia/Ib ratio error
   ratio_error=np.empty_like(Ia)
   for s in range(len(ratio_error)):
       ratio_error[s]=Ia[s]/Ib[s]*np.sqrt((Ia_error[s]/Ib[s])**2+(Ia[s]*Ib_error[s]/Ib[s]**2)**2)
   
   def objective(fit_params):

       eta = fit_params['eta']  
       delta = fit_params['delta']  
   
       return function(eta,delta,t)-Ia/Ib
   
   result = minimize(objective,fit_params)
   
   print(fit_report(result))
   
   with open("fit_report.txt", 'a') as fh:
       fh.write("##################\n")
       fh.write("Residue " + str(res)+"\n")
       fh.write("##################\n")
       fh.write(fit_report(result)+"\n\n")
   opt_params = result.params
   
   opt_eta = opt_params["eta"].value 
   std_eta = opt_params["eta"].stderr
   
   opt_delta = opt_params["delta"].value 
   std_delta = opt_params["delta"].stderr
   
   S2tc = opt_eta*rH**6/((P2x**2)*(gh**4)*(hbar**2))*1E14*10/9.
   S2tc_error = std_eta*rH**6/((P2x**2)*(gh**4)*(hbar**2))*1E14*10/9.

   
   tmp_se = pd.Series([name, res,opt_eta,std_eta,opt_delta,std_delta,
                       S2tc, S2tc_error],
   index=df.columns )
   df=df.append(tmp_se, ignore_index=True )
   
   
   tsim=np.linspace(0,1.1*np.max(t),100)
   ax1 = fig1.add_subplot(111)
   ax1.plot(tsim*1E3,function(opt_eta,opt_delta,tsim),color='black',ls="-")
   ax1.scatter(t*1E3,Ia/Ib,color='black',marker='o',facecolor="None")
   ax1.errorbar(t*1E3,Ia/Ib,fmt='none',yerr=ratio_error,ecolor="black")
   ax1.set_ylabel('$I_a$/$I_b$ ratio',fontsize=14)
   ax1.set_xlabel('Relaxation time [ms]',fontsize=14)
   ax1.yaxis.major.formatter._useMathText = True
   ax1.set_title(name,fontsize=14)
   #ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
   ax1.tick_params(direction='out',axis='both',length=5,width=2,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=12)
   ax1.text(0.35,0.2,"$S^2\\tau_c$ = %.2e $\pm$ %.2e "%  
   (S2tc,S2tc_error),transform=ax1.transAxes,va="top",fontsize=14)
   #ax1.legend()
   ax1.grid(linestyle='dashed',linewidth=2,dash_capstyle='round',dashes=(1,3))
   plt.tight_layout()
   pdf.savefig()
   plt.clf()

plt.close('all')
pdf.close() 
df.to_csv(outname+".csv")   