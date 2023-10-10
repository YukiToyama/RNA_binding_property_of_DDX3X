# -*- coding: utf-8 -*-
"""
# Fitting of peak intensities to the one-site binding model

# numpy ver. 1.24.3
# matplotlib ver. 3.7.1
# pandas ver. 1.5.3
# lmfit ver. 1.2.2
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

#########################
## Data import
#########################
# Peak intensities 
data = pd.read_csv("input.csv")

# Residues used to fit the data
residuelist = ['M221','M352','M355','M370','M380']

# Ligand concentration in molar
conc = np.array(data[(data['residue']==residuelist[0])&(data['state']=="free")].conc*1E-6)

# Protein concentration in molar
CT = 50E-6 

#########################
## 1 site binding
#########################

def onesite(CT,conc,K): 
    L = (-1*K*CT+K*conc-1+np.sqrt((K*CT-K*conc+1)**2+4*K*conc))/(2*K)
    PL = conc-L
    P = CT-PL
    free_pop = P/CT
    bound_pop = PL/CT  
    return free_pop, bound_pop


#########################
## Define parameters minimize function
#########################

## Global parameter
fit_params = Parameters()
fit_params.add('K',value=1500,min=0)
  
## Residue specific parameter
for i in residuelist:
    if ((data['residue'] == str(i)) & (data['conc'] == 0) & (data['state'] == "free")).any():
        fit_params.add('I0_'+str(i)+'_free',value=3E11,min=0)
    if ((data['residue'] == str(i)) & (data['conc'] == 0) & (data['state'] == "bound")).any():
        fit_params.add('I0_'+str(i)+'_bound',value=3E11,min=0)  
        
def objective(fit_params):
    
    K = fit_params['K'] 
    free_pop, bound_pop = onesite(CT,conc,K)
    
    residual = np.zeros(0)
    
    for i in residuelist:     
      
      if ((data['residue'] == str(i)) & (data['conc'] == 0) & (data['state'] == "free")).any():
          I0 = fit_params['I0_'+str(i)+'_free'] 
          volume = np.array(data[(data['residue']==str(i)) & (data['state']=="free")].volume)
          residual = np.append(residual,free_pop*I0 - volume)

                        
      if ((data['residue'] == str(i)) & (data['conc'] == 0) & (data['state'] == "bound")).any():
          I0 = fit_params['I0_'+str(i)+'_bound'] 
          volume = np.array(data[(data['residue']==str(i)) & (data['state']=="bound")].volume)
          residual = np.append(residual,bound_pop*I0 - volume)
    
    return residual 

result = minimize(objective,fit_params,method="leastsq")

print(fit_report(result))
with open("report.txt", 'w') as fh:
    fh.write(fit_report(result))
   
opt_params = result.params
opt_K = opt_params["K"].value 
std_K = opt_params["K"].stderr
RMSD = np.sqrt(result.chisqr/result.ndata)

#########################
## Error analysis
#########################

## Update the parameter
## Global parameter
fit_params = Parameters()
fit_params.add('K',value=opt_K,min=0)
  
## Residue specific parameter
for i in residuelist:
    if ((data['residue'] == str(i)) & (data['conc'] == 0) & (data['state'] == "free")).any():
        fit_params.add('I0_'+str(i)+'_free', value=opt_params['I0_'+str(i)+'_free'].value, min=0)
    if ((data['residue'] == str(i)) & (data['conc'] == 0) & (data['state'] == "bound")).any():
        fit_params.add('I0_'+str(i)+'_bound', value=opt_params['I0_'+str(i)+'_bound'].value, min=0)  

## Monte Carlo iteration
cycle = 1000
MC_K = np.zeros(cycle)

for k in range(cycle):
    noise = np.random.normal(0,RMSD,result.ndata)
    def MC(fit_params):
        
        K = fit_params['K'] 
        
        free_pop, bound_pop = onesite(CT,conc,K)
        opt_free_pop, opt_bound_pop = onesite(CT,conc,opt_K)
        
        residual = np.zeros(0)
        
        for i in residuelist:     
          
          if ((data['residue'] == str(i)) & (data['conc'] == 0) & (data['state'] == "free")).any():
              
              I0 = fit_params['I0_'+str(i)+'_free'] 
              syntheticdata = opt_params['I0_'+str(i)+'_free'].value * opt_free_pop 
              
              residual = np.append(residual,free_pop*I0 - syntheticdata)
    
                            
          if ((data['residue'] == str(i)) & (data['conc'] == 0) & (data['state'] == "bound")).any():
              I0 = fit_params['I0_'+str(i)+'_bound'] 
              syntheticdata = opt_params['I0_'+str(i)+'_bound'].value * opt_bound_pop
              
              residual = np.append(residual,bound_pop*I0 - syntheticdata)
        
        return residual + noise 
    
    result_temp = minimize(MC,fit_params,method="leastsq")   
    with open("MC.txt", 'a') as fh:
        fh.write("\n\n#######################\n")
        fh.write("MC iterlation " + str(k))
        fh.write("\n#######################\n\n")
        fh.write(fit_report(result_temp))
       
    print(k)
    print(fit_report(result_temp))
    temp_params = result_temp.params
    MC_K[k] = temp_params["K"].value 


with open("report.txt", 'a') as fh:
    fh.write("\n\n##################\n")
    fh.write("Parameters from the Monte Carlo error analysis")
    fh.write("\n##################\n\n")
    fh.write("K = "+str(opt_K)+" (M^-1)\n")
    fh.write("K_std = "+str(std_K)+" (M^-1)\n")
    fh.write("K_MC = "+str(np.std(MC_K))+" (M^-1)\n")
       
#########################
## Plot
#########################

pdf = PdfPages('plot.pdf')

concsim = np.arange(1,101,1)*1E-6
free_sim, bound_sim = onesite(CT,concsim,opt_K)
fig1 = plt.figure(figsize=(4,3))
plt.rcParams["font.family"] = "Arial"  

for i in residuelist:
  
  if ((data['residue'] == str(i)) & (data['conc'] == 0) & (data['state'] == "free")).any():
      
      ax1 = fig1.add_subplot(111)
      opt_I0 = opt_params["I0_"+str(i)+"_free"].value 
      volume = np.array(data[(data['residue']==str(i)) & (data['state']=="free")].volume)
      ax1.plot(conc*1E6,volume/opt_I0,c="orangered",marker='o',ls="None",label="Exp")
      ax1.plot(concsim*1E6,free_sim,c="orangered",label="Fit")
      ax1.set_ylabel('Normalized intensity',fontsize=12)
      ax1.set_xlabel('Concentration [$\mu$M]',fontsize=12)
      ax1.yaxis.major.formatter._useMathText = True       
      ax1.set_title(str(i)+" free",fontsize=16) 
      ax1.set_ylim(0,1.1) 
      
      ax1.legend()
      plt.tight_layout()

      pdf.savefig()
      plt.clf()
      
  if ((data['residue'] == str(i)) & (data['conc'] == 0) & (data['state'] == "bound")).any():
      
      
      ax1 = fig1.add_subplot(111)
      opt_I0 = opt_params["I0_"+str(i)+"_bound"].value 
      volume = np.array(data[(data['residue']==str(i)) & (data['state']=="bound")].volume)
      ax1.plot(conc*1E6,volume/opt_I0,c="skyblue",marker='o',ls="None",label="Exp")
      ax1.plot(concsim*1E6,bound_sim,c="skyblue",label="Fit")
      ax1.set_ylabel('Normalized intensity',fontsize=12)
      ax1.set_xlabel('Concentration [$\mu$M]',fontsize=12)
      ax1.yaxis.major.formatter._useMathText = True       
      ax1.set_title(str(i)+" bound",fontsize=16)
      ax1.set_ylim(0,1.1) 
      ax1.legend()
      plt.tight_layout()

      pdf.savefig()
      plt.clf()
  
  
plt.close()
pdf.close()

    

    
 
