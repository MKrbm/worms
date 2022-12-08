# -*- coding: utf-8 -*-
"""Plotscript random stoquastic Hamiltonians

This script produces the plots for the random stoquastic Hamiltonians provided that 
the data was previously generated.  

The script was used to create the plots in 

	Hangleiter, Roth, Nagaj, and Eisert. "Easing the Monte Carlo sign problem" 
	https://arxiv.org/abs/1906.02309

Authors: 
	Dominik Hangleiter and Ingo Roth

Licence: 
	This project is licensed under the MIT License - see the LICENSE.md file for details.


"""


import numpy as np
import matplotlib.pyplot as plt

#Parameters
alpha = 40

path = 'data/'
savepath = 'plots/randomStoquasticPlot'
dataFile =  'plots/modelH_StoquasticHaarProjected-init_identity-locality_2-localDim_6-pars_None-alpha_50-L1Cutoff_None-max_steps_500-froStep_grid-l1Step_grid-hybrid_False_2.csv'

plotData= np.loadtxt(dataFile,delimiter = ',')
stamps = [2,3,4,5,6]

boxprops = dict(linewidth=1, color='black')
flierprops = dict(marker='o',markeredgecolor = 'black', markerfacecolor = 'white', markersize=3,
                  linestyle='None')
medianprops = dict(linewidth=1, color='magenta')
whiskerprops = dict(linestyle='-', linewidth=1, color='black')


plt.boxplot(plotData,manage_xticks=True,patch_artist = True,positions = stamps,
		whiskerprops = whiskerprops, boxprops = boxprops,flierprops = flierprops,medianprops = medianprops)
plt.yscale('log')
plt.xlabel(r'Local dimension')
plt.ylabel(r'$\nu_1(\mathcal{O}H\mathcal{O})/\nu_1(H)$')

plt.tight_layout()
plt.savefig(savepath+'.pdf', bbox_inches='tight')
	
plt.show()

