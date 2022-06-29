# -*- coding: utf-8 -*-
"""Plotscript j0j1j2j3 model

This script produces the plots for the j0j1j2j3 model provided that
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
import os

nSamples = 21
j2Max = 4.
j3Max = 4.

# init = 'singlet-triplet'
# init = 'perturbedIdentity'
init = 'random'
alpha = 100
max_steps = 2000

# Plot the results
path = 'data/'
savepath = 'plots/jModel/jModel_plotData_init_{}'.format(init)

filenames = []

dataArray = np.zeros((0,3))

for file in os.listdir(path):
    filename = os.fsdecode(file)
    if (filename.startswith('modelH_j0j1j2j3orig-init_{}-'.format(init)) and filename.endswith('alpha_{}-L1Cutoff_None-max_steps_{}-froStep_grid-l1Step_grid-hybrid_True.npy'.format(alpha,max_steps))): # whatever file types you're using...:
        data = np.load(path+filename).item()
        # compute percentage improvement
        zValue = data['objvals'][0]/data['objvals'][1]
        plotData = [data['parameter values'][2],data['parameter values'][3],zValue]     
        dataArray = np.vstack([dataArray,plotData])

dataArray = dataArray[np.argsort(dataArray[:, 1])]
dataArray = dataArray[np.argsort(dataArray[:, 0], kind='mergesort')]
np.savetxt(savepath+'.csv', dataArray, delimiter=' ')   # X is an array

# print(dataArray[0:50,:])
dataMatrix = dataArray[:,2].reshape((nSamples,nSamples)).T

# Save the plot results
plt.imshow(dataMatrix,extent=[0,j2Max,j3Max,0])
plt.ylabel(r"$J_3/J$")
plt.xlabel(r"$J_2/J$")
plt.gca().invert_yaxis()
plt.title(r'Optimized nonstoquasticity of the $J_0-J_1-J_2-J_3$ model')
plt.tight_layout()
plt.colorbar()
plt.savefig(savepath+'.pdf', bbox_inches='tight')
plt.show()