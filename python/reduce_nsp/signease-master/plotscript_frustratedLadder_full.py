# -*- coding: utf-8 -*-
"""Plotscript frustrated Ladder model

This script produces the plots for the frustrated Ladder model provided that 
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
import os
import matplotlib.pyplot as plt

nSamples = 20
JorthMax = 1.5
JcrossMax = 1.5


alpha = 40
max_steps = 2000
init = 'random'
init = 'perturbedIdentityUncorrected'
savepath = 'plots/frustratedLadder/frustratedLadder_plotData_init-{}_alpha={}'.format(init,alpha)
path = 'data/'

filenames = []

avSignRatioDataArray = np.zeros((0,3))
avSignBeforeDataArray = np.zeros((0,3))
avSignAfterDataArray = np.zeros((0,3))
nonStoqDataArray = np.zeros((0,3))


for file in os.listdir(path):
	filename = os.fsdecode(file)
	if (filename.startswith('modelH_frustratedLadder-init_{}-locality_'.format(init) ) and filename.endswith('alpha_{}-L1Cutoff_None-max_steps_{}-froStep_grid-l1Step_grid-hybrid_True.npy'.format(alpha,max_steps))): # whatever file types you're using...
		data = np.load(path+filename).item()
		# compute percentage improvement
		
		if data['avsigns'][1] ==1: # Hack to get rid of first row and column.
			pass
		else:	
			avSignRatio = -np.log(data['avsigns'][0])/-np.log(data['avsigns'][1])
			avSignBefore = -np.log(data['avsigns'][1])
			avSignAfter = -np.log(data['avsigns'][0])

			plotDataAvSignRatio = [data['parameter values'][0],data['parameter values'][2],avSignRatio]
			avSignRatioDataArray = np.vstack([avSignRatioDataArray,plotDataAvSignRatio])

			plotDataAvSignBefore = [data['parameter values'][0],data['parameter values'][2],avSignBefore]
			avSignBeforeDataArray = np.vstack([avSignBeforeDataArray,plotDataAvSignBefore])

			plotDataAvSignAfter = [data['parameter values'][0],data['parameter values'][2],avSignAfter]
			avSignAfterDataArray = np.vstack([avSignAfterDataArray,plotDataAvSignAfter])
			

			nonStoq = data['objvals'][0]/data['objvals'][1]
			plotDataNonStoq = [data['parameter values'][0],data['parameter values'][2],nonStoq]
			nonStoqDataArray = np.vstack([nonStoqDataArray,plotDataNonStoq])
		

		
avSignRatioDataArray = avSignRatioDataArray[np.argsort(avSignRatioDataArray[:, 1])]
avSignRatioDataArray = avSignRatioDataArray[np.argsort(avSignRatioDataArray[:, 0], kind='mergesort')]
print(avSignRatioDataArray.shape)
np.savetxt(savepath+'_avSign_ratio.csv', avSignRatioDataArray, delimiter=' ')   # X is an array
avSignRatioMatrix = avSignRatioDataArray[:,2].reshape((nSamples,nSamples)).T

avSignBeforeDataArray = avSignBeforeDataArray[np.argsort(avSignBeforeDataArray[:, 1])]
avSignBeforeDataArray = avSignBeforeDataArray[np.argsort(avSignBeforeDataArray[:, 0], kind='mergesort')]
print(avSignBeforeDataArray.shape)
np.savetxt(savepath+'_avSign_before.csv', avSignBeforeDataArray, delimiter=' ')   # X is an array
avSignBeforeMatrix = avSignBeforeDataArray[:,2].reshape((nSamples,nSamples)).T

avSignAfterDataArray = avSignAfterDataArray[np.argsort(avSignAfterDataArray[:, 1])]
avSignAfterDataArray = avSignAfterDataArray[np.argsort(avSignAfterDataArray[:, 0], kind='mergesort')]
print(avSignAfterDataArray.shape)
np.savetxt(savepath+'_avSign_after.csv', avSignAfterDataArray, delimiter=' ')   # X is an array
avSignAfterMatrix = avSignAfterDataArray[:,2].reshape((nSamples,nSamples)).T


nonStoqDataArray = nonStoqDataArray[np.argsort(nonStoqDataArray[:, 1])]
nonStoqDataArray = nonStoqDataArray[np.argsort(nonStoqDataArray[:, 0], kind='mergesort')]
print(nonStoqDataArray.shape)
np.savetxt(savepath+'_nonStoq.csv', nonStoqDataArray, delimiter=' ')   # X is an array
nonStoqMatrix = nonStoqDataArray[:,2].reshape((nSamples,nSamples)).T



plt.rc('text', usetex=True)
# plt.rc('font', family='Helvetica')
# Creates two subplots and unpacks the output array immediately
f, ax = plt.subplots(2, 2)
# , sharey='row',sharex='col')

im1 = ax[0,0].imshow(nonStoqMatrix,extent=[JorthMax/nSamples,JorthMax,JcrossMax,JcrossMax/nSamples,])
ax[0,0].invert_yaxis()
plt.colorbar(im1,ax = ax[0,0])

im2 = ax[0,1].imshow(avSignRatioMatrix,extent=[JorthMax/nSamples,JorthMax,JcrossMax,JcrossMax/nSamples])
ax[0,1].invert_yaxis()
plt.colorbar(im2,ax = ax[0,1])

im3 = ax[1,0].imshow(avSignBeforeMatrix,extent=[JorthMax/nSamples,JorthMax,JcrossMax,JcrossMax/nSamples])
ax[1,0].invert_yaxis()
plt.colorbar(im2,ax = ax[1,0])

im4 = ax[1,1].imshow(avSignAfterMatrix,extent=[JorthMax/nSamples,JorthMax,JcrossMax,JcrossMax/nSamples])
ax[1,1].invert_yaxis()
plt.colorbar(im4,ax = ax[1,1])


ax[1,0].set_xlabel(r"$J_\times/J_\parallel$")
ax[0,0].set_ylabel(r"$J_\perp/J_\parallel$")

ax[1,0].set_ylabel(r'$J_\perp/J_\parallel$')
ax[1,1].set_xlabel(r'$J_\times/J_\parallel$')

ax[0,0].set_title('Nonstoquasticity improvement')
ax[0,1].set_title(r'$\log \langle \mathrm{sign} \rangle ^{-1}$ improvement')
ax[1,0].set_title(r'$\log \langle \mathrm{sign} \rangle ^{-1}$ before optimization')
ax[1,1].set_title(r'$\log \langle \mathrm{sign} \rangle ^{-1}$ after optimization')
plt.tight_layout()
plt.savefig(savepath+'.pdf', bbox_inches='tight')
plt.show()
