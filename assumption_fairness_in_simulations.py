# Check how fair is the assumption of same Q accross the brain for all hypotheses
import os
import glob
import numpy
import nibabel
import nilearn.plotting
import nilearn.maskers
import warnings
import matplotlib.pyplot as plt
from nilearn.datasets import load_mni152_gm_mask, load_mni152_gm_template
from nilearn.datasets import load_mni152_wm_mask, load_mni152_brain_mask
import seaborn
import pandas

import importlib
import utils
import sys

importlib.reload(utils) # reupdate imported codes, useful for debugging


##################
# SETTING UP NECESSARY INFO
##################

results_dir = 'results_Q_assumptions_in_simulations'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

if not os.path.exists('results_Q_assumptions_in_simulations/temp'):
    os.mkdir('results_Q_assumptions_in_simulations/temp')


##################
# Load masks
##################

masks_for_masker = glob.glob('results_Q_assumptions/temp/*mask.nii'.format(results_dir))
masks_for_masker.append('masking/mask_99.nii')
names = [mask_for_masker.split('/')[-1][:-4] for mask_for_masker in masks_for_masker]

##################
# define generated data properties
##################

J = 1524955 # voxels nb in gm mask
K = 20 # 20 studies
try:
    corr = float(sys.argv[1]) #0.1, 0.3, 0.5, 0.8
except:
    corr = 0.8
print('****')
print("CORR = ", corr)
mean=2

#######################################
# Simulation 0: The dumbest, null case: independent pipelines, mean 0, variance 1 (totally iid data)
#######################################
mu=0
sigma=1
rng = numpy.random.default_rng()
Zmaps_sim0  = mu + sigma * rng.standard_normal(size=(K,J))


#######################################
# Simulation 1: Null data with correlation: Induce correlation Q, mean 0, variance 1
#######################################

Zmaps_sim1 = utils.null_data(J=J, K=K, covar=corr) 


#######################################
# Simulation 2: Non-null but totally homogeneous data: Correlation Q but all pipelines share mean mu=mu1>0, variance 1
#######################################

Zmaps_sim2 = utils.non_null_homogeneous_data(J=J, K=K, covar=corr, mean=mean)


#######################################
# Simulation 3: Non-null but totally heterogeneous data: Correlation Q, all pipelines share same mean, but 50% of voxels have mu=mu1, 
#######################################
Zmaps_sim3 = utils.non_null_data_heterogeneous(J=J, K=K, covar=corr, mean=mean)


#######################################
# COMPUPTE Q FOR EACH SIMULATION and save Q matrices heatmap per simulation
#######################################

compute_Q=True
for sim_number, Zmaps in enumerate([Zmaps_sim0, Zmaps_sim1, Zmaps_sim2, Zmaps_sim3]):
    # create fake nii with MNI dimension
    if compute_Q:
        masker = nilearn.maskers.NiftiMasker(
                mask_img=load_mni152_gm_mask()).fit()
        print("***")
        print("simulation ", sim_number)
        print("***")
        niftis_from_zmaps = []
        print("Inverse transforming Zmaps")
        for ind, Zmap in enumerate(Zmaps):
            nifti_from_zmap = masker.inverse_transform(Zmap)
            niftis_from_zmaps.append(nifti_from_zmap)
        print("Inverse transforming DONE")
        print('---')
        print("masking and computing Q...")
        # compute Q with different AAL masks
        for ind, mask_for_masker in enumerate(masks_for_masker):
            masker = nilearn.maskers.NiftiMasker(
                mask_img=mask_for_masker)
            data = masker.fit_transform(niftis_from_zmaps)
            # Computing Q and saving
            Q = numpy.corrcoef(data)      
            numpy.save('{}/temp/Q_{}_sim{}'.format(results_dir, names[ind], sim_number), Q)
    print('---')
    print("Computing and saving simulation ", sim_number)
    utils.display_matrices(results_dir, sim_number, corr)



import shutil
shutil.rmtree("{}/temp".format(results_dir))