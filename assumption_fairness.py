# Check how fair is the assumption of same Q accross the brain for all hypotheses
import os
import glob
import numpy
import nibabel
import nilearn.plotting
import nilearn.input_data
from nilearn import masking
from nilearn import plotting
from nilearn import image
import warnings
import matplotlib.pyplot as plt
from nilearn.datasets import load_mni152_gm_mask, load_mni152_gm_template
from nilearn.datasets import load_mni152_wm_mask, load_mni152_brain_mask
import seaborn
import sklearn.metrics
import importlib

import utils

importlib.reload(utils) # reupdate imported codes, useful for debugging

##################
# SETTING UP NECESSARY INFO
##################

# gather all 3D brain data that will be used to compute a mask
subjects = []
for path_to_sub in glob.glob("/home/jlefortb/narps_open_pipelines/data/neurovault/[1-9]*"):
    if os.path.exists(os.path.join(path_to_sub, 'hypo1_unthresh.nii.gz')):
        subjects.append(path_to_sub.split('/')[-1])

data_path = '/home/jlefortb/narps_open_pipelines/data/neurovault'
results_dir = 'results_Q_assumptions'

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

if not os.path.exists('results_Q_assumptions/temp'):
    os.mkdir('results_Q_assumptions/temp')


##################
# Create AAL masks
##################
from nilearn.datasets import fetch_atlas_aal
atlas_aal = fetch_atlas_aal()
frontal = ['Frontal_Sup_L',
 'Frontal_Sup_R',
 'Frontal_Sup_Orb_L',
 'Frontal_Sup_Orb_R',
 'Frontal_Mid_L',
 'Frontal_Mid_R',
 'Frontal_Mid_Orb_L',
 'Frontal_Mid_Orb_R',
 'Frontal_Inf_Oper_L',
 'Frontal_Inf_Oper_R',
 'Frontal_Inf_Tri_L',
 'Frontal_Inf_Tri_R',
 'Frontal_Inf_Orb_L',
 'Frontal_Inf_Orb_R',
 'Frontal_Sup_Medial_L',
 'Frontal_Sup_Medial_R',
 'Frontal_Med_Orb_L',
 'Frontal_Med_Orb_R']

occipital =[
 'Occipital_Sup_L',
 'Occipital_Sup_R',
 'Occipital_Mid_L',
 'Occipital_Mid_R',
 'Occipital_Inf_L',
 'Occipital_Inf_R'
]
parietal =[
 'Parietal_Sup_L',
 'Parietal_Sup_R',
 'Parietal_Inf_L',
 'Parietal_Inf_R',
]
temporal = [
 'Temporal_Sup_L',
 'Temporal_Sup_R',
 'Temporal_Pole_Sup_L',
 'Temporal_Pole_Sup_R',
 'Temporal_Mid_L',
 'Temporal_Mid_R',
 'Temporal_Pole_Mid_L',
 'Temporal_Pole_Mid_R',
 'Temporal_Inf_L',
 'Temporal_Inf_R'
]
cerebellum = [
 'Cerebelum_Crus1_L',
 'Cerebelum_Crus1_R',
 'Cerebelum_Crus2_L',
 'Cerebelum_Crus2_R',
 'Cerebelum_3_L',
 'Cerebelum_3_R',
 'Cerebelum_4_5_L',
 'Cerebelum_4_5_R',
 'Cerebelum_6_L',
 'Cerebelum_6_R',
 'Cerebelum_7b_L',
 'Cerebelum_7b_R',
 'Cerebelum_8_L',
 'Cerebelum_8_R',
 'Cerebelum_9_L',
 'Cerebelum_9_R',
 'Cerebelum_10_L',
 'Cerebelum_10_R'
 ]

indices_frontal = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in frontal]]
indices_occipital = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in occipital]]
indices_parietal = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in parietal]]
indices_temporal = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in temporal]]
indices_cerebellum = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in cerebellum]]
indices_aal = [atlas_aal.indices[i] for i in [atlas_aal.labels.index(roi) for roi in atlas_aal.labels]]

atlas_aal_nii = nibabel.load(atlas_aal.maps)
# resample MNI gm mask space
atlas_aal_nii = image.resample_to_img(
                        atlas_aal_nii,
                        load_mni152_brain_mask(),
                        interpolation='nearest')

# function to save PNG of mask
def compute_save_display_mask(ROI_name, indices):
    # compute ROI mask
    indexes_ROI = [numpy.where(atlas_aal_nii.get_fdata() == int(indice)) for indice in indices]
    fake_ROI = numpy.zeros(atlas_aal_nii.get_fdata().shape)
    for indexes in indexes_ROI:
        fake_ROI[indexes] = 1
    ROI_img = nilearn.image.new_img_like(atlas_aal_nii, fake_ROI)
    # shape ROI_mask from mask_participant to ensure all voxels are present
    mask_participant = nibabel.load('masking/mask_99.nii') # load mask made from participant zmaps + MNI gm mask
    masks = [mask_participant, ROI_img]
    ROI_img = masking.intersect_masks(masks, threshold=1, connected=True)
    print("saving... ",ROI_name)
    nibabel.save(ROI_img, "{}/temp/{}_mask.nii".format(results_dir, ROI_name))
    # Visualize the resulting image
    nilearn.plotting.plot_roi(ROI_img, title="{} regions of AAL atlas".format(ROI_name))
    plt.savefig('{}/{}_mask.png'.format(results_dir, ROI_name), dpi=300)
    plt.close('all')


compute_save_display_mask('Frontal_aal', indices_frontal)
compute_save_display_mask('occipital_aal', indices_occipital)
compute_save_display_mask('parietal_aal', indices_parietal)
compute_save_display_mask('temporal_aal', indices_temporal)
compute_save_display_mask('cerebellum_aal', indices_cerebellum)
compute_save_display_mask('brain_aal', indices_aal)


##############
# create each masker with the AAL roi masked with the mask made with all subjects to ensure data is present
##############


##############
# Compute Q matrices per hypothesis and save them
##############
# Testing for all hypotheses
hypotheses = {1: '+gain: equal indiff',
              2: '+gain: equal range',
              3: '+gain: equal indiff',
              4: '+gain: equal range',
              5: '-loss: equal indiff',
              6: '-loss: equal range',
              7: '+loss: equal indiff',
              8: '+loss: equal range',
              9: '+loss:ER>EI'}
hypnums = [1, 2, 5, 6, 7, 8, 9]

masks_for_masker = glob.glob('{}/temp/*mask.nii'.format(results_dir))
masks_for_masker.append('masking/mask_99.nii')
names = [mask_for_masker.split('/')[-1][:-4] for mask_for_masker in masks_for_masker]

for hyp in hypnums:
    unthreshold_maps = [os.path.join(data_path, sub, 'hypo{}_unthresh.nii.gz'.format(hyp)) for sub in subjects]
    if hyp == 9:
        unthreshold_maps.remove('/home/jlefortb/narps_open_pipelines/data/neurovault/4961_K9P0/hypo9_unthresh.nii.gz') # remove weird zmaps
    unthreshold_maps.sort()

    # need to resample to get same affine for each
    unthreshold_maps_resampled = []
    for ind, file in enumerate(unthreshold_maps):
        print("Doing {}/{} for hyp : {}".format(ind, len(unthreshold_maps), hyp))
        # create resampled file
        # ignore nilearn warnings
        # these occur on some of the unthresholded images
        # that contains NaN values
        # we probably don't want to set those to zero
        # because those would enter into interpolation
        # and then would be treated as real zeros later
        # rather than "missing data" which is the usual
        # intention
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resampled_gm = nilearn.image.resample_to_img(
                file,
                load_mni152_gm_template(),
                interpolation='nearest')
        try:
            unthreshold_maps_resampled.append(resampled_gm)
        except:
            print('pb with', file)

    if hyp != 9: # because nothing was significant in hyp 9 so 0 voxel in the mask
        # compute Q with narps mask
        print("masking with narps mask...")
        mask_narps = "/home/jlefortb/IMBA/masking/hyp{}_narps_mask.nii.gz".format(hyp)
        print("nb voxel in mask hyp {} = {}".format(hyp, nibabel.load(mask_narps).get_fdata().sum()))
        masker = nilearn.input_data.NiftiMasker(
                mask_img=mask_narps)
        data = masker.fit_transform(unthreshold_maps_resampled)
        print("Computing Q and saving...")
        Q = numpy.corrcoef(data)      
        numpy.save('{}/temp/Q_narps_mask_hyp{}'.format(results_dir, hyp), Q)

    # compute Q with all other masks
    for ind, mask_for_masker in enumerate(masks_for_masker):
        print("masking...", names[ind])
        masker = nilearn.input_data.NiftiMasker(
            mask_img=mask_for_masker)
        data = masker.fit_transform(unthreshold_maps_resampled)
        print("Computing Q and saving...")
        Q = numpy.corrcoef(data)      
        numpy.save('{}/temp/Q_{}_hyp{}'.format(results_dir, names[ind], hyp), Q)


##############
# Plot Q matrices per hypothesis 
##############
# organized + raw matrices
hypnums = [1, 2, 5, 6, 7, 8, 9]
for hyp in hypnums:
    print(hyp)
    correlation_matrices = glob.glob('{}/temp/Q_*_hyp{}.npy'.format(results_dir, hyp))
    correlation_matrices.sort()
    # put the participant mask at index 0 to fit louvain and sorting according
    # to participant mask and not frontal mask (originaly at index 0)
    new_order = [3, 0, 1, 2, 4, 5, 6] if hyp==9 else [3, 0, 1, 2, 4, 5, 6, 7]
    correlation_matrices = [correlation_matrices[ind] for ind in new_order]

    # load reference matrix (correlation matrix with participant mask) for similarity computation
    matrix_reference_path = '{}/temp/Q_mask_99_hyp{}.npy'.format(results_dir, hyp)
    matrix_reference = numpy.load(matrix_reference_path)


    f, axs = plt.subplots(4, 8, figsize=(25, 15))  
    for ind, matrice in enumerate(correlation_matrices):
        matrix = numpy.load(matrice)
        if ind == 0:
            organised_ind = numpy.argsort(matrix, axis=0)
            partition = utils.compute_louvain_community(numpy.abs(matrix))
        matrix_organized_louvain = utils.reorganize_with_louvain_community(matrix, partition)
        matrix_organized = numpy.take_along_axis(matrix, organised_ind, axis=0)
        if ind < 4:
            row = ind
            col = 0
        else:
            row = ind - 4
            col = 1

        if matrice.split('/')[-1] == "Q_mask_99_hyp{}.npy".format(hyp):
            name_roi = "participant_mask"
        elif matrice.split('/')[-1] == "Q_narps_mask_hyp{}.npy".format(hyp):
            name_roi = "Narps sign mask"
        else:
            name_roi = matrice.split('/')[-1][2:-18]
        title = name_roi + ' ' + str(numpy.round(numpy.mean(numpy.load(matrice))*100, 1))
        title_organized = name_roi


        if matrice != matrix_reference_path:
            # similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(matrix, matrix_reference)
            similarity_matrix = matrix - matrix_reference
            similarity_matrix_ratio = matrix/matrix.shape[0]**2 / matrix_reference/matrix.shape[0]**2
            similarity_matrix_perc_diff = (matrix/matrix.shape[0]**2 - matrix_reference/matrix.shape[0]**2)/matrix_reference/matrix.shape[0]**2
            # Frobenius Norm => (Sum(abs(value)**2))**1/2
            Fro = numpy.linalg.norm(similarity_matrix, ord='fro')
            Fro_div2 = numpy.linalg.norm(similarity_matrix/2, ord='fro')
            Fro_ratio = numpy.linalg.norm(similarity_matrix_ratio, ord='fro')
            Fro_perc_diff = numpy.linalg.norm(similarity_matrix_perc_diff, ord='fro')

            title_similarity = (name_roi 
                + '\n{}|{}|{}|{}'.format(numpy.round(Fro, 1), numpy.round(Fro_div2, 1), numpy.round(Fro_ratio, 1), numpy.round(Fro_perc_diff , 1)))
            seaborn.heatmap(similarity_matrix, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[row, col+6], cbar_kws={'shrink': 0.6})
            axs[row, col+6].title.set_text(title_similarity)


        seaborn.heatmap(matrix, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[row, col], cbar_kws={'shrink': 0.6})
        seaborn.heatmap(matrix_organized, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[row, col+2], cbar_kws={'shrink': 0.6})
        seaborn.heatmap(matrix_organized_louvain, center=0, cmap='coolwarm', robust=True, square=True, ax=axs[row, col+4], cbar_kws={'shrink': 0.6})


        axs[row, col].title.set_text(title)
        axs[row, col+2].title.set_text(title_organized)
        axs[row, col+4].title.set_text(title_organized)
    axs[0, 6].axis('off') # get rid of reference mask used for similarity matrix

    axs[0, 6].text(0.1, 0.7, 'frobenius score as:') 
    axs[0, 6].text(0.1, 0.6, '    a|b|c|d') 
    axs[0, 6].text(0.1, 0.5, 'a: Qi -Qb') 
    axs[0, 6].text(0.1, 0.4, 'b: (Qi-Qb)/2') 
    axs[0, 6].text(0.1, 0.3, 'c: (Qi/K**2)/(Qb/K**2)') 
    axs[0, 6].text(0.1, 0.2, 'd: ((Qi/K**2)-(Qb/K**2))/(Qb/K**2)') 


    if hyp == 9:
        axs[-1, 5].axis('off') # get rid of matrice using mask from narps (it's empty)
        axs[-1, 1].axis('off') # get rid of matrice using mask from narps (it's empty)
        axs[-1, 3].axis('off') # get rid of matrice using mask from narps (it's empty)
        axs[-1, 7].axis('off') # get rid of matrice using mask from narps (it's empty)
    else:
        mask_narps = "/home/jlefortb/IMBA/masking/hyp{}_narps_mask.nii.gz".format(hyp)
        # add nb sign voxel in title for narps sign mask 
        title_nb_voxel = (title 
                        + '\n nb_sign_voxel='
                        + '\n' + str(nibabel.load(mask_narps).get_fdata().sum()))
        axs[0, 1].title.set_text(title_nb_voxel)

    plt.suptitle('hyp  {}'.format(hyp), size=16, fontweight='bold')
    f.subplots_adjust(top=0.78) 
    plt.figtext(0.1,0.95,"Original", va="center", ha="center", size=12, fontweight='bold')
    plt.figtext(0.35,0.95,"Sorted : Intensity", va="center", ha="center", size=12, fontweight='bold')
    plt.figtext(0.6,0.95,"Sorted : Louvain", va="center", ha="center", size=12, fontweight='bold')
    plt.figtext(0.87,0.95,"Similarity matrix", va="center", ha="center", size=12, fontweight='bold')
    line = plt.Line2D((.75,.75),(.1,.9), color="k", linewidth=3)
    f.add_artist(line)
    plt.tight_layout()
    plt.savefig('{}/hyp_{}.png'.format(results_dir, hyp), dpi=300)
    plt.close('all')



files_to_del = glob.glob("{}/temp/*.npy".format(results_dir))
for file in files_to_del:
    os.remove(file)