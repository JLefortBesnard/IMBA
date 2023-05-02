from nilearn.datasets import load_mni152_gm_mask, load_mni152_gm_template
from nilearn.datasets import load_mni152_wm_mask, load_mni152_brain_mask
from nilearn import masking
from nilearn import plotting
from nilearn import image
import glob, os
import matplotlib.pyplot as plt
import warnings
import numpy
import nibabel

##################
# gather all 3D brain data that will be used to compute a mask
##################
subjects = []
for path_to_sub in glob.glob("/home/jlefortb/narps_open_pipelines/data/neurovault/[1-9]*"):
    if os.path.exists(os.path.join(path_to_sub, 'hypo1_unthresh.nii.gz')):
        subjects.append(path_to_sub.split('/')[-1])
data_path = '/home/jlefortb/narps_open_pipelines/data/neurovault'

# check if not mixing thresholded and unthresholded : 4947_X19V
# thresholded IS thersholded

##################
# display all mask next to zmaps to check for weird masking
##################
plt.close('all')
for subject in subjects:
	unthreshold_maps = glob.glob(os.path.join(data_path, '{}/hypo*unthresh.nii.gz'.format(subject)))
	for ind, unthreshold_map in enumerate(unthreshold_maps):
		# create figure and build first map
		if ind == 0:
			print('resampling sub ', subject)
			mask = masking.compute_background_mask(unthreshold_map)
			resampled_map = image.resample_to_img(
		                    mask,
		                    load_mni152_brain_mask(),
		                    interpolation='nearest')
			f, axs = plt.subplots(len(unthreshold_maps)-1, 2, figsize=(10, 30))
			# addmask
			plotting.plot_img(mask, cut_coords=(-21, 0, 9),  figure=f, axes=axs[0, 0])
			axs[0, 0].set_title(unthreshold_map.split('/')[-2:],fontsize=15)
			# add raw zmaps
			plotting.plot_stat_map(nibabel.load(unthreshold_map), cut_coords=(-21, 0, 9),  figure=f, axes=axs[0, 1])
			axs[0, 1].set_title(unthreshold_map.split('/')[-2:],fontsize=15)
			print('*')
		# save the figure
		elif ind == len(unthreshold_maps)-1:
			print('saving...')
			plt.savefig("masking/debugmaps/debugmaps_{}.png".format(subject))
			plt.close('all')
		# add a map
		else:
			mask = masking.compute_background_mask(unthreshold_map)
			resampled_map = image.resample_to_img(
		                    mask,
		                    load_mni152_brain_mask(),
		                    interpolation='nearest')
			# add mask
			plotting.plot_img(mask, cut_coords=(-21, 0, 9),  figure=f, axes=axs[ind, 0])
			axs[ind, 0].set_title(unthreshold_map.split('/')[-2:],fontsize=15)
			# add raw zmaps
			plotting.plot_stat_map(nibabel.load(unthreshold_map), cut_coords=(-21, 0, 9),  figure=f, axes=axs[ind, 1])
			axs[ind, 1].set_title(unthreshold_map.split('/')[-2:],fontsize=15)

##################
# COMPUTE MASK WITHOUT WEIRD TEAM MASK USING MNI GM MASK
##################
masks = []
for ind, subject in enumerate(subjects):
	print(ind, '/', len(subjects), subject)
	for unthreshold_map in glob.glob(os.path.join(data_path, '{}/hypo*_unthresh.nii.gz'.format(subject))):
		# zmaps to remove from mask because weird: 4961_K9P0 hypo 9 only,
		if unthreshold_map == os.path.join(data_path, '4961_K9P0/hypo9_unthresh.nii.gz'):
			print(unthreshold_map, ' weird thus passed')
			continue
		mask = masking.compute_background_mask(unthreshold_map)
		resampled_mask = image.resample_to_img(
			                    mask,
			                    load_mni152_gm_mask(),
			                    interpolation='nearest')
		masks.append(resampled_mask)

##################
# COMPUTE MASK FOR DIFFERENT THRESHOLDS AND DISPLAY IT
##################
plt.close('all')
thresholds = numpy.arange(0.9, 1.01, 0.01)
f, axs = plt.subplots(10, figsize=(8, 20))
for row, t in enumerate(thresholds):
	print(row)
	participants_mask = masking.intersect_masks(masks, threshold=t, connected=True)
	reshape_as_MNI_gm_mask = [participants_mask, load_mni152_gm_mask()]
	participants_mask = masking.intersect_masks(reshape_as_MNI_gm_mask, threshold=1, connected=True)
	plotting.plot_img(participants_mask, cut_coords=(-21, 0, 9),  figure=f, axes=axs[row])
	axs[row].set_title('t={}'.format(int(t*100)),fontsize=12)
plt.savefig("masking/participants_made_mask_90_100.png")
plt.show()


##################
# SAVE MASK COMPUTED WITH SELECTED THRESHOLD (RESHAPED AD MNI GM MASK)
##################
selected_threshold = 0.99
participants_mask = masking.intersect_masks(masks, threshold=selected_threshold, connected=True)
# reshape final as MNI gm mask 
reshape_as_MNI_gm_mask = [participants_mask, load_mni152_gm_mask()]
participants_mask = masking.intersect_masks(reshape_as_MNI_gm_mask , threshold=1, connected=True)
# save the final mask
nibabel.save(participants_mask, "masking/mask_{}.nii".format(int(selected_threshold*100)))
plt.close('all')
plotting.plot_img(participants_mask, cut_coords=(-21, 0, 9))
plt.show()

##################
# COMPUTE MASK USING NARPS RESULTS
##################

# save narps results as nii img
thresh = 0.95
hypnums = [1, 2, 5, 6, 7, 8, 9]
for i, hyp in enumerate(hypnums):
	print(hyp)
	pmap = '/home/jlefortb/narps_open_pipelines/IBMA/results_consensus_analysis/hypo{}_1-fdr.nii.gz'.format(hyp)
	tmap = '/home/jlefortb/narps_open_pipelines/IBMA/results_consensus_analysis/hypo{}_t.nii.gz'.format(hyp)
	pimg = nibabel.load(pmap)
	timg = nibabel.load(tmap)
	pdata = pimg.get_fdata()
	tdata = timg.get_fdata()[:, :, :, 0]
	threshdata = (pdata > thresh)*tdata
	print("Should be ", (pdata > thresh).sum(), " voxels")
	threshimg = nibabel.Nifti1Image(threshdata, affine=timg.affine)
	nibabel.save(threshimg, "masking/hyp{}_narps.nii.gz".format(hyp))
	narps_mask = masking.compute_background_mask(threshimg)
	nibabel.save(narps_mask, "masking/hyp{}_narps_mask.nii.gz".format(hyp))
	print("and is ", narps_mask.get_fdata().sum(), " voxels")
