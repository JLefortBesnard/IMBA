
import os
import sys
import argparse
import glob
import numpy
import nibabel
import scipy
import nilearn.plotting
import nilearn.input_data
import warnings
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from nilearn.datasets import load_mni152_gm_mask, load_mni152_gm_template

# code from https://github.com/poldrack/narps/blob/master/ImageAnalyses/ConsensusAnalysis.py
# compute tau^2 per Tom's notes in CorrelatedMetaNotes.html
def tau(data, Q):
    n = data.shape[0]
    R = numpy.eye(n) - numpy.ones((n, 1)).dot(numpy.ones((1, n)))/n
    sampvar_est = numpy.trace(R.dot(Q))
    tau2 = numpy.zeros(data.shape[1])
    for i in range(data.shape[1]):
        Y = data[:, i]
        tau2[i] = (1/sampvar_est)*Y.T.dot(R).dot(Y)
    return(numpy.sqrt(tau2))

def t_corr(y, res_mean=None, res_var=None, Q=None):
    """
    perform a one-sample t-test on correlated data
    y = data (n observations X n vars)
    res_mean = Common mean over voxels and results
    res_var  = Common variance over voxels and results
    Q = "known" correlation across observations
    - (use empirical correlation based on maps)
    """

    npts = y.shape[0]
    X = numpy.ones((npts, 1))

    if res_mean is None:
        res_mean = 0

    if res_var is None:
        res_var = 1

    if Q is None:
        Q = numpy.eye(npts)

    VarMean = res_var * X.T.dot(Q).dot(X) / npts**2

    # T  =  mean(y,0)/s-hat-2
    # use diag to get s_hat2 for each variable
    T = (numpy.mean(y, 0)-res_mean
         )/numpy.sqrt(VarMean)*numpy.sqrt(res_var) + res_mean

    # Assuming variance is estimated on whole image
    # and assuming infinite df
    p = 1 - scipy.stats.norm.cdf(T)

    return(T, p)


subjects = []
for path_to_sub in glob.glob("/home/jlefortb/narps_open_pipelines/data/neurovault/[1-9]*"):
    if os.path.exists(os.path.join(path_to_sub, 'hypo1_unthresh.nii.gz')):
        subjects.append(path_to_sub.split('/')[-1])


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

data_path = '/home/jlefortb/narps_open_pipelines/data/neurovault'

masker = nilearn.input_data.NiftiMasker(
    mask_img=load_mni152_gm_mask())
results_dir = 'results_consensus_analysis'

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

if not os.path.exists(os.path.join(results_dir, subjects[0])):
    for sub in subjects:
        os.mkdir(os.path.join(results_dir, sub))


for hyp in hypnums:
    print('running consensus analysis for hypothesis', hyp)

    unthreshold_maps = [os.path.join(data_path, sub, 'hypo{}_unthresh.nii.gz'.format(hyp)) for sub in subjects]
    unthreshold_maps.sort()

    # code from https://github.com/poldrack/narps/blob/master/ImageAnalyses/narps.py

    # resample images with mni
    # use linear interpolation for binarized maps, then threshold at 0.5
    # this avoids empty voxels that can occur with NN interpolation
    interp_type = {'thresh': 'linear', 'unthresh': 'continuous'}
    unthreshold_maps_resampled = []
    for ind, file in enumerate(unthreshold_maps):
        print("Doing {}/{}".format(ind, len(unthreshold_maps)))
        sub_numb = file.split('/')[-2]
        file_name = 'resample_' +  file.split('/')[-1]
        outfile = '{}/{}'.format(sub_numb, file_name)
        outfile = os.path.join(results_dir, outfile)

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
            resampled = nilearn.image.resample_to_img(
                file,
                load_mni152_gm_template(),
                interpolation=interp_type['unthresh'])
        resampled.to_filename(outfile)
        try:
            nibabel.load(outfile).get_data().shape
            unthreshold_maps_resampled.append(outfile)
        except:
            print('pb with', outfile)

    data = masker.fit_transform(unthreshold_maps_resampled)

    # get estimated mean, variance, and correlation for t_corr
    img_mean = numpy.mean(data)
    img_var = numpy.mean(numpy.var(data, 1))
    cc = numpy.corrcoef(data)
    mean_cc = numpy.mean(cc[numpy.triu_indices_from(cc, 1)])

    tau_est = tau(data, cc)
    tauimg = masker.inverse_transform(tau_est)
    tauimg.to_filename(os.path.join(
        results_dir,
        'hypo{}_tau.nii.gz'.format(hyp)))

    # perform t-test
    tvals, pvals = t_corr(data,
                          res_mean=img_mean,
                          res_var=img_var,
                          Q=cc)

    # Empty memory/hard space
    data = 'emptied'
    for tempfile in glob.glob("results_consensus_analysis/*/*.nii.gz"):
        os.remove(tempfile)

    # move back into image format
    timg = masker.inverse_transform(tvals)
    timg.to_filename(os.path.join(results_dir, 'hypo{}_t.nii.gz'.format(hyp)))
    pimg = masker.inverse_transform(1-pvals)
    pimg.to_filename(os.path.join(results_dir, 'hypo{}_1-p.nii.gz'.format(hyp)))
    fdr_results = multipletests(pvals[0, :], 0.05, 'fdr_tsbh')
    fdrimg = masker.inverse_transform(1 - fdr_results[1])
    fdrimg.to_filename(os.path.join(
        results_dir,
        'hypo{}_1-fdr.nii.gz'.format(hyp)))


    
fig, ax = plt.subplots(7, 1, figsize=(12, 24))
cut_coords = [-24, -10, 4, 18, 32, 52, 64]
thresh = 0.95

for i, hyp in enumerate(hypnums):
    pmap = os.path.join(
        results_dir,
        'hypo{}_1-fdr.nii.gz'.format(hyp))
    tmap = os.path.join(
        results_dir,
        'hypo{}_t.nii.gz'.format(hyp))
    pimg = nibabel.load(pmap)
    timg = nibabel.load(tmap)
    pdata = pimg.get_fdata()
    tdata = timg.get_fdata()[:, :, :, 0]
    threshdata = (pdata > thresh)*tdata
    threshimg = nibabel.Nifti1Image(threshdata, affine=timg.affine)
    nilearn.plotting.plot_stat_map(
        threshimg,
        threshold=0.1,
        display_mode="z",
        colorbar=True,
        title='hyp {}: {}'.format(hyp, hypotheses[hyp]),
        vmax=8,
        cmap='jet',
        cut_coords=cut_coords,
        axes=ax[i])

plt.savefig(os.path.join(
    results_dir,
    'consensus_map.pdf'), bbox_inches='tight')
plt.close(fig)

# create tau figures
fig, ax = plt.subplots(7, 1, figsize=(12, 24))
tauhist = {}
for i, hyp in enumerate(hypnums):
    taumap = os.path.join(
        results_dir,
        'hypo{}_tau.nii.gz'.format(hyp))
    tauimg = nibabel.load(taumap)
    taudata = masker.fit_transform(tauimg)

    tauhist[i] = numpy.histogram(
        taudata, bins=numpy.arange(0, 5, 0.01))
    nilearn.plotting.plot_stat_map(
        tauimg,
        threshold=0.0,
        display_mode="z",
        colorbar=True,
        title='hyp {}: {}'.format(hyp, hypotheses[hyp]),
        vmax=4.,
        cmap='jet',
        cut_coords=cut_coords,
        axes=ax[i])
plt.savefig(os.path.join(
    results_dir,
    'tau_maps.pdf'), bbox_inches='tight')
plt.close(fig)

# create tau histograms
fig, ax = plt.subplots(7, 1, figsize=(12, 24))
for i, hyp in enumerate(hypnums):
    ax[i].plot(tauhist[i][1][1:], tauhist[i][0])
    ax[i].set_xlabel('tau')
    ax[i].set_ylabel('# of voxels')
    ax[i].set_title('hyp {}: {}'.format(hyp, hypotheses[hyp]))
    plt.tight_layout()
plt.savefig(os.path.join(
    results_dir,
    'tau_histograms.pdf'), bbox_inches='tight')
plt.close(fig)