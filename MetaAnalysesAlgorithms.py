import numpy
import seaborn
import matplotlib.pyplot as plt
import utils
import importlib
import scipy
from os.path import join as opj
from sklearn.preprocessing import (StandardScaler, MinMaxScaler)

importlib.reload(utils) # reupdate imported codes, useful for debugging


def run_all_MA_algorithms(results_dir, matrix_betas):
     K = matrix_betas.shape[0]
     J = matrix_betas.shape[1]

     # generate p values matrix from generated betas values (transform into T, and then p)
     matrix_z = StandardScaler().fit_transform(matrix_betas.T).T # scaling team wise
     matrix_p = 1 - scipy.stats.norm.cdf(matrix_z)

     # generate p values matrix with from left to right less and less significant p values
     # add 5 voxels significant in every studies
     matrix_p_generated = []
     for i in range(1, J+1):
         loc = (i*0.003 + 0.005)
         matrix_p_generated.append(numpy.abs(numpy.random.default_rng().normal(loc=loc, scale=0.02, size=(K))))
     matrix_p_generated = numpy.array(matrix_p_generated).T
     matrix_p_generated.sort(axis=0)

     ###############################################################################
     # PART. 1
     # Minimal Data Needed for Valid & Accurate Image-Based fMRI Meta-Analysis 
     # Camille Maumet, Thomas Nichols 

     # UNCONDITIONAL FRAMEWORK : referencing the sampling variability if this experiment 
     # could be repeated and a new datasets sampled from the population).
     ###############################################################################


     ##################
     ### FFX GLM
     ##################
     # INPUTS ARE BETA VALUES !
     # final stats is sum(Bi/si2)/sqrt(sum(1/si2))

     # W = s**2
     W_FFX = numpy.var(matrix_betas, axis=0) # variance of the contrast estimate for each voxel
     deltas, deltas_var = utils.compute_deltas(matrix_betas, W_FFX, GLM='FFX')
     # T threshold at k-1 = 2.262 (should be (sum_at_k ni-1)-1)
     T_map = utils.meta_analyse_FFX(deltas, deltas_var)
     utils.plot_results('1.FFX', matrix_betas, deltas, deltas_var, T_map, W_FFX, results_dir)


     ##################
     ### MFX GLM
     ##################
     # INPUTS ARE BETA VALUES !
     # final stats is sum(ki*Bi)/sqrt(sum(Ki)) with ki = 1/(tau2 + si2)

     # W = s**2 + t**2
     W_MFX = numpy.var(matrix_betas, axis=0) +  utils.tau(matrix_betas) # variance of the contrast estimate for each voxel
     deltas, deltas_var = utils.compute_deltas(matrix_betas, W_MFX, GLM='MFX')
     # T threshold at k-1 = 2.262
     T_map = utils.meta_analyse_MFX(deltas, deltas_var)
     utils.plot_results('2.MFX', matrix_betas, deltas, deltas_var, T_map, W_MFX, results_dir)


     ##################
     ### RFX GLM
     ##################
     # INPUTS ARE BETA VALUES !
     # final stats is sum(Bi/sqrt(k))/(tau2+s2)

     # If the si2 are unavailable, the contrast estimates βi can be combined by 
     # assuming that the within-study contrast variance is negligible in 
     # comparison to the between-study variance
     # then σ2 combines the within and between-study variances, i.e. σ2 = Tau2
     W_RFX = utils.tau(matrix_betas) 
     # or that σ2 i /ni is constant (σ2 i /ni = σ2 ∀i)
     # then σ2 = Tau2 + s2
     W_RFX = utils.tau(matrix_betas) + numpy.var(matrix_betas, axis=0)
     deltas, deltas_var = utils.compute_deltas(matrix_betas, W_RFX, GLM='RFX')
     # T threshold at k-1 = 2.262
     T_map = utils.meta_analyse_RFX(deltas, deltas_var)
     utils.plot_results('3.RFX', matrix_betas, deltas, deltas_var, T_map, W_RFX, results_dir)


     ##################
     ### Contrast Perm
     ##################
     # INPUTS ARE BETA VALUES !
     # final stats is sum(Bi/sqrt(k))/(tau2+s2)


     ##################
     ### Fisher FFX
     ##################
     # INPUTS ARE P VALUES !
     # final stats is -2 sum(log(pi))

     ## With data generated to have decreasing amount of singificant p-values
     #---------------------
     vector_fisher = utils.meta_analyse_Fisher(matrix_p_generated)

     # K=10 thus Chi2 threshold at 2k(20) => 31.41
     f, axs = plt.subplots(4, figsize=(5, 8))
     seaborn.heatmap(matrix_p_generated[:, :50], vmin=0, cmap='Reds_r', ax=axs[0], cbar_kws={'shrink': 0.5})
     axs[0].title.set_text("generated p values")
     axs[0].set_ylabel("K teams")
     axs[0].set_xlabel('J Voxels')
     seaborn.heatmap(matrix_p_generated[:, :50], vmin=0, vmax=0.06, cmap='Reds_r', mask=matrix_p_generated[:, :50] > 0.05, ax=axs[1],cbar_kws={'shrink': 0.5})
     axs[1].title.set_text("significant p values")
     axs[1].set_ylabel("K teams")
     axs[1].set_xlabel('J Voxels')
     seaborn.heatmap(matrix_p_generated[:, :50].mean(axis=0).reshape(1, -1), cmap='Reds_r', square=True, ax=axs[2],cbar_kws={'shrink': 0.5})
     axs[2].title.set_text("Mean p values")
     axs[2].set_ylabel("p")
     axs[2].set_xlabel('J Voxels')
     seaborn.heatmap(vector_fisher.reshape(1, -1)[:, :50], cmap='Reds', square=True, mask=vector_fisher.reshape(1, -1)[:, :50] < 55.758, ax=axs[3],cbar_kws={'shrink': 0.5})
     ratio = (vector_fisher>=55.758).sum()/vector_fisher.reshape(-1).__len__()
     lim = 2*numpy.sqrt(0.05*(1-0.05)/20000)
     verdict = 0.05-lim <= ratio <= 0.05+lim
     axs[3].title.set_text("Sig X2 >= 55.758, ratio={}, {}".format(numpy.round(ratio, 3), verdict))
     axs[3].set_ylabel("X2")
     axs[3].set_xlabel('J Voxels')

     plt.suptitle('Results for Fisher with generated p')
     plt.tight_layout()
     plt.savefig(opj(results_dir, "4a.Fisher_decreasing_pvalues_maps.png"))
     plt.close('all')


     ## With same data as in the other simulations
     #---------------------
     vector_fisher = utils.meta_analyse_Fisher(matrix_p)

     # K=20 thus Chi2 threshold at 2k (40) => 55.758
     f, axs = plt.subplots(4, figsize=(5, 8))
     seaborn.heatmap(matrix_p[:, :50], vmin=0, cmap='Reds_r', ax=axs[0], cbar_kws={'shrink': 0.5})
     axs[0].title.set_text("p values (from standardized Betas)")
     axs[0].set_ylabel("K teams")
     axs[0].set_xlabel('J Voxels')
     seaborn.heatmap(matrix_p[:, :50], vmin=0, vmax=0.06, cmap='Reds_r', mask=matrix_p[:, :50] > 0.05, ax=axs[1],cbar_kws={'shrink': 0.5})
     axs[1].title.set_text("significant p values")
     axs[1].set_ylabel("K teams")
     axs[1].set_xlabel('J Voxels')
     seaborn.heatmap(matrix_p.mean(axis=0).reshape(1, -1)[:, :50], cmap='Reds_r', square=True, ax=axs[2],cbar_kws={'shrink': 0.5})
     axs[2].title.set_text("Mean p values")
     axs[2].set_ylabel("p")
     axs[2].set_xlabel('J Voxels')
     seaborn.heatmap(vector_fisher.reshape(1, -1)[:, :50], cmap='Reds', square=True, mask=vector_fisher.reshape(1, -1)[:, :50] < 55.758, ax=axs[3],cbar_kws={'shrink': 0.5})
     ratio = (vector_fisher>=55.758).sum()/vector_fisher.reshape(-1).__len__()
     lim = 2*numpy.sqrt(0.05*(1-0.05)/20000)
     verdict = 0.05-lim <= ratio <= 0.05+lim
     axs[3].title.set_text("Sig X2 >= 55.758, ratio={}, {}".format(numpy.round(ratio, 3), verdict))
     axs[3].set_ylabel("X2")
     axs[3].set_xlabel('J Voxels')

     plt.suptitle('Results for Fisher')
     plt.tight_layout()
     plt.savefig(opj(results_dir, "4b.Fisher_maps.png"))
     plt.close('all')


     ##################
     ### Stouffer (adapted Fisher) FFX
     ##################
     # final stats is sqrt(k)*(sum(Zi)/k
     vector_stouffer = utils.meta_analyse_Stouffer(matrix_z)
     stouffer_p = 1 - scipy.stats.norm.cdf(vector_stouffer)

     f, axs = plt.subplots(4, figsize=(5, 8))
     seaborn.heatmap(matrix_z[:, :50], center=0, cmap='coolwarm', ax=axs[0],cbar_kws={'shrink': 0.5})
     axs[0].title.set_text("Z values (Standardized Betas)")
     axs[0].set_ylabel("K teams")
     axs[0].set_xlabel('J Voxels')
     seaborn.heatmap(matrix_z[:, :50].mean(axis=0).reshape(1, -1), center=0, cmap='coolwarm', square=True, ax=axs[1],cbar_kws={'shrink': 0.5})
     axs[1].title.set_text("Mean Z values")
     axs[1].set_ylabel("Z")
     axs[1].set_xlabel('J Voxels')
     seaborn.heatmap(vector_stouffer.reshape(1, -1)[:, :50], center=0, cmap='coolwarm', square=True, ax=axs[2],cbar_kws={'shrink': 0.5})
     axs[2].title.set_text("Stouffer Z values")
     axs[2].set_ylabel("Z")
     axs[2].set_xlabel('J Voxels')
     seaborn.heatmap(stouffer_p.reshape(1, -1)[:, :50], vmin=0, vmax=0.06, cmap='Reds_r', square=True, mask=stouffer_p.reshape(1, -1)[:, :50] > 0.05, ax=axs[3],cbar_kws={'shrink': 0.5})
     ratio = (stouffer_p<=0.05).sum()/stouffer_p.reshape(-1).__len__()
     lim = 2*numpy.sqrt(0.05*(1-0.05)/20000)
     verdict = 0.05-lim <= ratio <= 0.05+lim
     axs[3].title.set_text("Sign p values, ratio={}, {}".format(numpy.round(ratio, 3), verdict))
     axs[3].set_ylabel("p")
     axs[3].set_xlabel('J Voxels')

     plt.suptitle('Results for Stouffer')
     plt.tight_layout()
     plt.savefig(opj(results_dir, "5.Stouffer_maps.png"))
     plt.close('all')


     ##################
     ### Weighted Z (Weighted Stouffer)
     ##################
     # final stats is 1/sum(ni) * sum(sqrt(ni)*Zi)


     ##################
     ### Z RFX
     ##################
     # final stats is sum(Zi)/sqrt(k*var)


     ##################
     ### Z perm
     ##################
     # final stats is sum(Zi)/sqrt(k)








     ###############################################################################
     # PART. 2
     # Same Data Meta-Analysis Notes
     # Thomas Nichols, Jean-Baptiste Poline

     # CONDITIONAL FRAMEWORK : considering the original data as fixed
     ###############################################################################

     matrix_z = matrix_betas.copy()
     matrix_z = matrix_z.T # j rows, k columns

     ##################
     ### Intuitive solution
     ##################

     Q = numpy.corrcoef(matrix_z.T) # shape K*K
     Intuitive_solution = matrix_z.sum(axis=1)/matrix_z.shape[1]
     Intuitive_solution_var = numpy.ones(K).dot(Q)/K**2

     ##################
     ### consensus (scaling before)
     ##################

     # variance of each pipeline is assumed to be equal but allowed to vary over space

     # compute a standardized map Z∗ k for each pipeline k
     scaler = StandardScaler()
     Z_star_k = scaler.fit_transform(matrix_z) # no Transpose as in Stouffer, because already shape (J,K)
     # z* = (z - z_mean) / s 
     # with s = image-wise std for pipeline k
     # with z_mean = image-wise mean for pipeline k
     # numpy.divide(numpy.subtract(matrix_z, matrix_z.mean(axis=0)), matrix_z.std(axis=0))

     # These standardized maps are averaged over pipelines to create a map Z∗
     Z_star_mean_j = Z_star_k.mean(axis=1) # shape J

     # image-wise mean of this average of standard maps is zero
     assert Z_star_mean_j.mean() < 0.0001

     # and which is finally standardized, scaled and shifted 
     # to the consensus standard deviation and mean
     consensus_var = matrix_z.var(axis=0).sum() / K # scalar
     consensus_std = numpy.sqrt(consensus_var) # scalar
     consensus_mean = matrix_z.mean(axis=0).sum() / K # scalar
     attenuated_variance = numpy.divide(numpy.ones(K).T.dot(Q).dot(numpy.ones(K)), K**2) # shape K
     attenuated_std = numpy.sqrt(attenuated_variance)
     Z_star_consensus = numpy.divide(Z_star_mean_j.reshape(-1, 1), attenuated_std.reshape(1, -1)).dot(consensus_std) + consensus_mean
     p = 1 - scipy.stats.norm.cdf(Z_star_consensus)

     f, axs = plt.subplots(5, figsize=(5, 8))
     seaborn.heatmap(matrix_z.T[:, :50], center=0, cmap='coolwarm', ax=axs[0],cbar_kws={'shrink': 0.5})
     axs[0].title.set_text("Original Z values")
     axs[0].set_ylabel("K teams")
     axs[0].set_xlabel('J Voxels')
     # debugging
     # a = Z_star_k.T
     # a.sort()
     seaborn.heatmap(Z_star_k.T[:, :50], center=0, cmap='coolwarm', ax=axs[1],cbar_kws={'shrink': 0.5})
     axs[1].title.set_text("Standardized Z values")
     axs[1].set_ylabel("K teams")
     axs[1].set_xlabel('J Voxels')
     seaborn.heatmap(matrix_z.mean(axis=1).reshape(1, -1)[:, :50], center=0, cmap='coolwarm', square=True, ax=axs[2],cbar_kws={'shrink': 0.5})
     axs[2].title.set_text("Mean Z values")
     axs[2].set_ylabel("Z")
     axs[2].set_xlabel('J Voxels')
     seaborn.heatmap(Z_star_consensus.reshape(1, -1)[:, :50], center=0, cmap='coolwarm', square=True, ax=axs[3],cbar_kws={'shrink': 0.5})
     axs[3].title.set_text("T values")
     axs[3].set_ylabel("T")
     axs[3].set_xlabel('J Voxels')
     seaborn.heatmap(p.reshape(1, -1)[:, :50], vmin=0, cmap='Reds_r', square=True, mask=p.reshape(1, -1)[:, :50] > 0.05, ax=axs[4],cbar_kws={'shrink': 0.5})
     ratio = (p<=0.05).sum()/p.reshape(-1).__len__()
     lim = 2*numpy.sqrt(0.05*(1-0.05)/20000)
     verdict = 0.05-lim <= ratio <= 0.05+lim
     axs[4].title.set_text("Sign p values, ratio={}, {}".format(numpy.round(ratio, 3), verdict))
     axs[4].set_ylabel("p")
     axs[4].set_xlabel('J Voxels')
     plt.suptitle('Results for consensus')
     plt.tight_layout()
     plt.savefig(opj(results_dir, "6.Consensus_maps.png"))
     plt.close('all')


     ##################
     ### NARPS code consensus (scaling supposed to be done, but instead res_var used in the equation)
     ##################
     res_mean = numpy.mean(matrix_betas)
     res_var = numpy.mean(numpy.var(matrix_betas, 1))

     X = numpy.ones((K, 1))
     Q0 = numpy.corrcoef(matrix_betas)

     VarMean = res_var * X.T.dot(Q0).dot(X) / K**2

     # T  =  mean(y,0)/s-hat-2
     # use diag to get s_hat2 for each variable
     T = (numpy.mean(matrix_betas, 0) - res_mean
          )/numpy.sqrt(VarMean)*numpy.sqrt(res_var) + res_mean

     # Assuming variance is estimated on whole image
     # and assuming infinite df
     p = 1 - scipy.stats.norm.cdf(T)


     f, axs = plt.subplots(4, figsize=(5, 8))
     seaborn.heatmap(matrix_betas[:, :50], center=0, cmap='coolwarm', ax=axs[0],cbar_kws={'shrink': 0.5})
     axs[0].title.set_text("Original Z values")
     axs[0].set_ylabel("K teams")
     axs[0].set_xlabel('J Voxels')
     seaborn.heatmap(matrix_betas.mean(axis=0).reshape(1, -1)[:, :50], center=0, cmap='coolwarm', square=True, ax=axs[1],cbar_kws={'shrink': 0.5})
     axs[1].title.set_text("Mean Z values")
     axs[1].set_ylabel("Z")
     axs[1].set_xlabel('J Voxels')
     seaborn.heatmap(T.reshape(1, -1)[:, :50], center=0, cmap='coolwarm', square=True, ax=axs[2],cbar_kws={'shrink': 0.5})
     axs[2].title.set_text("T values")
     axs[2].set_ylabel("T")
     axs[2].set_xlabel('J Voxels')
     seaborn.heatmap(p.reshape(1, -1)[:, :50], vmin=0, cmap='Reds_r', square=True, mask=p.reshape(1, -1)[:, :50] > 0.05, ax=axs[3],cbar_kws={'shrink': 0.5})
     ratio = (p<=0.05).sum()/p.reshape(-1).__len__()
     lim = 2*numpy.sqrt(0.05*(1-0.05)/20000)
     verdict = 0.05-lim <= ratio <= 0.05+lim
     axs[3].title.set_text("Sign p values, ratio={}, {}".format(numpy.round(ratio, 3), verdict))
     axs[3].set_ylabel("p")
     axs[3].set_xlabel('J Voxels')
     plt.suptitle('Results for Narps consensus')
     plt.tight_layout()
     plt.savefig(opj(results_dir, "7.ConsensusNarps_maps.png"))
     plt.close('all')


     ##################
     ### same data consensus meta analysis (no scaling)
     ##################
     res_mean = numpy.mean(matrix_betas)
     res_var = numpy.mean(numpy.var(matrix_betas, 1))

     X = numpy.ones((K, 1))
     Q0 = numpy.corrcoef(matrix_betas)

     VarMean = X.T.dot(Q0).dot(X) / K**2

     # T  =  mean(y,0)/s-hat-2
     # use diag to get s_hat2 for each variable
     T = (numpy.mean(matrix_betas, 0) - res_mean
          )/numpy.sqrt(VarMean) + res_mean

     # Assuming variance is estimated on whole image
     # and assuming infinite df
     p = 1 - scipy.stats.norm.cdf(T)


     f, axs = plt.subplots(4, figsize=(5, 8))
     seaborn.heatmap(matrix_betas[:, :50], center=0, cmap='coolwarm', ax=axs[0],cbar_kws={'shrink': 0.5})
     axs[0].title.set_text("Original Z values")
     axs[0].set_ylabel("K teams")
     axs[0].set_xlabel('J Voxels')
     seaborn.heatmap(matrix_betas.mean(axis=0).reshape(1, -1)[:, :50], center=0, cmap='coolwarm', square=True, ax=axs[1],cbar_kws={'shrink': 0.5})
     axs[1].title.set_text("Mean Z values")
     axs[1].set_ylabel("Z")
     axs[1].set_xlabel('J Voxels')
     seaborn.heatmap(T.reshape(1, -1)[:, :50], center=0, cmap='coolwarm', square=True, ax=axs[2],cbar_kws={'shrink': 0.5})
     axs[2].title.set_text("T values")
     axs[2].set_ylabel("T")
     axs[2].set_xlabel('J Voxels')
     seaborn.heatmap(p.reshape(1, -1)[:, :50], vmin=0, cmap='Reds_r', square=True, mask=p.reshape(1, -1)[:, :50] > 0.05, ax=axs[3],cbar_kws={'shrink': 0.5})
     ratio = (p<=0.05).sum()/p.reshape(-1).__len__()
     lim = 2*numpy.sqrt(0.05*(1-0.05)/20000)
     verdict = 0.05-lim <= ratio <= 0.05+lim
     axs[3].title.set_text("Sign p values, ratio={}, {}".format(numpy.round(ratio, 3), verdict))
     axs[3].set_ylabel("p")
     axs[3].set_xlabel('J Voxels')
     plt.suptitle('Results for Same Data consensus')
     plt.tight_layout()
     plt.savefig(opj(results_dir, "8.SameDataConsensus_maps.png"))
     plt.close('all')


     ##################
     ### Same data fixed effects meta analysis
     ##################
     res_mean = numpy.mean(matrix_betas)
     res_var = numpy.mean(numpy.var(matrix_betas, 1))

     X = numpy.ones((K, 1))
     Q0 = numpy.corrcoef(matrix_betas)

     VarMean = X.T.dot(Q0).dot(X) / K**2

     # T  =  mean(y,0)/s-hat-2
     # use diag to get s_hat2 for each variable
     T = numpy.mean(matrix_betas, 0)/numpy.sqrt(VarMean)

     # Assuming variance is estimated on whole image
     # and assuming infinite df
     p = 1 - scipy.stats.norm.cdf(T)


     f, axs = plt.subplots(4, figsize=(5, 8))
     seaborn.heatmap(matrix_betas[:, :50], center=0, cmap='coolwarm', ax=axs[0],cbar_kws={'shrink': 0.5})
     axs[0].title.set_text("Original Z values")
     axs[0].set_ylabel("K teams")
     axs[0].set_xlabel('J Voxels')
     seaborn.heatmap(matrix_betas.mean(axis=0).reshape(1, -1)[:, :50], center=0, cmap='coolwarm', square=True, ax=axs[1],cbar_kws={'shrink': 0.5})
     axs[1].title.set_text("Mean Z values")
     axs[1].set_ylabel("Z")
     axs[1].set_xlabel('J Voxels')
     seaborn.heatmap(T.reshape(1, -1)[:, :50], center=0, cmap='coolwarm', square=True, ax=axs[2],cbar_kws={'shrink': 0.5})
     axs[2].title.set_text("T values")
     axs[2].set_ylabel("T")
     axs[2].set_xlabel('J Voxels')
     seaborn.heatmap(p.reshape(1, -1)[:, :50], vmin=0, cmap='Reds_r', square=True, mask=p.reshape(1, -1)[:, :50] > 0.05, ax=axs[3],cbar_kws={'shrink': 0.5})
     ratio = (p<=0.05).sum()/p.reshape(-1).__len__()
     lim = 2*numpy.sqrt(0.05*(1-0.05)/20000)
     verdict = 0.05-lim <= ratio <= 0.05+lim
     axs[3].title.set_text("Sign p values, ratio={}, {}".format(numpy.round(ratio, 3), verdict))
     axs[3].set_ylabel("p")
     axs[3].set_xlabel('J Voxels')
     plt.suptitle('Results for Same data FX')
     plt.tight_layout()
     plt.savefig(opj(results_dir, "9.SameDataFX_maps.png"))
     plt.close('all')



     ##################
     ### Same data random effects meta analysis
     ##################

     X = numpy.ones((K, 1))
     # no idea how to compute Q1 thus using Q0 for now
     Q0 = numpy.corrcoef(matrix_betas)
     Q1 = Q0.copy()

     # # compute random effect Q (Tau2*Q)
     # R = numpy.eye(K) - numpy.ones((K, 1)).dot(numpy.ones((1, K)))/K
     # sampvar_est = numpy.trace(R.dot(Q0))

     # tau2 = numpy.zeros(J)
     # for j in range(J): # voxel wise
     #    Y = matrix_betas[:, j] # specific voxel value for each team
     #    tau2[j] = (1/sampvar_est)*Y.T.dot(R).dot(Y) # Tau2 for a specific voxel
     # tau2 = numpy.sqrt(tau2)

     # Q1 = tau2*Q0

     VarMean = X.T.dot(Q1).dot(X) / K**2

     # T  =  mean(y,0)/s-hat-2
     # use diag to get s_hat2 for each variable
     T = numpy.mean(matrix_betas, 0)/numpy.sqrt(VarMean)

     # Assuming variance is estimated on whole image
     # and assuming infinite df
     p = 1 - scipy.stats.norm.cdf(T)


     f, axs = plt.subplots(4, figsize=(5, 8))
     seaborn.heatmap(matrix_betas[:, :50], center=0, cmap='coolwarm', ax=axs[0],cbar_kws={'shrink': 0.5})
     axs[0].title.set_text("Original Z values")
     axs[0].set_ylabel("K teams")
     axs[0].set_xlabel('J Voxels')
     seaborn.heatmap(matrix_betas.mean(axis=0).reshape(1, -1)[:, :50], center=0, cmap='coolwarm', square=True, ax=axs[1],cbar_kws={'shrink': 0.5})
     axs[1].title.set_text("Mean Z values")
     axs[1].set_ylabel("Z")
     axs[1].set_xlabel('J Voxels')
     seaborn.heatmap(T.reshape(1, -1)[:, :50], center=0, cmap='coolwarm', square=True, ax=axs[2],cbar_kws={'shrink': 0.5})
     axs[2].title.set_text("T values")
     axs[2].set_ylabel("T")
     axs[2].set_xlabel('J Voxels')
     seaborn.heatmap(p.reshape(1, -1)[:, :50], vmin=0, cmap='Reds_r', square=True, mask=p.reshape(1, -1)[:, :50] > 0.05, ax=axs[3],cbar_kws={'shrink': 0.5})
     ratio = (p<=0.05).sum()/p.reshape(-1).__len__()
     lim = 2*numpy.sqrt(0.05*(1-0.05)/20000)
     verdict = 0.05-lim <= ratio <= 0.05+lim
     axs[3].title.set_text("Sign p values, ratio={}, {}".format(numpy.round(ratio, 3), verdict))
     axs[3].set_ylabel("p")
     axs[3].set_xlabel('J Voxels')
     plt.suptitle('Results for Random data FX')
     plt.tight_layout()
     plt.savefig(opj(results_dir, "10.RandomDataFX_maps.png"))
     plt.close('all')

     print("** ENDED WELL **")



if __name__ == "__main__":
   print('This file is intented to be used as imported only')