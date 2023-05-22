import numpy
import seaborn
import matplotlib.pyplot as plt
import utils
import importlib
import scipy
from os.path import join as opj
from sklearn.preprocessing import (StandardScaler, MinMaxScaler)

importlib.reload(utils) # reupdate imported codes, useful for debugging


def run_all_MA_algorithms(matrix_betas, simulation_nb):


     K = matrix_betas.shape[0]
     J = matrix_betas.shape[1]

     # generate p values matrix from generated betas values (transform into T, and then p)
     matrix_z = StandardScaler().fit_transform(matrix_betas.T).T # scaling team wise and back to normal shape
     matrix_p = 1 - scipy.stats.norm.cdf(matrix_z)

     # store results for latter plotting
     results_simulation = {} 
     results_simulation['data'] = {'matrix_betas':matrix_betas, 'matrix_z':matrix_z, 'matrix_p':matrix_p}



     ###############################################################################
     # PART. 1
     # Minimal Data Needed for Valid & Accurate Image-Based fMRI Meta-Analysis 
     # Camille Maumet, Thomas Nichols 

     # UNCONDITIONAL FRAMEWORK : referencing the sampling variability if this experiment 
     # could be repeated and a new datasets sampled from the population).
     ###############################################################################

     ##################
     ### FFX GLM si2 = matrix of 1s
     ##################
     # INPUTS ARE BETA VALUES !
     # final stats is sum(Bi/si2)/sqrt(sum(1/si2))

     # si2 = variance of contrast estimate for study i
     # matrix of ones ONLY if within study variance is perfectly known
     si2 = numpy.ones((K, J))

     # compute meta-analytic statistics 
     T_map = utils.meta_analyse_FFX(matrix_betas, si2)
     T_map = T_map.reshape(-1)
     # compute p-values for inference
     # df = (ni - 1)*K -1 with ni = 40, K=20
     FFX_ns_pvalues = {}
     ns = [20, 40, 60, 80, 100] # compute this for latter comparison
     for n in ns:
          p_values = 1 - scipy.stats.t.cdf(T_map, df=(n - 1)*K -1)
          p_values = p_values.reshape(-1)
          FFX_ns_pvalues[n] = p_values
     # df = (ni - 1)*K -1 with ni = 20, K=20
     p_values = 1 - scipy.stats.t.cdf(T_map, df=379)
     p_values = p_values.reshape(-1)
     # compute ratio of significant p-values
     ratio_significance_raw = (p_values<=0.05).sum()/len(p_values)
     ratio_significance = numpy.round(ratio_significance_raw*100, 4)
     # for simulation 0 only, check if ratio roughly = 5%
     lim = 2*numpy.sqrt(0.05*(1-0.05)/J)
     verdict = 0.05-lim <= ratio_significance_raw <= 0.05+lim
     # save results
     results_simulation['FFX'] = {'T_map':T_map, 'p_values':p_values, 'ratio_significance':ratio_significance, 'verdict':verdict, 'si2':si2, 'distrib':'T distribution, df=379 (sum(ni-1)-1)'}



     ##################
     ### FFX GLM si2 => X2 with df = n-1
     ##################
     # si2 = variance of contrast estimate for study i
     # matrix of random values with a X2 distribution with df = ni-1
     ns = [20, 40, 60, 80, 100]
     for n in ns:
          si2 = numpy.array([scipy.stats.chi2.rvs(size=J, df=n-1)/100 for i in range(K)])
          # compute meta-analytic statistics 
          T_map = utils.meta_analyse_FFX(matrix_betas, si2)
          T_map = T_map.reshape(-1)
          # compute p-values for inference
          p_values = 1 - scipy.stats.t.cdf(T_map, df=(n - 1)*K -1 )
          p_values = p_values.reshape(-1)
          # compute ratio of significant p-values
          ratio_significance_raw = (p_values<=0.05).sum()/len(p_values)
          ratio_significance = numpy.round(ratio_significance_raw*100, 4)
          # for simulation 0 only, check if ratio roughly = 5%
          lim = 2*numpy.sqrt(0.05*(1-0.05)/J)
          verdict = 0.05-lim <= ratio_significance_raw <= 0.05+lim
          # save results
          results_simulation['FFX_X2_n{}'.format(n)] = {'T_map':T_map, 'p_values':p_values, 'ratio_significance':ratio_significance, 'verdict':verdict, 'si2':si2, 'distrib':'T distribution, df={} (sum(ni-1)-1)'.format((n - 1)*K -1), 'FFX_ns_pvalues':FFX_ns_pvalues}



     ##################
     ### MFX GLM
     ##################
     # INPUTS ARE BETA VALUES !
     # final stats is sum(ki*Bi)/sqrt(sum(Ki)) with ki = 1/(tau2 + si2)

     # si2 = variance of contrast estimate for study i
     # matrix of ones ONLY if within study variance is perfectly known
     # si2 = numpy.ones((K, J))
     si2 = numpy.array([scipy.stats.chi2.rvs(size=J, df=20-1)/100 for i in range(K)])
     # tau2 = between study variance estimated
     tau2 = utils.tau(matrix_betas) # or numpy.var(matrix_betas, axis=0) # vector shape J, variance of the contrast estimate for each voxel
     # compute meta-analytic statistics
     T_map = utils.meta_analyse_MFX(matrix_betas, si2, tau2)
     T_map = T_map.reshape(-1)
     # compute p-values for inference
     # df = k-1 
     p_values = 1 - scipy.stats.t.cdf(T_map, df=19)
     p_values = p_values.reshape(-1)
     # compute ratio of significant p-values
     ratio_significance_raw = (p_values<=0.05).sum()/len(p_values)
     ratio_significance = numpy.round(ratio_significance_raw*100, 4)
     # for simulation 0 only, check if ratio roughly = 5%
     lim = 2*numpy.sqrt(0.05*(1-0.05)/J)
     verdict = 0.05-lim <= ratio_significance_raw <= 0.05+lim
     # save results
     results_simulation['MFX'] = {'T_map':T_map, 'p_values':p_values, 'ratio_significance':ratio_significance, 'verdict':verdict, 'si2':si2, 'tau2':tau2, 'distrib':'T distribution, df=19 (K-1)'}



     ##################
     ### RFX GLM
     ##################
     # INPUTS ARE BETA VALUES !
     # final stats is sum(Bi/sqrt(k))/(tau2+s2)

     # within study contrast variance assumes to constant 
     # thus sigma2 = between study variance estimated + constant within study contrast variance
     # sigma2 = unbiased sample variance 
     sigma2 = utils.tau(matrix_betas) + scipy.stats.chi2.rvs(size=J, df=20-1)/100
     # compute meta-analytic statistics
     T_map = utils.meta_analyse_RFX(matrix_betas, sigma2)
     T_map = T_map.reshape(-1)
     # compute p-values for inference
     # df = k-1 
     p_values = 1 - scipy.stats.t.cdf(T_map, df=19)
     p_values = p_values.reshape(-1)
     # compute ratio of significant p-values
     ratio_significance_raw = (p_values<=0.05).sum()/len(p_values)
     ratio_significance = numpy.round(ratio_significance_raw*100, 4)
     # for simulation 0 only, check if ratio roughly = 5%
     lim = 2*numpy.sqrt(0.05*(1-0.05)/J)
     verdict = 0.05-lim <= ratio_significance_raw <= 0.05+lim
     # save results
     results_simulation['RFX'] = {'T_map':T_map, 'p_values':p_values, 'ratio_significance':ratio_significance, 'verdict':verdict, 'si2':si2, 'tau2':tau2, 'sigma2':sigma2, 'distrib':'T distribution, df=19 (K-1)'}


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
     # compute meta-analytic statistics for inference
     vector_fisher = utils.meta_analyse_Fisher(matrix_p)
     # compute p-values for inference (chi2 distribution)
     # df = 2k
     p_values = 1 - scipy.stats.chi2.cdf(vector_fisher, df=40)
     p_values = p_values.reshape(-1)
     # compute ratio of significant p-values
     ratio_significance_raw = (p_values<=0.05).sum()/len(p_values)
     ratio_significance = numpy.round(ratio_significance_raw*100, 4)
     # for simulation 0 only, check if ratio roughly = 5%
     lim = 2*numpy.sqrt(0.05*(1-0.05)/J)
     verdict = 0.05-lim <= ratio_significance_raw <= 0.05+lim
     # save results
     results_simulation['Fisher'] = {'T_map':vector_fisher, 'p_values':p_values, 'ratio_significance':ratio_significance, 'verdict':verdict, 'distrib':'Chi2 distribution, df=20 (2K)'}


     ##################
     ### Stouffer (adapted Fisher) FFX
     ##################
     # INPUTS ARE Z VALUES !
     # final stats is sqrt(k*1/k*sum(Zi))

     # compute meta-analytic statistics
     T_map = utils.meta_analyse_Stouffer(matrix_z)
     T_map = T_map.reshape(-1)
     # compute p-values for inference
     p_values = 1 - scipy.stats.norm.cdf(T_map)
     p_values = p_values.reshape(-1)
     # compute ratio of significant p-values
     ratio_significance_raw = (p_values<=0.05).sum()/len(p_values)
     ratio_significance = numpy.round(ratio_significance_raw*100, 4)
     # for simulation 0 only, check if ratio roughly = 5%
     lim = 2*numpy.sqrt(0.05*(1-0.05)/J)
     verdict = 0.05-lim <= ratio_significance_raw <= 0.05+lim
     # save results
     results_simulation['Stouffer'] = {'T_map':T_map, 'p_values':p_values, 'ratio_significance':ratio_significance, 'verdict':verdict, 'distrib':'N(0,1) distribution'}




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

     ##################
     ### Intuitive solution
     ##################

     Q = numpy.corrcoef(matrix_z) # shape K*K
     # compute meta-analytic statistics
     Intuitive_solution = matrix_z.sum(axis=0)/matrix_z.shape[0]
     Intuitive_solution = Intuitive_solution.reshape(-1)
     Intuitive_solution_var = numpy.ones(K).dot(Q)/K**2
     # compute p-values for inference
     p_values = 1 - scipy.stats.norm.cdf(T_map)
     p_values = p_values.reshape(-1)
     # compute ratio of significant p-values
     ratio_significance_raw = (p_values<=0.05).sum()/len(p_values)
     ratio_significance = numpy.round(ratio_significance_raw*100, 4)
     
     # for simulation 0 only, check if ratio roughly = 5%
     lim = 2*numpy.sqrt(0.05*(1-0.05)/J)
     verdict = 0.05-lim <= ratio_significance_raw <= 0.05+lim

     # save results
     results_simulation['intuitive_sol'] = {'T_map':Intuitive_solution, 'p_values':p_values, 'ratio_significance':ratio_significance, 'verdict':verdict, 'Q':Q, 'distrib':'N(0,1) distribution'}

     ##################
     ### consensus (scaling before)
     ##################

     # variance of each pipeline is assumed to be equal but allowed to vary over space

     # compute a standardized map Z∗ k for each pipeline k
     scaler = StandardScaler()
     Z_star_k = scaler.fit_transform(matrix_z.T).T # scaling team wise and back to normal shape
     # z* = (z - z_mean) / s 
     # with s = image-wise var for pipeline k
     # with z_mean = image-wise mean for pipeline k
     # numpy.divide(numpy.subtract(matrix_z, matrix_z.mean(axis=0)), matrix_z.std(axis=0))

     # These standardized maps are averaged over pipelines to create a map Z∗
     Z_star_mean_j = Z_star_k.mean(axis=0) # shape J

     # image-wise mean of this average of standard maps is zero
     assert Z_star_mean_j.mean() < 0.0001

     # and which is finally standardized, scaled and shifted 
     # to the consensus standard deviation and mean
     consensus_var = matrix_z.var(axis=1).sum() / K # scalar
     consensus_std = numpy.sqrt(consensus_var) # scalar
     consensus_mean = matrix_z.mean(axis=1).sum() / K # scalar
     attenuated_variance = numpy.divide(numpy.ones(K).T.dot(Q).dot(numpy.ones(K)), K**2) # shape K
     attenuated_std = numpy.sqrt(attenuated_variance)
     Z_star_consensus = numpy.divide(Z_star_mean_j.reshape(-1, 1), attenuated_std.reshape(1, -1)).dot(consensus_std) + consensus_mean
     Z_star_consensus = Z_star_consensus.reshape(-1)
     p_values = 1 - scipy.stats.norm.cdf(Z_star_consensus)
     p_values = p_values.reshape(-1)
     # compute ratio of significant p-values
     ratio_significance_raw = (p_values<=0.05).sum()/len(p_values)
     ratio_significance = numpy.round(ratio_significance_raw*100, 4)
     
     # for simulation 0 only, check if ratio roughly = 5%
     lim = 2*numpy.sqrt(0.05*(1-0.05)/J)
     verdict = 0.05-lim <= ratio_significance_raw <= 0.05+lim
     # save results
     results_simulation['consensus'] = {'T_map':Z_star_consensus, 'p_values':p_values, 'ratio_significance':ratio_significance, 'verdict':verdict, 'Q':Q, 'distrib':'N(0,1) distribution'}

     ##################
     ### NARPS code consensus (scaling supposed to be done, but instead res_var used in the equation)
     ##################

     res_mean = numpy.mean(matrix_z)
     res_var = numpy.mean(numpy.var(matrix_z, 1))

     X = numpy.ones((K, 1))
     Q0 = numpy.corrcoef(matrix_z)

     VarMean = res_var * X.T.dot(Q0).dot(X) / K**2

     # T  =  mean(y,0)/s-hat-2
     # use diag to get s_hat2 for each variable
     T_map = (numpy.mean(matrix_z, 0) - res_mean
          )/numpy.sqrt(VarMean)*numpy.sqrt(res_var) + res_mean
     T_map = T_map.reshape(-1)

     # Assuming variance is estimated on whole image
     # and assuming infinite df

     p_values = 1 - scipy.stats.norm.cdf(T_map)
     p_values = p_values.reshape(-1)
     # compute ratio of significant p-values
     ratio_significance_raw = (p_values<=0.05).sum()/len(p_values)
     ratio_significance = numpy.round(ratio_significance_raw*100, 4)
     # for simulation 0 only, check if ratio roughly = 5%
     lim = 2*numpy.sqrt(0.05*(1-0.05)/J)
     verdict = 0.05-lim <= ratio_significance_raw <= 0.05+lim
     # save results
     results_simulation['consensus_narps'] = {'T_map':T_map, 'p_values':p_values, 'ratio_significance':ratio_significance, 'verdict':verdict, 'Q':Q0, 'distrib':'N(0,1) distribution'}

     ##################
     ### same data consensus meta analysis (no scaling)
     ##################
     res_mean = numpy.mean(matrix_z)
     res_var = numpy.mean(numpy.var(matrix_z, 1))

     X = numpy.ones((K, 1))
     Q0 = numpy.corrcoef(matrix_z)

     VarMean = X.T.dot(Q0).dot(X) / K**2

     # T  =  mean(y,0)/s-hat-2
     # use diag to get s_hat2 for each variable
     T_map = (numpy.mean(matrix_z, 0) - res_mean
          )/numpy.sqrt(VarMean) + res_mean
     T_map = T_map.reshape(-1)

     # Assuming variance is estimated on whole image
     # and assuming infinite df
     p_values = 1 - scipy.stats.norm.cdf(T_map)
     p_values = p_values.reshape(-1)

     # compute ratio of significant p-values
     ratio_significance_raw = (p_values<=0.05).sum()/len(p_values)
     ratio_significance = numpy.round(ratio_significance_raw*100, 4)

     # for simulation 0 only, check if ratio roughly = 5%
     lim = 2*numpy.sqrt(0.05*(1-0.05)/J)
     verdict = 0.05-lim <= ratio_significance_raw <= 0.05+lim
     # save results
     results_simulation['samedata_consensus'] = {'T_map':T_map, 'p_values':p_values, 'ratio_significance':ratio_significance, 'verdict':verdict, 'Q':Q0, 'distrib':'N(0,1) distribution'}


     ##################
     ### Same data fixed effects meta analysis
     ##################
     res_mean = numpy.mean(matrix_z)
     res_var = numpy.mean(numpy.var(matrix_z, 1))

     X = numpy.ones((K, 1))
     Q0 = numpy.corrcoef(matrix_z)

     VarMean = X.T.dot(Q0).dot(X) / K**2

     # T  =  mean(y,0)/s-hat-2
     # use diag to get s_hat2 for each variable
     T_map = numpy.mean(matrix_z, 0)/numpy.sqrt(VarMean)
     T_map = T_map.reshape(-1)

     # Assuming variance is estimated on whole image
     # and assuming infinite df
     p_values = 1 - scipy.stats.norm.cdf(T_map)
     p_values = p_values.reshape(-1)
     # compute ratio of significant p-values
     ratio_significance_raw = (p_values<=0.05).sum()/len(p_values)
     ratio_significance = numpy.round(ratio_significance_raw*100, 4)
     
     # for simulation 0 only, check if ratio roughly = 5%
     lim = 2*numpy.sqrt(0.05*(1-0.05)/J)
     verdict = 0.05-lim <= ratio_significance_raw <= 0.05+lim
     # save results
     results_simulation['samedata_FFX'] = {'T_map':T_map, 'p_values':p_values, 'ratio_significance':ratio_significance, 'verdict':verdict, 'Q':Q0, 'distrib':'N(0,1) distribution'}


     ##################
     ### Same data random effects meta analysis
     ##################

     X = numpy.ones((K, 1))
     # no idea how to compute Q1 thus using Q0 for now
     Q0 = numpy.corrcoef(matrix_z)
     Q1 = Q0.copy()

     # # compute random effect Q (Tau2*Q)
     # R = numpy.eye(K) - numpy.ones((K, 1)).dot(numpy.ones((1, K)))/K
     # sampvar_est = numpy.trace(R.dot(Q0))

     # tau2 = numpy.zeros(J)
     # for j in range(J): # voxel wise
     #    Y = matrix_z[:, j] # specific voxel value for each team
     #    tau2[j] = (1/sampvar_est)*Y.T.dot(R).dot(Y) # Tau2 for a specific voxel
     # tau2 = numpy.sqrt(tau2)

     # Q1 = tau2*Q0

     VarMean = X.T.dot(Q1).dot(X) / K**2

     # T  =  mean(y,0)/s-hat-2
     # use diag to get s_hat2 for each variable
     T_map = numpy.mean(matrix_z, 0)/numpy.sqrt(VarMean)
     T_map = T_map.reshape(-1)

     # Assuming variance is estimated on whole image
     # and assuming infinite df
     p_values = 1 - scipy.stats.norm.cdf(T_map)
     p_values = p_values.reshape(-1)
     # compute ratio of significant p-values
     ratio_significance_raw = (p_values<=0.05).sum()/len(p_values)
     ratio_significance = numpy.round(ratio_significance_raw*100, 4)
     
     # for simulation 0 only, check if ratio roughly = 5%
     lim = 2*numpy.sqrt(0.05*(1-0.05)/J)
     verdict = 0.05-lim <= ratio_significance_raw <= 0.05+lim
     # save results
     results_simulation['samedata_RFX'] = {'T_map':T_map , 'p_values':p_values, 'ratio_significance':ratio_significance, 'verdict':verdict, 'Q':Q1, 'distrib':'N(0,1) distribution'}

     print("** ENDED WELL **")
     return results_simulation



if __name__ == "__main__":
   print('This file is intented to be used as imported only')