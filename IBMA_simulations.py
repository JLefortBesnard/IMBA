import numpy
import importlib
import os
import MetaAnalysesAlgorithms
import utils


from scipy.stats import norm
import seaborn
import matplotlib.pyplot as plt
import pandas

importlib.reload(utils) # reupdate imported codes, useful for debugging
importlib.reload(MetaAnalysesAlgorithms)


# starting point
K=20 # nb of teams
J=20000 # nb of voxels

# simulation 0
sigma=1
# simulation 0 and 1
mu = 0
# simulation 2 and 3
mu1 = 2 # should be at least 1

corr = 0.8 # correlation accross teams

data_generated = {} # for plotting 

print("RUNNING SIMULATION 0")


#######################################
# Simulation 0: The dumbest, null case: independent pipelines, mean 0, variance 1 (totally iid data)
# => any sensible method should work fine.
# generate iid matrix of dimension K columns, J rows
#######################################

# generate sample 
rng = numpy.random.default_rng()
matrix_betas = mu + sigma * rng.standard_normal(size=(K,J))
data_generated["sim0"] = matrix_betas
# results dir
results_dir = "results_simulations"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
# launch analyses
simulation_nb = 0
results_simulation_0 = MetaAnalysesAlgorithms.run_all_MA_algorithms(matrix_betas, simulation_nb)
utils.plot_simulation_results(simulation_nb, results_simulation_0)



print("RUNNING SIMULATION 1")
#######################################
# Simulation 1: Null data with correlation: Induce correlation Q, mean 0, variance 1
# => verifies that the 1’Q1/K^2 term is correctly accounting for dependence.
#######################################
# generate betas
matrix_betas = utils.null_data(J=J, K=K, covar=corr)
data_generated["sim1"] = matrix_betas
# launch analyses
simulation_nb = 1
results_simulation_1 = MetaAnalysesAlgorithms.run_all_MA_algorithms(matrix_betas, simulation_nb)
utils.plot_simulation_results(simulation_nb, results_simulation_1)

print("RUNNING SIMULATION 2")
#######################################
# Simulation 2: Non-null but totally homogeneous data: Correlation Q but all pipelines share mean mu=mu1>0, variance 1
# => adds signal, but in a totally homogeneous way that ensures Q will be estimated in an unbiased manner; the consensus estimate should be an unbiased estimate of mu1/sqrt(1’Q1/K^2).
#######################################
# generate betas
matrix_betas = utils.non_null_homogeneous_data(J=J, K=K, covar=corr, mean=mu1)
data_generated["sim2"] = matrix_betas
# launch analyses
simulation_nb = 2
results_simulation_2 = MetaAnalysesAlgorithms.run_all_MA_algorithms(matrix_betas, simulation_nb)
utils.plot_simulation_results(simulation_nb, results_simulation_2)


print("RUNNING SIMULATION 3")
#######################################
# Simulation 3: Non-null but totally heterogeneous data: Correlation Q, all pipelines share same mean, but 50% of voxels have mu=mu1, 50% of voxels have mu = -mu1
# => adds signal that is heterogeneous, +/-mu1, which will inflate our estimate of Q; i.e. I suspect that the consensus estimate will be a biased estimate of +/-mu1/sqrt(1’Q1/K^2).
#######################################
# generate betas
matrix_betas = utils.non_null_data_heterogeneous(J=J, K=K, covar=corr, mean=mu1)
data_generated["sim3"] = matrix_betas
# launch analyses
simulation_nb = 3
results_simulation_3 = MetaAnalysesAlgorithms.run_all_MA_algorithms(matrix_betas, simulation_nb)
utils.plot_simulation_results(simulation_nb, results_simulation_3)


utils.plot_generated_data(data_generated)


import scipy
import matplotlib.patches as mpatches
#######################################
# Simulation 0 with different size of K
#######################################
Ks = [20, 40, 60, 80, 100]
ns = [20, 40, 60, 80, 100]
results_dir = "results_simulations"
FFX_sim0_results = {}
for K in Ks:
    results_temp={}
    # generate sample 
    rng = numpy.random.default_rng()
    matrix_betas = mu + sigma * rng.standard_normal(size=(K,J))
    for n in ns:
        si2 = numpy.ones((K, J))
        # compute meta-analytic statistics 
        T_map = utils.meta_analyse_FFX(matrix_betas, si2)
        T_map = T_map.reshape(-1)
        # compute p-values for inference
        p_values = 1 - scipy.stats.t.cdf(T_map, df=(n - 1)*K -1)
        p_values = p_values.reshape(-1)
        # save results
        FFX_sim0_results['K={}n={}s=1'.format(K, n)] = p_values

        si2 = numpy.array([scipy.stats.chi2.rvs(size=J, df=n-1)/100 for i in range(K)])
        # compute meta-analytic statistics 
        T_map = utils.meta_analyse_FFX(matrix_betas, si2)
        T_map = T_map.reshape(-1)
        # compute p-values for inference
        p_values = 1 - scipy.stats.t.cdf(T_map, df=(n - 1)*K -1)
        p_values = p_values.reshape(-1)
        # save results
        FFX_sim0_results['K={}n={}s=X2'.format(K, n)] = p_values

# plot results :
p_cum = utils.distribution_inversed(J)

plt.close('all')
f = plt.figure(figsize=(20, 8))
for K in Ks:
    for n in ns:
        p_values = FFX_sim0_results['K={}n={}s=1'.format(K, n)]
        p_values.sort()
        p_obs_p_cum = -numpy.log10(p_values) - -numpy.log10(p_cum) 
        plt.plot(-numpy.log10(p_cum), p_obs_p_cum, color='skyblue' if K<80 else 'dodgerblue')
        p_values_ = FFX_sim0_results['K={}n={}s=X2'.format(K, n)]
        p_values_.sort()
        p_obs_p_cum = -numpy.log10(p_values_) - -numpy.log10(p_cum) 
        plt.plot(-numpy.log10(p_cum), p_obs_p_cum, color='y' if K<80 else 'gold')

plt.title("p-plot for FFX independant data : K and n among [20, 40, 60, 80, 100]")
plt.xlabel("-log10 cumulative p")
plt.ylabel("observed p - cumulative p")
plt.vlines(-numpy.log10(0.05), ymin=-1, ymax=1, color='black', linewidth=0.5, linestyle='--')
plt.hlines(0, xmin=-1, xmax=6, color='black', linewidth=0.5, linestyle='--')
plt.xlim(0,4.5)
plt.ylim(-0.5,1)
plt.text(0.4, 0.8, 'n=20', color='y')
plt.text(0.75, 0.75, 'n=40', color='y')
plt.text(1, 0.6, 'n=60', color='y')
plt.text(1.5, 0.4, 'n=80', color='y')
plt.text(1.5, 0.1, 'n=100', color='y')
plt.text(4.1, 0.6, 'K=80', color='gold')
plt.text(4.1, 0.55, 'K=100', color='gold')
plt.text(4.3, -0.2, 'K=80', color='gold')
plt.text(4.2, -0.25, 'K=100', color='gold')
plt.text(4.4, 0, 'K=80', color='dodgerblue')
plt.text(4.2, -0.4, 'K=100', color='dodgerblue')
plt.text(1.35, -0.45, 'p>0.05', color='black')
plt.text(1.1, -0.45, 'p<0.05', color='black')

X2_patch = mpatches.Patch(color='yellow', label='si2 = X2 distribution (df = n-1)')
ones_patch = mpatches.Patch(color='skyblue', label='si2 = 1s')
plt.legend(handles=[X2_patch, ones_patch], loc='lower left', handleheight=0.2)

plt.savefig('results_simulations/FFX_pplots_sim0.png')

# compute ratio of significant p-values
ratio_significance_raw = (p_values<=0.05).sum()/len(p_values)
ratio_significance = numpy.round(ratio_significance_raw*100, 4)
# for simulation 0 only, check if ratio roughly = 5%
lim = 2*numpy.sqrt(0.05*(1-0.05)/J)
verdict = 0.05-lim <= ratio_significance_raw <= 0.05+lim