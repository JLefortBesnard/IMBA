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