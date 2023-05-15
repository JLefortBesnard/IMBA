"""
utility functions for meta-analysis comparisons
"""
import numpy
import seaborn
import matplotlib.pyplot as plt
from math import log
import scipy
from os.path import join as opj
import sklearn.metrics
import pandas

#######################################
# UTILS
#######################################


# DATA GENERATOR
#######################################

def null_data(K: int, J: int, covar: float) -> numpy.ndarray:
    """
    Simulation 1: Null data with correlation: Induce correlation Q, mean 0, variance 1
    
    Parameters
    ---------
    K : int
        dimensionality of samples
    J : int
        number of samples generated
    covar : float
        uniform covariance for samples
    
    Returns
    -------
    samples : numpy.ndarray
        samples in as (J, d)-matrix
    """
    cov_mat = numpy.ones((K, K)) * covar
    numpy.fill_diagonal(cov_mat, 1)
    offset = numpy.zeros(K) # mean 0
    return numpy.random.multivariate_normal(offset, cov_mat, size=J).T # normal thus var = 1, transposed to get shape K,J


def non_null_homogeneous_data(K: int, J: int, covar: float, mean: float) -> numpy.ndarray:
    """
    Simulation 2: Non-Null data with correlation: Induce correlation Q, mean >= 1, variance 1

    Parameters
    ---------
    K : int
        dimensionality of samples
    J : int
        number of samples generated
    covar : float
        uniform covariance for samples
    mean : float
        uniform mean for samples
    
    Returns
    -------
    samples : numpy.ndarray
        samples in as (K, J)-matrix
    """
    cov_mat = numpy.ones((K, K)) * covar
    numpy.fill_diagonal(cov_mat, 1)
    offset = numpy.ones(K) * mean 

    return numpy.random.multivariate_normal(offset, cov_mat, size=J).T # normal thus var = 1, transposed to get shape K,J


def non_null_data_heterogeneous(K: int, J: int, covar: float, mean: float) -> numpy.ndarray:
    """
    Simulation 3: Non-null but totally heterogeneous data: Correlation Q + 
    all pipelines share same mean, but 50% of voxels have mu=mu1, 50% of voxels have mu = -mu1

    Parameters
    ---------
    K : int
        dimensionality of samples
    J : int
        number of samples generated
    covar : float
        uniform covariance for samples
    mean : float
        diagonal value between mean and -mean for samples
    
    Returns
    -------
    samples : numpy.ndarray
        samples in as (J, d)-matrix
    """
    cov_mat = numpy.ones((K, K)) * covar
    numpy.fill_diagonal(cov_mat, 1)
    offset = numpy.random.choice([mean, -mean], size=K, replace=True)

    return numpy.random.multivariate_normal(offset, cov_mat, size=J).T # normal thus var = 1, transposed to get shape K,J



# MATRIX OPERATIONS
#######################################

def is_invertible(matrix: numpy.ndarray) -> bool:
    """
    Parameters
    ---------
    matrix : numpy.ndarray  

    Returns
    -------
    Bool : True if matrix is invertible
    """
    return a.shape[0] == a.shape[1] and numpy.linalg.matrix_rank(a) == a.shape[0]


# voxels by voxels
def compute_delta_for_one_voxel(vector_betas: numpy.ndarray, w: float, GLM: str) -> list:
    """
    Compute delta for a voxel given a number of teams

    Problem stated as B = X * delta + residuals
    solution (OLS solution) => delta = inverse(X_transpose*W_inverse*X)*X_transpose*W_inverse*B 
    
    Parameters
    ---------
    vector_betas : numpy.ndarray
        Vector of contrast estimates dimension k studies * 1 voxel
    w : scalar
        scalar of the contrast estimate variance for 1 voxel
    GLM : str
        if 'RDX', formulas to compute delta and delat_var are differents:
        delta = numpy.linalg.inv(X.T**X)*X.T*matrix_betas
        delta_var  = delta.dot(w)
        instead of 
        delta = numpy.linalg.inv(X.T*numpy.linalg.inv(w)*X)*X.T*numpy.linalg.inv(w)*matrix_betas
        delta_var  = numpy.linalg.inv((X.T*numpy.linalg.inv(w)).dot(X))
    
    Returns
    -------
    delta : numpy.ndarray
        estimated meta-analytic parameter
    delta_var : numpy.ndarray
        estimated sampling variance of meta-analytic parameter
    """
    vector_betas = vector_betas.reshape(-1, 1) # reshape cause needs 2D to compute inverse matrix
    X = numpy.ones(vector_betas.shape) # design matrix
    
    if GLM == '3.RFX':
        # delta = numpy.linalg.inv(X.T**X)*X.T*matrix_betas
        try:
            delta = numpy.linalg.inv(X.T.dot(X))
            delta_var  = delta.dot(w)
            delta = delta.dot(X.T)
            delta = delta.dot(vector_betas)
            return [float(delta), float(delta_var)]
        except:
            print("Cannot compute delta, matrix is singular...")

   
    # delta = numpy.linalg.inv(X.T*numpy.linalg.inv(w)*X)*X.T*numpy.linalg.inv(w)*matrix_betas
    try:
        delta = X.T*numpy.linalg.inv(w)
        delta= delta.dot(X)
        delta_var  = numpy.linalg.inv(delta)
        delta = delta_var.dot(X.T)
        delta = delta*numpy.linalg.inv(w)  # the bigger the variance, the smaller the final delta
        delta = delta.dot(vector_betas)

        return [float(delta), float(delta_var)]
    except:
        print("Cannot compute delta, matrix is singular...")


def compute_deltas(matrix_betas: numpy.ndarray, W: float, GLM='1.FFX') -> numpy.ndarray:
    """
    Iterate through each voxel to compute their associated delta, given a number of teams
    print the output

    Parameters
    ---------
    matrix_betas : numpy.ndarray
        Matrix of contrast estimates dimension K studies * J voxels
    W : numpy.ndarray
        Vector of the contrast estimate variance for each voxel
    GLM : str
        Default : 'FFX'
        if 'RDX', impact next step => compute_delta_for_one_voxel()

    Returns
    -------
    numpy.ndarray : including the deltas (one for each voxel) and the deltas variance (one for each voxel)
    """
    J = matrix_betas.shape[1]
    deltas = []
    deltas_var = []
    for j_voxel in range(J):
        d, d_var = compute_delta_for_one_voxel(matrix_betas[:, j_voxel], W[j_voxel].reshape(-1, 1), GLM=GLM) 
        deltas.append(d)
        deltas_var.append(d_var)
    ## Debugging
    # print('Deltas per voxel')
    # print(deltas)
    # print('Mean per voxel')
    # print(matrix_betas.mean(axis=0))
    # print('Deltas variance per voxel')
    # print(deltas_var)
    return numpy.array([deltas, deltas_var])


def tau(matrix: numpy.ndarray) -> float:
    """
    Compute Tau**2

    Parameters
    ---------
    matrix : numpy.ndarray
        Matrix of contrast estimates dimension K studies * J voxels

    Returns
    -------
    float : Tau**2 value
    """
    K = matrix.shape[0]
    J = matrix.shape[1]
    Q = numpy.corrcoef(matrix)
    # matrix of size K*K with diagnonal 0.9 and rest -0.1
    R = numpy.eye(K) - numpy.ones((K, 1)).dot(numpy.ones((1, K)))/K
    # scalar: 3.6
    sampvar_est = numpy.trace(R.dot(Q))
    tau2 = numpy.zeros(J)
    for j in range(J): # voxel wise
        Y = matrix[:, j] # specific voxel value for each team
        tau2[j] = (1/sampvar_est)*Y.T.dot(R).dot(Y) # Tau2 for a specific voxel
    return(numpy.sqrt(tau2))



# META ANALYSIS MODELING
#######################################

def meta_analyse_FFX(matrix_betas: numpy.ndarray, si2: numpy.ndarray) -> numpy.ndarray:
    """
    Compute final T map as sum(Bi/si2)/sqrt(sum(1/si2))
    with si2 (variance of contrast estimate for study i) 
    being assumed to be equal to between study variance tau2

    Parameters
    ---------
    matrix_betas : numpy.ndarray
        matrix of generated beta values shape K,J
    si2 : numpy.ndarray
        vector of shape J

    Returns
    -------
    vector : numpy.ndarray
        Vector of meta-analytic statistics for inference
    """
    top = numpy.sum(numpy.divide(matrix_betas, si2), axis=0) # vector shape J
    down = numpy.sqrt(numpy.sum(1/si2, axis=0)) # vector shape J
    return top/down

def meta_analyse_MFX(matrix_betas: numpy.ndarray, si2: numpy.ndarray, tau2: numpy.ndarray) -> numpy.ndarray:
    """
    Compute final statmap as sum(ki*Bi)/sqrt(sum(Ki)) with ki = 1/(tau2 + si2)
    with si2 (variance of contrast estimate for study i) 
    being assumed to be equal to between study variance tau2

    Parameters
    ---------
    matrix_betas : numpy.ndarray
        matrix of generated beta values shape K,J
    si2 : numpy.ndarray
        vector of shape J
    tau2 : numpy.ndarray
        vector of shape J

    Returns
    -------
    vector : numpy.ndarray
        Vector of meta-analytic statistics for inference
    """
    k = 1/(si2 + tau2) # matrix of shape K,J
    top = numpy.sum(numpy.multiply(matrix_betas,k), axis=0) # vector
    down = numpy.sqrt(numpy.sum(k, axis=0)) # vector of shape J
    return top/down


def meta_analyse_RFX(matrix_betas: numpy.ndarray, sigma2: numpy.ndarray) -> numpy.ndarray:
    """
    Compute final statmap as sum(Bi/sqrt(k))/(tau2+s2)
    with si2 (variance of contrast estimate for study i) 
    being assumed to be equal to between study variance tau2

    Parameters
    ---------
    matrix_betas : numpy.ndarray
        matrix of generated beta values shape K,J
    sigma2 : numpy.ndarray
        unbiased sample variance of shape J

    Returns
    -------
    vector : numpy.ndarray
        Vector of meta-analytic statistics for inference
    """
    K = matrix_betas.shape[0] # 20
    top = numpy.sum(numpy.divide(matrix_betas, numpy.sqrt(K)), axis=0) # vector
    return top/sigma2


def meta_analyse_Fisher(matrix_p_values: numpy.ndarray) -> numpy.ndarray:
    matrix_fisherized = numpy.zeros(matrix_p_values.shape)
    from math import log
    for x, vector_p in enumerate(matrix_p_values):
        for y, scalar_p in enumerate(vector_p):
            matrix_fisherized[x][y] = log(scalar_p)
    vector_fisherized = matrix_fisherized.sum(axis=0) # team wise
    return -2 * vector_fisherized


def meta_analyse_Stouffer(matrix_z_values: numpy.ndarray) -> numpy.ndarray:
    K = matrix_z_values.shape[0] # 20
    return numpy.sqrt(K)*1/K*matrix_z_values.sum(axis=0) # team wise



# PLOTTING
#######################################
def plot_generated_data(data_generated: dict):
    print("Plotting generated data")
    f, axs = plt.subplots(3, 4, figsize=(20, 8)) 
    for index, title in enumerate(data_generated.keys()):
        matrix_betas = data_generated[title]
        mean = numpy.round(numpy.mean(matrix_betas), 2)
        var = numpy.round(numpy.var(matrix_betas), 2)
        seaborn.heatmap(matrix_betas[:, :50], center=0, vmin=matrix_betas.min(), vmax=matrix_betas.max(), cmap='coolwarm', ax=axs[0, index],cbar_kws={'shrink': 0.5})
        axs[0, index].title.set_text("simulation {}\nGenerated betas\nmean={}, var={}".format(index, mean, var))
        axs[0, index].set_xlabel("J voxels")
        axs[0, index].set_ylabel("K teams")

        corr_mat = numpy.corrcoef(matrix_betas.T)
        seaborn.heatmap(corr_mat[:50,:50], vmin=0, vmax=1, cmap='Reds', square=True, ax=axs[1, index],cbar_kws={'shrink': 0.3})
        axs[1, index].title.set_text("Spatial corr matrix ({})".format(numpy.round(corr_mat.mean(), 2)))
        axs[1, index].set_ylabel("J voxels")
        axs[1, index].set_xlabel('J voxels')

        corr_mat = numpy.corrcoef(matrix_betas)
        seaborn.heatmap(corr_mat, vmin=0, vmax=1, cmap='Reds', square=True, ax=axs[2, index],cbar_kws={'shrink': 0.3})
        axs[2, index].title.set_text("Q0 matrix ({})".format(numpy.round(corr_mat.mean(), 2)))
        axs[2, index].set_ylabel("K teams")
        axs[2, index].set_xlabel('K teams')

    plt.suptitle('{} : information (first 50 voxels)'.format(title))
    plt.tight_layout()
    plt.savefig("results_simulations/generated_data_info.png")
    plt.close('all')



def plot_simulation_results(simulation_nb, results):
    print("Plotting results for simulation {}".format(simulation_nb))
    matrix_betas = results['data']['matrix_betas']
    matrix_z = results['data']['matrix_z']
    matrix_p = results['data']['matrix_p']
    p_theoretical = results['Stouffer']['p_values'].copy()
    p_theoretical.sort()
    f, axs = plt.subplots(4, 5, figsize=(20, 8)) 
    for index, title in enumerate(['Fisher', 'Stouffer', 'FFX', 'MFX', 'RFX']):
        if title == 'Fisher':
            vector_fisher = results[title]['vector_fisher'].copy()
            ratio_significance = results[title]['ratio_significance']
            verdict = results[title]['verdict']

            axs[0, index].axis('off')
            axs[0, index].title.set_text(" {}\n ".format(title))
            axs[1, index].axis('off')

            vector_fisher_sorted = vector_fisher.copy()
            vector_fisher_sorted.sort()
            axs[2, index].hist(vector_fisher_sorted, bins=100, color='y')
            axs[2, index].title.set_text("X2 values distribution")
            axs[2, index].set_ylabel("frequency")
            axs[2, index].set_xlabel("X2 value")
            axs[2, index].axvline(55.758, ymin=0, color='black', linewidth=0.5, linestyle='--')

            p_obs_p_cum = vector_fisher_sorted - vector_fisher_sorted
            # pobs-pcum and pcum distribution
            axs[3, index].plot(vector_fisher_sorted, p_obs_p_cum, color='y')
            axs[3, index].title.set_text("p-plot")
            axs[3, index].set_xlabel("cumulative X2")
            axs[3, index].set_ylabel("observed X2 - cumulative X2")
            axs[3, index].axvline(55.758, ymin=-1, color='black', linewidth=0.5, linestyle='--')
            axs[3, index].axhline(0, color='black', linewidth=0.5, linestyle='--')
            axs[3, index].set_xlim(0, 100)
            axs[3, index].set_ylim(-0.5, 0.5)
            axs[3, index].text(2, 0.25, 'ratio={}%, {}'.format(ratio_significance, verdict))

            plt.suptitle('{}'.format(title))
            continue # next loop iteration

        # else    
        T_map = results[title]['T_map']
        p_values = results[title]['p_values']
        ratio_significance = results[title]['ratio_significance']
        verdict = results[title]['verdict']

        # display p over t disctribution
        df_obs = pandas.DataFrame(data=numpy.array([p_values, T_map]).T, columns=["p_values", "T_values"])
        df_cum = df_obs.sort_values(by=['p_values'])
        # t and p distribution
        axs[0, index].plot(df_cum['T_values'].values, df_cum['p_values'].values, color='tab:blue')
        axs[0, index].title.set_text(" {}\np values of t statistics (1-tail)".format(title))
        axs[0, index].set_xlabel("t statistics")
        axs[0, index].set_ylabel("p values")
        axs[0, index].set_xlim(0, 4)

        axs[1, index].hist(df_cum['T_values'].values, bins=100, color='green')
        axs[1, index].title.set_text("t statistics distribution")
        axs[1, index].set_ylabel("frequency")
        axs[1, index].set_xlabel("t value")

        axs[2, index].hist(df_cum['p_values'].values, bins=100, color='y')
        axs[2, index].title.set_text("p values distribution")
        axs[2, index].set_ylabel("frequency")
        axs[2, index].set_xlabel("p value")
        axs[2, index].axvline(0.05, ymin=0, color='black', linewidth=0.5, linestyle='--')

        p_obs_p_cum = df_cum['p_values'].values - p_theoretical
        # pobs-pcum and pcum distribution
        axs[3, index].plot(-numpy.log10(p_theoretical), p_obs_p_cum, color='y')
        axs[3, index].title.set_text("p-plot")
        axs[3, index].set_xlabel("-log10 cumulative p")
        axs[3, index].set_ylabel("observed p - cumulative p")
        axs[3, index].axvline(-numpy.log10(0.05), ymin=-1, color='black', linewidth=0.5, linestyle='--')
        axs[3, index].axhline(0, color='black', linewidth=0.5, linestyle='--')
        axs[3, index].set_xlim(0, 6)
        axs[3, index].set_ylim(-0.5, 0.5)
        axs[3, index].text(2, 0.25, 'ratio={}%, {}'.format(ratio_significance, verdict))
    plt.suptitle('Simulation {}'.format(simulation_nb))
    plt.tight_layout()
    plt.savefig("results_simulations/distributions_sim{}.png".format(simulation_nb))
    plt.close('all')

    p_theoretical = results['intuitive_sol']['p_values'].copy()
    p_theoretical.sort()
    f, axs = plt.subplots(4, 5, figsize=(20, 8)) 
    for index, title in enumerate(['intuitive_sol', 'consensus', 'samedata_consensus', 'samedata_FFX', 'samedata_RFX']):
        # else    
        T_map = results[title]['T_map']
        p_values = results[title]['p_values']
        ratio_significance = results[title]['ratio_significance']
        verdict = results[title]['verdict']

        # display p over t disctribution
        df_obs = pandas.DataFrame(data=numpy.array([p_values, T_map]).T, columns=["p_values", "T_values"])
        df_cum = df_obs.sort_values(by=['p_values'])
        # t and p distribution
        axs[0, index].plot(df_cum['T_values'].values, df_cum['p_values'].values, color='tab:blue')
        axs[0, index].title.set_text(" {}\np values of t statistics (1-tail)".format(title))
        axs[0, index].set_xlabel("t statistics")
        axs[0, index].set_ylabel("p values")
        axs[0, index].set_xlim(0, 4)

        axs[1, index].hist(df_cum['T_values'].values, bins=100, color='green')
        axs[1, index].title.set_text("t statistics distribution")
        axs[1, index].set_ylabel("frequency")
        axs[1, index].set_xlabel("t value")

        axs[2, index].hist(df_cum['p_values'].values, bins=100, color='y')
        axs[2, index].title.set_text("p values distribution")
        axs[2, index].set_ylabel("frequency")
        axs[2, index].set_xlabel("p value")

        p_obs_p_cum = df_cum['p_values'].values - p_theoretical
        # pobs-pcum and pcum distribution
        axs[3, index].plot(-numpy.log10(p_theoretical), p_obs_p_cum, color='y')
        axs[3, index].title.set_text("p-plot")
        axs[3, index].set_xlabel("-log10 cumulative p")
        axs[3, index].set_ylabel("observed p - cumulative p")
        axs[3, index].axvline(-numpy.log10(0.05), ymin=-1, color='black', linewidth=0.5, linestyle='--')
        axs[3, index].axhline(0, color='black', linewidth=0.5, linestyle='--')
        axs[3, index].set_xlim(0, 6)
        axs[3, index].set_ylim(-0.5, 0.5)
        axs[3, index].text(2, 0.25, 'ratio={}%, {}'.format(ratio_significance, verdict))
        plt.suptitle('{}'.format(title))
    plt.suptitle('Simulation {}'.format(simulation_nb))
    plt.tight_layout()
    plt.savefig("results_simulations/distributions_samedata_sim{}.png".format(simulation_nb))
    plt.close('all') 

    print("Plotting done")






# ASSUMPTION FAIRNESS TESTING
#######################################

from community import community_louvain
import networkx as nx
import pandas
import glob

def compute_louvain_community(matrix):
    ''' Compute network graph, then louvain community, save it,
    then reorganized the covariance matrix by community and plot it

    Parameters
    ----------
    matrix : correlation matrix (n_roi*n_roi)

    Returns
    ----------
    louvain community as dictionnary

    '''
    # compute the best partition
    G = nx.Graph(matrix)  
    # nx.draw(G, with_labels=True) 
    partition = community_louvain.best_partition(G, random_state=0)
    return partition

def reorganize_with_louvain_community(matrix, partition):
    ''' Reorganized the covariance matrix according to the partition

    Parameters
    ----------
    matrix : correlation matrix (n_roi*n_roi)

    Returns
    ----------
    matrix reorganized as louvain community 

    '''
    # compute the best partition
    louvain = numpy.zeros(matrix.shape).astype(matrix.dtype)
    labels = range(len(matrix))
    labels_new_order = []
    
    ## reorganize matrix abscissa wise
    i = 0
    # iterate through all created community
    for values in numpy.unique(list(partition.values())):
        # iterate through each ROI
        for key in partition:
            if partition[key] == values:
                louvain[i] = matrix[key]
                labels_new_order.append(labels[key])
                i += 1
    # check positionning from original matrix to louvain matri
    # get index of first roi linked to community 0
    index_roi_com0_louvain = list(partition.values()).index(0)
    # get nb of roi in community 0
    nb_com0 = numpy.unique(list(partition.values()), return_counts=True)[1][0]
    # # get index of first roi linked to community 1
    index_roi_com1_louvain = list(partition.values()).index(1)
    assert louvain[0].sum() == matrix[index_roi_com0_louvain].sum()
    assert louvain[nb_com0].sum() == matrix[index_roi_com1_louvain].sum() 

    df_louvain = pandas.DataFrame(index=labels_new_order, columns=labels, data=louvain)

    ## reorganize matrix Ordinate wise
    df_louvain = df_louvain[df_louvain.index]
    return df_louvain.values


def display_matrices(results_dir, sim_number, corr):
    # organized + raw matrices
    correlation_matrices = glob.glob('{}/temp/Q_*_sim{}.npy'.format(results_dir, sim_number))
    correlation_matrices.sort()

    # put the participant mask at index 0 to fit louvain and sorting according
    # to participant mask and not frontal mask (originaly at index 0)
    new_order = [3, 0, 1, 2, 4, 5, 6]
    correlation_matrices = [correlation_matrices[ind] for ind in new_order]

    # load reference matrix (correlation matrix with participant mask) for similarity computation
    matrix_reference_path = '{}/temp/Q_mask_99_sim{}.npy'.format(results_dir, sim_number)
    matrix_reference = numpy.load(matrix_reference_path)


    f, axs = plt.subplots(4, 8, figsize=(25, 15))  
    for ind, matrice in enumerate(correlation_matrices):
        matrix = numpy.load(matrice)
        if ind == 0:
            organised_ind = numpy.argsort(matrix, axis=0)
            partition = compute_louvain_community(numpy.abs(matrix))
        matrix_organized_louvain = reorganize_with_louvain_community(matrix, partition)
        matrix_organized = numpy.take_along_axis(matrix, organised_ind, axis=0)
        if ind < 4:
            row = ind
            col = 0
        else:
            row = ind - 4
            col = 1

        if matrice.split('/')[-1] == "Q_mask_99_sim{}.npy".format(sim_number):
            name_roi = "participant_mask"
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

    axs[-1, 5].axis('off') # get rid of matrice using mask from narps (it's empty)
    axs[-1, 1].axis('off') # get rid of matrice using mask from narps (it's empty)
    axs[-1, 3].axis('off') # get rid of matrice using mask from narps (it's empty)
    axs[-1, 7].axis('off') # get rid of matrice using mask from narps (it's empty)

    plt.suptitle('simutation  {}, corr {}'.format(sim_number, corr), size=16, fontweight='bold')
    f.subplots_adjust(top=0.78) 
    plt.figtext(0.1,0.95,"Original", va="center", ha="center", size=12, fontweight='bold')
    plt.figtext(0.35,0.95,"Sorted : Intensity", va="center", ha="center", size=12, fontweight='bold')
    plt.figtext(0.6,0.95,"Sorted : Louvain", va="center", ha="center", size=12, fontweight='bold')
    plt.figtext(0.87,0.95,"Similarity matrix", va="center", ha="center", size=12, fontweight='bold')
    line = plt.Line2D((.75,.75),(.1,.9), color="k", linewidth=3)
    f.add_artist(line)
    plt.tight_layout()
    plt.savefig('{}/similation_{}_corr{}.png'.format(results_dir, sim_number, corr), dpi=300)
    plt.close('all')




if __name__ == "__main__":
   print('This file is intented to be used as imported only')