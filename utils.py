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

def meta_analyse_FFX(deltas: numpy.ndarray, deltas_var: numpy.ndarray) -> numpy.ndarray:
    """
    Compute final statmap as sum(Bi/si2)/sqrt(sum(1/si2))

    Parameters
    ---------
    deltas : numpy.ndarray
        Vector of meta-analytic parameters
    deltas_var : numpy.ndarray
        Vector of meta-analytic parameters variance
    Returns
    -------
    numpy.ndarray : Vector of meta-analytic statistics for inference
    """
    return deltas/deltas_var/numpy.sqrt(1/deltas_var)


def meta_analyse_MFX(deltas: numpy.ndarray, deltas_var: numpy.ndarray) -> numpy.ndarray:
    """
    Compute final statmap as sum(ki*Bi)/sqrt(sum(Ki)) with ki = 1/(tau2 + si2)

    Parameters
    ---------
    deltas : numpy.ndarray
        Vector of meta-analytic parameters
    deltas_var : numpy.ndarray
        Vector of meta-analytic parameters variance
    Returns
    -------
    numpy.ndarray : Vector of meta-analytic statistics for inference
    """
    return deltas*1/deltas_var/numpy.sqrt(1/deltas_var)


def meta_analyse_RFX(deltas: numpy.ndarray, deltas_var: numpy.ndarray) -> numpy.ndarray:
    """
    Compute final statmap as sum(Bi/sqrt(k))/(tau2+s2)

    Parameters
    ---------
    deltas : numpy.ndarray
        Vector of meta-analytic parameters
    deltas_var : numpy.ndarray
        Vector of meta-analytic parameters variance
    Returns
    -------
    numpy.ndarray : Vector of meta-analytic statistics for inference
    """
    return deltas/numpy.sqrt(deltas.__len__())/deltas_var


def meta_analyse_Fisher(matrix_p_values: numpy.ndarray) -> numpy.ndarray:
    matrix_fisherized = numpy.zeros(matrix_p_values.shape)
    from math import log
    for x, vector_p in enumerate(matrix_p_values):
        for y, scalar_p in enumerate(vector_p):
            matrix_fisherized[x][y] = log(scalar_p)
    vector_fisherized = matrix_fisherized.sum(axis=0) # team wise
    return -2 * vector_fisherized


def meta_analyse_Stouffer(matrix_z_values: numpy.ndarray) -> numpy.ndarray:
    k = matrix_z_values.shape[1]
    return numpy.sqrt(k)*1/k*matrix_z_values.sum(axis=0) # team wise



# PLOTTING
#######################################
def plot_generated_data(title: str, matrix_betas: numpy.ndarray, results_dir: str):
    # add a plot if TAU values are availables

    f = plt.subplots(figsize=(10, 10)) 
    
    #########
    ### Column 0
    #########

    # display generated data
    ax0 = plt.subplot2grid((6, 2), (0, 0), colspan=1, rowspan=2)
    seaborn.heatmap(matrix_betas[:, :50], center=0, vmin=matrix_betas.min(), vmax=matrix_betas.max(), cmap='coolwarm', ax=ax0,cbar_kws={'shrink': 0.5})
    ax0.title.set_text("Generated betas")
    ax0.set_ylabel("K teams")
    ax0.set_xlabel('J Voxels')

    # display inter-pipeline mean
    ax1 = plt.subplot2grid((6, 2), (2, 0), colspan=1, rowspan=1)
    seaborn.heatmap(numpy.mean(matrix_betas, 0).reshape(1, -1)[:, :50], center=0, vmin=numpy.mean(matrix_betas, 0).min(), vmax=numpy.mean(matrix_betas, 0).max(), cmap='coolwarm', square=True, ax=ax1, cbar_kws={'shrink': 0.3})
    ax1.set_ylabel("mean")
    ax1.set_xlabel('J Voxels')
    ax1.title.set_text("inter-pipeline mean ({})".format(numpy.round(numpy.mean(matrix_betas), 2)))

    # display inter-pipeline variance
    ax2 = plt.subplot2grid((6, 2), (3, 0), colspan=1, rowspan=1)
    seaborn.heatmap(numpy.var(matrix_betas, 0).reshape(1, -1)[:, :50], vmin=0, vmax=numpy.var(matrix_betas, 0).max(), cmap='Reds', square=True, ax=ax2, cbar_kws={'shrink': 0.3})
    ax2.set_ylabel("var")
    ax2.set_xlabel('J Voxels')
    ax2.title.set_text("inter-pipeline variance ({})".format(numpy.round(numpy.var(matrix_betas), 2)))


    # display spatial mean
    ax3 = plt.subplot2grid((6, 2), (4, 0), colspan=1, rowspan=1)
    seaborn.heatmap(numpy.mean(matrix_betas, 1).reshape(1, -1), center=0, vmin=numpy.mean(matrix_betas, 1).min(), vmax=numpy.mean(matrix_betas, 1).max(), cmap='coolwarm', square=True, ax=ax3, cbar_kws={'shrink': 0.3})
    ax3.set_ylabel("mean")
    ax3.set_xlabel('K teams')
    ax3.title.set_text("spatial mean")
    
    # display spatial variance
    ax4 = plt.subplot2grid((6, 2), (5, 0), colspan=1, rowspan=1)
    seaborn.heatmap(numpy.var(matrix_betas, 1).reshape(1, -1), vmin=0, vmax=numpy.var(matrix_betas, 1).max(), cmap='Reds', square=True, ax=ax4, cbar_kws={'shrink': 0.3})
    ax4.set_ylabel("var")
    ax4.set_xlabel('K teams')
    ax4.title.set_text("Spatial variance")

    #########
    ### Column 1
    #########

    # display correlation matrix between team
    ax5 = plt.subplot2grid((6, 2), (0, 1), colspan=1, rowspan=2)
    corr_mat = numpy.corrcoef(matrix_betas)
    seaborn.heatmap(corr_mat, vmin=0, vmax=1, cmap='Reds', square=True, ax=ax5,cbar_kws={'shrink': 0.3})
    ax5.title.set_text("Q0 matrix ({})".format(numpy.round(corr_mat.mean(), 2)))
    ax5.set_ylabel("K teams")
    ax5.set_xlabel('K teams')

    # display correlation matrix between voxel (spatial correlation)
    ax6 = plt.subplot2grid((6, 2), (2, 1), colspan=1, rowspan=2)
    corr_mat = numpy.corrcoef(matrix_betas.T)
    # seaborn cannot plot 20000 values... thus limit to 100
    seaborn.heatmap(corr_mat[:50,:50], vmin=0, vmax=1, cmap='Reds', square=True, ax=ax6,cbar_kws={'shrink': 0.3})
    ax6.title.set_text("Spatial correlation matrix ({})".format(numpy.round(corr_mat.mean(), 2)))
    ax6.set_ylabel("J Voxels")
    ax6.set_xlabel('J Voxels')

    plt.suptitle('{} : information (first 50 voxels)'.format(title))
    plt.tight_layout()
    plt.savefig(opj(results_dir, "{}_info.png".format(title)))
    plt.close('all')


def plot_results(title: str, matrix_betas: numpy.ndarray, deltas: numpy.ndarray, deltas_var: numpy.ndarray, T_map: numpy.ndarray, W: numpy.ndarray, results_dir: str):
    # add a plot if TAU values are availables

    f, axs = plt.subplots(6, figsize=(5, 8)) 
        
    # display generated data
    seaborn.heatmap(matrix_betas[:, :50], center=0, vmin=-3, vmax=3, cmap='coolwarm', ax=axs[0],cbar_kws={'shrink': 0.5})
    axs[0].title.set_text("Generated betas")
    axs[0].set_ylabel("K teams")
    axs[0].set_xlabel('J Voxels')

    # display mean voxel value accross team
    seaborn.heatmap(matrix_betas.mean(axis=0).reshape(1, -1)[:, :50], center=0, vmin=deltas.min(), vmax=deltas.max(), cmap='coolwarm', square=True, ax=axs[1],cbar_kws={'shrink': 0.3})
    axs[1].set_ylabel("Mean")
 
    # display Deltas and Deltas variance
    seaborn.heatmap(deltas.reshape(1, -1)[:, :50], center=0, vmax=deltas.max(), vmin=deltas.min(), cmap='coolwarm', square=True, ax=axs[2], cbar_kws={'shrink': 0.3})
    axs[2].set_ylabel("Deltas")
    seaborn.heatmap(deltas_var.reshape(1, -1)[:, :50], center=0, vmax=deltas_var.max(), vmin=0, cmap='coolwarm', square=True, ax=axs[3], cbar_kws={'shrink': 0.3})
    axs[3].set_ylabel("Variance deltas")


    # display T map
    seaborn.heatmap(T_map.reshape(1, -1)[:, :50], center=0, vmin=T_map.min(), vmax=T_map.max(), cmap='coolwarm', square=True, ax=axs[4], cbar_kws={'shrink': 0.3})
    axs[4].set_ylabel("T stat")

    # display p_values
    p_values = 1 - scipy.stats.norm.cdf(T_map)
    seaborn.heatmap(p_values.reshape(1, -1)[:, :50], vmin=0, vmax=0.06, cmap='Reds_r', square=True, mask=p_values.reshape(1, -1)[:, :50] > 0.05, ax=axs[5], cbar_kws={'shrink': 0.3})
    ratio = (p_values<=0.05).sum()/p_values.__len__()
    lim = 2*numpy.sqrt(0.05*(1-0.05)/20000)
    verdict = 0.05-lim <= ratio <= 0.05+lim
    print(ratio, lim, verdict)
    axs[5].title.set_text("sig p values, ratio={}, {}".format(numpy.round(ratio, 3), verdict))
    axs[5].set_ylabel("p values")

    plt.suptitle('Results for {}'.format(title))
    plt.tight_layout()
    plt.savefig(opj(results_dir, "{}_maps.png".format(title)))
    plt.close('all')





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
    f, axs = plt.subplots(4, 6, figsize=(15, 15))  
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
        seaborn.heatmap(matrix, center=0, vmax=0.8, vmin=-0.8, cmap='coolwarm', robust=True, square=True, ax=axs[row, col], cbar_kws={'shrink': 0.6})
        seaborn.heatmap(matrix_organized, center=0, vmax=0.8, vmin=-0.8, cmap='coolwarm', robust=True, square=True, ax=axs[row, col+2], cbar_kws={'shrink': 0.6})
        seaborn.heatmap(matrix_organized_louvain, center=0, vmax=0.8, vmin=-0.8, cmap='coolwarm', robust=True, square=True, ax=axs[row, col+4], cbar_kws={'shrink': 0.6})
        if matrice.split('/')[-1] == "Q_mask_99_sim{}.npy".format(sim_number):
            name_roi = "participant_mask"
        elif matrice.split('/')[-1] == "Q_znarps_mask_sim{}.npy".format(sim_number):
            name_roi = "Narps mask"
        else:
            name_roi = matrice.split('/')[-1][2:-18]
        title = name_roi + ' ' + str(numpy.round(numpy.mean(numpy.load(matrice)), 3))
        title_organized = name_roi
        axs[row, col].title.set_text(title)
        axs[row, col+2].title.set_text(title_organized)
        axs[row, col+4].title.set_text(title_organized)
        axs[-1, 1].axis('off') # get rid of matrice using mask from narps (it's empty)
        axs[-1, 3].axis('off') # get rid of matrice using mask from narps (it's empty)
        axs[-1, 5].axis('off') # get rid of matrice using mask from narps (it's empty)
    plt.suptitle('Simulation  {}'.format(sim_number))
    # adjust the subplots, i.e. leave more space at the top to accomodate the additional titles
    f.subplots_adjust(top=0.78) 
    plt.figtext(0.1,0.95,"Original", va="center", ha="center", size=12)
    plt.figtext(0.5,0.95,"Sorted : Intensity", va="center", ha="center", size=12)
    plt.figtext(0.8,0.95,"Sorted : Louvain", va="center", ha="center", size=12)
    plt.tight_layout()
    plt.savefig('{}/similation_{}_corr{}.png'.format(results_dir, sim_number, corr), dpi=300)
    plt.close('all')


def display_similarity_matrices(results_dir, sim_number, corr):
    # organized + raw matrices
    correlation_matrices = glob.glob('{}/temp/Q_*_sim{}.npy'.format(results_dir, sim_number))
    correlation_matrices.sort()
    # load reference matrix (correlation matrix with participant mask) for similarity computation
    matrix_reference_path = '{}/temp/Q_mask_99_sim{}.npy'.format(results_dir, sim_number)
    matrix_reference = numpy.load(matrix_reference_path)
    # no need to iterate over the matrix used as reference
    correlation_matrices.remove(matrix_reference_path) 

    f, axs = plt.subplots(3, 2, figsize=(5, 10))
    for ind, matrice in enumerate(correlation_matrices):
        if ind < 3:
            row = ind
            col = 0
        else:
            row = ind - 3
            col = 1
        matrix = numpy.load(matrice)
        # similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(matrix, matrix_reference)
        similarity_matrix = matrix - matrix_reference
        # Frobenius Norm => (Sum(abs(value)**2))**1/2
        Fro = numpy.linalg.norm((matrix - matrix_reference), ord='fro')
        # L1 Norm // manhatan distance => max(sum(abs(x), axis=0))
        L1 = numpy.linalg.norm((matrix - matrix_reference), ord=1)
        # L2 Norm // euclidian distance => 2-norm(largest sing. value)
        L2 = numpy.linalg.norm((matrix - matrix_reference), ord=2)
        seaborn.heatmap(similarity_matrix, center=0, vmax=0.04, vmin=-0.04, cmap='coolwarm', robust=True, square=True, ax=axs[row, col], cbar_kws={'shrink': 0.2})
        if matrice.split('/')[-1] == "Q_znarps_mask_sim{}.npy".format(sim_number):
            name_roi = "Narps mask"
        else:
            name_roi = matrice.split('/')[-1][2:-18]
        title = (name_roi 
                + '\n Mean corr =' + str(numpy.round(numpy.mean(similarity_matrix), 2)) 
                + '\n Fro norm = {}'.format(numpy.round(Fro, 2))
                + '\n L1 norm = {}'.format(numpy.round(L1, 2))
                + '\n L2 norm = {}'.format(numpy.round(L2, 2)))
        axs[row, col].set_title(title, fontsize=8)

    plt.suptitle('Simulation  {} : similarity matrix'.format(sim_number))
    plt.tight_layout()
    plt.savefig('{}/similation_{}_corr{}_similarity.png'.format(results_dir, sim_number, corr), dpi=300)
    plt.close('all')




if __name__ == "__main__":
   print('This file is intented to be used as imported only')