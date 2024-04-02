import numpy as np

def constraints_ssvdd(psi, C_val, Q, train_data, alpha_vector):
    """computing the regularization term for SSVDD based on psi parameter

    :param psi: int, determining the type of regularization term
    :param C_val: float, specifying the proportion of outliers
    :param Q: projection matrix
    :param train_data: training data
    :param alpha_vector: vector of values for each training instance defining datapoints' location in feature space
    :return: const; regularization term
    """

    if psi == 1:
        const = 0

    elif psi == 2:
        const = 2*(train_data.T @ train_data) @ Q

    elif psi == 3:
        const = 2*train_data.T @ (alpha_vector @ alpha_vector.T) @ (train_data) @ Q

    elif psi == 4:
        temp_alpha_vec = alpha_vector[:]
        temp_alpha_vec[temp_alpha_vec==C_val] = 0
        const = 2*np.transpose(train_data) @ (temp_alpha_vec @ temp_alpha_vec.T) @ (train_data) @ Q

    else:
        print("Only psi 1,2,3 or 4 is possible")
        const = None

    return const