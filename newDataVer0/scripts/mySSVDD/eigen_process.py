import numpy as np

def eigen_process(eig_val, eig_vec):
    """Extracting the positive eigen values and eigen vectors, sorting them and eliminating the non-valid
    (infinite and NaN) values from them

    :param eig_val: Eigen values
    :param eig_vec: Eigen vectors
    :return: eig_val, eig_vec; processed eigen values and vectors
    """

    if np.any(np.iscomplex(eig_val)):
        eig_val = np.abs(eig_val)
        eig_vec = np.abs(eig_vec)

    eig_val[~np.isfinite(eig_val)] = 0.0
    eig_vec[~np.isfinite(eig_vec)] = 0.0

    eig_val[eig_val < 1e-6] = 0.0

    positive_eigval_indices = eig_val > 0
    eig_val = eig_val[positive_eigval_indices]
    eig_vec = eig_vec[:, positive_eigval_indices]

    sorted_indices = np.argsort(-eig_val)
    eig_val = eig_val[sorted_indices]
    eig_vec = eig_vec[:, sorted_indices]

    if eig_val.size == 0:
        eig_val = np.array([0.0])
        eig_vec = np.zeros(len(eig_val), 1)

    return eig_val, eig_vec

