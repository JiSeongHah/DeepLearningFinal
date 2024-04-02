import numpy as np
import os
import sys

from mySSVDD.BaseSVDD import BaseSVDD
from mySSVDD.eigen_process import eigen_process
from mySSVDD.constraints_ssvdd import constraints_ssvdd

def ssvdd_train(x_train, y_train, iter, C, d, eta, kappa, beta, psi, npt):
    """ Train the SSVDD model for the given data and hyper-parameters

    :param x_train: matrix of floats, training data
    :param y_train: array of -1 or 1, labels for training data; 1 for target (fraudulent) class
    :param iter: int, number of iterations for robust training
    :param C: float, determining the proportion of instances forced to be outliers (hyper-parameter)
    :param d: int, d-dimensions for subspace learning (hyper-paramter)
    :param eta: float, determining the step of gradient (hyper-paramter)
    :param kappa: float, specifying the width of kernel function (hyper-paramter)
    :param beta: float, regularizing the psi parameter (hyper-paramter)
    :param psi: int, specifying the type of regularization term (in constraint function) for training
    :param npt: int; 0 for linear version, 1 for non-linear version of model
    :return: ssvdd_npt - list of values used for npt-based testing for non-linear model
             ssvdd_models - list of models for each iteration
             ssvdd_Q - projection matrix tuned in each iteration
    """
    

    if npt==1:
        print("NPT-based SSVDD running...")
        z=x_train.T
        N = np.shape(x_train)[0]
        dtrain = np.sum(x_train**2,1).reshape(N,1) @ np.ones(shape=(1,N)) + \
                 (np.sum(x_train**2,1).reshape(N,1)@np.ones(shape=(1,N))).T - \
                 (2*(x_train@z))
        sigma = kappa * np.mean(np.mean(dtrain))
        A = 2.0*sigma
        ktrain_exp = np.exp(-dtrain/A)
        N = np.shape(ktrain_exp)[0]
        ktrain = (np.identity(N)-np.ones(shape=(N,N))/N) @ ktrain_exp @ (np.identity(N)-np.ones(shape=(N,N))/N)

        eig_val, eig_vec = np.linalg.eig(ktrain)
        eig_val, eig_vec = eigen_process(eig_val, eig_vec)
        eigval_acc = np.cumsum(eig_val) / np.sum(eig_val)
        eig_diag = np.diag(eig_val)   # pos_eigvalue incase of alternative
        II = np.argwhere(eigval_acc >= 0.99)
        LL = II[0][0]
        eig_diag = eig_diag[0:LL, 0:LL]
        pmat = np.linalg.pinv(np.sqrt(eig_diag) @ eig_vec[:, 0:LL].T)     # pos_eigvalue incase of alternative
        phi = ktrain @ pmat
        ssvdd_npt = [1,A,ktrain_exp,phi,x_train]
        x_train = phi
    else:
        print("Linear SSVDD running...")
        ssvdd_npt = [0]  # flag is 0


    Q = np.random.rand(np.shape(x_train)[1], d)
    Q, _ = np.linalg.qr(Q)
    reduced_data = x_train@Q
    model = BaseSVDD(C=C, kernel='rbf', display='off')
    svdd_model = model.fit(reduced_data, y_train)
    ssvdd_Q = [Q]
    ssvdd_models = [model]

    for _ in range(iter):
        alpha_vector  = np.array(svdd_model.alpha)
        const = constraints_ssvdd(psi, C, Q, x_train, alpha_vector)
        Sa = x_train.T @ (np.diag(alpha_vector) - (alpha_vector @ alpha_vector.T)) @ x_train
        grad = 2 * (Sa @ Q) + (beta*const)
        Q = Q - eta*grad
        Q, _ = np.linalg.qr(Q)
        reduced_data = x_train @ Q

        model = BaseSVDD(C=C, kernel='rbf', display='off')
        svdd_model = model.fit(reduced_data, y_train)

        ssvdd_Q.append(Q)
        ssvdd_models.append(model)

    return ssvdd_npt, ssvdd_models, ssvdd_Q

