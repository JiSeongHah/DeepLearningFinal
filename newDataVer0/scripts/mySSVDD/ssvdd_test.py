import numpy as np
from SSVDD.baseSVDD.src.BaseSVDD import BaseSVDD
from sklearn.metrics.pairwise import rbf_kernel


def ssvdd_test(test_data, test_labels, ssvdd_model, ssvdd_Q, ssvdd_npt):
    """Testing the model trained or being trained

    :param test_data: matrix of floats, testing data
    :param test_labels:  array of -1 or 1, labels for testing data; 1 for target (fraudulent) class
    :param ssvdd_model: model trained or being trained
    :param ssvdd_Q: projection matrix
    :param ssvdd_npt: list of values used for npt-based testing for non-linear model
    :return: pred_labels, labels predicted for testing data
    """

    if ssvdd_npt[0] == 1:
        print("NPT-based SSVDD testing...")
        kernel_exp_test = rbf_kernel(X=ssvdd_npt[4], Y=test_data, gamma=ssvdd_npt[1])
        phi = ssvdd_npt[3]
        k_train = ssvdd_npt[2]
        N = np.shape(k_train)[1]
        M = np.shape(kernel_exp_test)[1]
        kernel_func_test = (np.identity(N) - np.ones(shape=(N,N))/N) @ \
                           (kernel_exp_test - (k_train @ np.ones(shape=(N,1))/N) @ np.ones(shape=(1,M)))
        test_data = (np.linalg.pinv(phi)@kernel_func_test).T


    else:
        print("Linear SSVDD testing...")

    testdata_red = test_data @ ssvdd_Q
    pred_labels = ssvdd_model.predict(testdata_red)
    
    anomaly_score= ssvdd_model.decision_function(testdata_red)
    return pred_labels,anomaly_score