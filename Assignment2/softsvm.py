import cvxopt
import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def softsvm(l, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    m, d = trainX.shape
    H = spmatrix([2 * int(l)] * d, range(d), range(d), size=(m + d, m + d))
    # epsilon = spmatrix(10^12, range(d), range(d), size = H.size)
    # H = H + epsilon
    A = define_A(trainX, trainy)
    u = matrix([0] * d + [1 / m] * m)
    v = matrix([1] * m + [0] * m, tc='d')
    sol = solvers.qp(H, u, -A, -v)
    w = np.array(sol['x'][:d])
    return w


def define_A(X: np.array, y: np.array):
    m, d = X.shape
    down_left = spmatrix([], [], [], size=(m, d))
    up_right = spmatrix(1, range(m), range(m))

    up_left = matrix(np.array([[y[i] * X[i][j] for j in range(d)] for i in range(m)]))
    down_right = spmatrix(1, range(m), range(m))

    A = sparse([[up_left, down_left], [up_right, down_right]])
    return A


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


def get_data():
    data = np.load('EX2q2_mnist.npz')
    return data['Xtrain'], data['Xtest'], data['Ytrain'], data['Ytest']


def get_train_sample(size, trainX, trainy):
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:size]]
    _trainy = trainy[indices[:size]]
    return _trainX, _trainy


def experiment(train_size, ls, num_of_iterations):
    trainX, testX, trainy, testy = get_data()
    avg_errors_train, max_errors_train, min_errors_train = [], [], []
    avg_errors_test, max_errors_test, min_errors_test = [], [], []
    for l in ls:
        curr_max_train, curr_min_train = 0, 1
        curr_max_test, curr_min_test = 0, 1
        accum_error_train, accum_error_test = 0, 0
        for iter in range(num_of_iterations):
            _trainX, _trainy = get_train_sample(train_size, trainX, trainy)
            w = softsvm(l, _trainX, _trainy)
            train_pred = np.sign(_trainX @ w)
            test_pred = np.sign(testX @ w)
            train_error = np.mean(_trainy != train_pred.T)
            test_error = np.mean(testy != test_pred.T)

            accum_error_train += train_error
            accum_error_test += test_error

            curr_min_train = curr_min_train if curr_min_train < train_error else train_error
            curr_max_train = curr_max_train if curr_max_train > train_error else train_error
            curr_min_test = curr_min_test if curr_min_test < test_error else test_error
            curr_max_test = curr_max_test if curr_max_test > test_error else test_error

        min_errors_train.append(curr_min_train)
        max_errors_train.append(curr_max_train)
        avg_errors_train.append(accum_error_train/num_of_iterations)
        min_errors_test.append(curr_min_test)
        max_errors_test.append(curr_max_test)
        avg_errors_test.append(accum_error_test/num_of_iterations)

    ax = plt.axes()
    ax.set_xscale("log")
    plt.title(f"Average Error Soft Svm with QP. train size= {train_size}")
    plt.xlabel("Lambda in semilog scale")
    plt.ylabel("Error")
    plt.errorbar(ls,avg_errors_test, yerr = (min_errors_test, max_errors_test))
    plt.errorbar(ls,avg_errors_train, yerr = (min_errors_train, max_errors_train))
    plt.legend(["test set", "train set"])
    plt.show()

if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()

    # here you may add any code that uses the above functions to solve question 2
    experiment(100,[10 ** i for i in range(10)], 10)
    experiment(1000,[10 ** i for i in [1,3,5,8]], 1)
