import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmbf(l: float, sigma: float, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m = trainX.shape[0]

    G = matrix([[gaussian_kernel(trainX[i], trainX[j], sigma) for j in range(m)] for i in range(m)], tc='d')
    zeros = spmatrix([], [], [], (m, m))
    H = sparse([[2 * l * G, zeros], [zeros, zeros]])
    A = define_A(G, trainy)
    u = matrix([0] * m + [1 / m] * m)
    v = matrix([1] * m + [0] * m, tc='d')
    sol = solvers.qp(H, u, -A, -v)
    alpha = np.array(sol['x'][:m])
    return alpha


def define_A(G: np.matrix, y: np.array):
    m = G.size[1]
    up_left = matrix([[y[i][0] * G[i, j] for j in range(m)] for i in range(m)], tc='d')
    up_right = spmatrix(1, range(m), range(m))
    down_right = spmatrix(1, range(m), range(m))
    down_left = spmatrix([], [], [], (m, m))
    A = sparse([[up_left, down_left], [up_right, down_right]])
    return A


def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(1 / (2 * sigma)) * (np.linalg.norm(x1 - x2) ** 2))


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvmbf(10, 0.1, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


def get_data():
    data = np.load('ex2q4_data.npz')
    return data['Xtrain'], data['Xtest'], data['Ytrain'], data['Ytest']


def task_4a():
    trainX, _, trainy, _ = get_data()
    colors = ['red' if trainy[i] == 1 else 'blue' for i in range(trainy.shape[0])]
    plt.scatter(trainX[:, 0], trainX[:, 1], c=colors)
    plt.title("Q4 Data")
    plt.show()


def task_4b():
    k_fold_softsvmbf(num_of_folds=5, sigmas=[0.01, 0.5, 1], ls=[1, 10, 100])
    k_fold_softsvm(num_of_folds=5, ls=[1, 10, 100])


def k_fold_softsvm(num_of_folds, ls):
    from softsvm import softsvm
    trainX, testX, trainy, testy = get_data()
    fold_size = int(trainX.shape[0] / num_of_folds)
    subsamples_inx = [fold_size * i for i in range(num_of_folds)]
    min_error = 1
    best_l = None
    for l in ls:
        accum_error = 0
        for inx in subsamples_inx:
            trainX_exclude, trainy_exclude = trainX[inx:inx + fold_size], trainy[inx:inx + fold_size]
            trainX_include = np.concatenate((trainX[:inx], trainX[inx + fold_size:]))
            trainy_include = np.concatenate((trainy[:inx], trainy[inx + fold_size:]))
            w = softsvm(l, trainX_include, trainy_include)
            preds = np.sign(trainX_exclude @ w)
            curr_error = np.mean(preds != trainy_exclude.T)
            accum_error += curr_error
        error = accum_error / num_of_folds
        if error < min_error:
            min_error = error
            best_l = l
        print(f"error of l = {l} is {error}")
    print(f"min error is {min_error}")
    w = softsvm(best_l, trainX, trainy)
    preds = np.sign(testX @ w)
    test_error = np.mean(preds != testy.T)
    print(f"Error for best l = {best_l} on test set is {test_error}")


def k_fold_softsvmbf(num_of_folds, sigmas, ls):
    trainX, testX, trainy, testy = get_data()
    fold_size = int(trainX.shape[0] / num_of_folds)
    subsamples_inx = [fold_size * i for i in range(num_of_folds)]
    min_error = 1
    best_sigma, best_l = None, None
    for sigma in sigmas:
        for l in ls:
            accum_error = 0
            for inx in subsamples_inx:
                trainX_exclude, trainy_exclude = trainX[inx:inx + fold_size], trainy[inx:inx + fold_size]
                trainX_include = np.concatenate((trainX[:inx], trainX[inx + fold_size:]))
                trainy_include = np.concatenate((trainy[:inx], trainy[inx + fold_size:]))
                alpha = softsvmbf(l, sigma, trainX_include, trainy_include)
                preds = np.sign([np.sum([alpha[i] * gaussian_kernel(trainX_include[i], x_new, sigma) for i in
                                         range(trainX_include.shape[0])]) for x_new in trainX_exclude])

                curr_error = np.mean(preds != trainy_exclude.T)
                accum_error += curr_error
            error = accum_error / num_of_folds
            if error < min_error:
                min_error = error
                best_sigma = sigma
                best_l = l
            print(f"error of sigma = {sigma}, l = {l} is {error}")
    print(f"min error is {min_error}")
    alpha = softsvmbf(best_l, best_sigma, trainX, trainy)
    preds = np.sign([np.sum([alpha[i] * gaussian_kernel(trainX[i], x_new, best_sigma) for i in
                             range(trainX.shape[0])]) for x_new in testX])

    test_error = np.mean(preds != testy.T)
    print(f"Error for sigma = {best_sigma}, l = {best_l} on test set is {test_error}")


def task_4d():
    l = 100
    sigmas = [0.01, 0.5, 0.1]
    trainX, _, trainy, _ = get_data()
    grid_size = 5
    x_axis = np.linspace(np.min(trainX[:, 0]) - 2, np.max(trainX[:, 0]) + 2, grid_size)
    y_axis = np.linspace(np.min(trainX[:, 1]) - 2, np.max(trainX[:, 1]) + 2, grid_size)
    grid = np.array([[[x_axis[i],y_axis[j]]for i in range(grid_size)] for j in range(grid_size)])
    for sigma in sigmas:
        alpha = softsvmbf(l, sigma, trainX, trainy)
        preds = np.zeros((grid_size,grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                preds[i][j] = np.sign(np.sum([alpha[k] * gaussian_kernel(trainX[k], grid[i][j], sigma) for k in range(trainX.shape[0])]))
        plt.contourf(x_axis, y_axis, preds)
        plt.show()
if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()
    # task_4a()
    # task_4b()
    task_4d()
    # here you may add any code that uses the above functions to solve question 4
