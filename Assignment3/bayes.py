import numpy as np
import matplotlib.pyplot as plt


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


def bayeslearn(x_train: np.array, y_train: np.array):
    """

    :param x_train: 2D numpy array of size (m, d) containing the the training set. The training samples should be binarized
    :param y_train: numpy array of size (m, 1) containing the labels of the training set
    :return: a triple (allpos, ppos, pneg) the estimated conditional probabilities to use in the Bayes predictor
    """
    num_of_features = x_train.shape[1]
    allpos = (y_train > 0).sum() / y_train.shape[0]
    pos_y = np.where(y_train == 1)[0]
    if len(pos_y) == 0:
        ppos = np.full([num_of_features, 1], np.nan)
    else:
        ppos = (np.sum(x_train[pos_y], axis=0) / len(pos_y)).reshape(num_of_features, 1)

    neg_y = np.where(y_train == -1)[0]
    if len(neg_y) == 0:
        pneg = np.full([num_of_features, 1], np.nan)
    else:
        pneg = (np.sum(x_train[neg_y], axis=0) / len(neg_y)).reshape(num_of_features, 1)

    return allpos, ppos, pneg


def bayespredict(allpos: float, ppos: np.array, pneg: np.array, x_test: np.array):
    """

    :param allpos: scalar between 0 and 1, indicating the fraction of positive labels in the training sample
    :param ppos: numpy array of size (d, 1) containing the empirical plug-in estimate of the positive conditional probabilities
    :param pneg: numpy array of size (d, 1) containing the empirical plug-in estimate of the negative conditional probabilities
    :param x_test: numpy array of size (n, d) containing the test samples
    :return: numpy array of size (n, 1) containing the predicted labels of the test samples
    """
    preds = np.sign([compute_row(row, allpos, ppos, pneg) for row in x_test])
    return preds


def compute_row(row, allpos, ppos, pneg):
    return np.log(allpos / (1 - allpos)) + np.sum(
        [np.log(ppos[i][0] / pneg[i][0]) if row[i] == 1 else -1 * np.log((1 - pneg[i]) / (1 - ppos[i])) for i in
         range(len(row))])


def simple_test():
    # load sample data from question 2, digits 3 and 5 (this is just an example code, don't forget the other part of
    # the question)
    data = np.load('mnist_all.npz')

    train3 = data['train3']
    train5 = data['train5']

    test3 = data['test3']
    test5 = data['test5']

    m = 500
    n = 50
    d = train3.shape[1]

    x_train, y_train = gensmallm([train3, train5], [-1, 1], m)

    x_test, y_test = gensmallm([test3, test5], [-1, 1], n)

    # threshold the images (binarization)
    threshold = 128
    x_train = np.where(x_train > threshold, 1, 0)
    x_test = np.where(x_test > threshold, 1, 0)

    # run naive bayes algorithm
    allpos, ppos, pneg = bayeslearn(x_train, y_train)

    assert isinstance(ppos, np.ndarray) \
           and isinstance(pneg, np.ndarray), "ppos and pneg should be numpy arrays"

    assert 0 <= allpos <= 1, "allpos should be a float between 0 and 1"

    y_predict = bayespredict(allpos, ppos, pneg, x_test)

    assert isinstance(y_predict, np.ndarray), "The output of the function bayespredict should be numpy arrays"
    assert y_predict.shape == (n, 1), f"The output of bayespredict should be of size ({n}, 1)"

    print(f"Prediction error = {np.mean(y_test.reshape(n, 1) != y_predict)}")


def extract_data(first_num, sec_num):
    data = np.load('mnist_all.npz')

    train_first = data[f'train{first_num}']
    train_sec = data[f'train{sec_num}']

    test_first = data[f'test{first_num}']
    test_sec = data[f'test{sec_num}']

    return train_first, test_first, train_sec, test_sec


def task_2a():
    train_sizes = [i * 1000 for i in range(1, 11)]
    for first, sec in [(3,5),(0,1)]:
        errors = []
        for train_size in train_sizes:
            train3, test3, train5, test5 = extract_data(first, sec)
            x_train, y_train = gensmallm([train3, train5], [-1, 1], train_size)
            x_test, y_test = gensmallm([test3, test5], [-1, 1], test3.shape[0] + test5.shape[0])
            threshold = 128
            x_train = np.where(x_train > threshold, 1, 0)
            x_test = np.where(x_test > threshold, 1, 0)
            allpos, ppos, pneg = bayeslearn(x_train, y_train)
            y_predict = bayespredict(allpos, ppos, pneg, x_test)
            errors.append(np.mean(y_test.reshape(test3.shape[0] + test5.shape[0], 1) != y_predict))
        plt.plot(train_sizes, errors)
    plt.legend(["3 vs 5", "0 vs 1"])
    plt.xlabel("sample size")
    plt.ylabel("error")
    plt.show()


def task_2c():
    train0, _, train1, _ = extract_data(0,1)
    x_train, y_train = gensmallm([train0, train1], [-1, 1], 10000)
    threshold = 128
    x_train = np.where(x_train > threshold, 1, 0)
    _, ppos, pneg = bayeslearn(x_train, y_train)
    im = plt.imshow(ppos.reshape(28,28), cmap='hot')
    plt.colorbar(im)
    plt.title("ppos heatmap")
    plt.show()
    plt.imshow(pneg.reshape(28,28), cmap='hot')
    plt.title("pneg heatmap")
    plt.show()


def task_2d():
    for first, sec in [(0,1),(3,5)]:
        train_first, test_first, train_sec, test_sec = extract_data(first,sec)
        x_train, y_train = gensmallm([train_first, train_sec], [-1, 1], 10000)
        threshold = 128
        x_train = np.where(x_train > threshold, 1, 0)
        allpos, ppos, pneg = bayeslearn(x_train, y_train)
        x_test, y_test = gensmallm([test_first, test_sec], [-1, 1], test_first.shape[0] + test_sec.shape[0])
        x_test = np.where(x_test > threshold, 1, 0)
        y_origin_allpos = bayespredict(allpos, ppos, pneg, x_test)
        y_new_allpos = bayespredict(0.75, ppos, pneg, x_test)
        err_orig_allpos = np.mean(y_test.reshape(test_first.shape[0] + test_sec.shape[0], 1) != y_origin_allpos)
        err_new_allpos = np.mean(y_test.reshape(test_first.shape[0] + test_sec.shape[0], 1) != y_new_allpos)
        
        print(f"[{first},{sec}] :error with original allpos {allpos} is {err_orig_allpos} error with allpos 0.75 is {err_new_allpos}")

        changed_from_1 = len(np.where(y_origin_allpos[np.where(y_new_allpos == -1)[0]] == 1)[0])/ len(np.where(y_origin_allpos == 1)[0])
        print(f"% of cahnge from 1 to -1: {changed_from_1}")

        changed_to_1 = len(np.where(y_origin_allpos[np.where(y_new_allpos == 1)[0]] == -1)[0])/ len(np.where(y_origin_allpos == -1)[0])
        print(f"% of cahnge from -1 to 1: {changed_to_1}")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()
    # here you may add any code that uses the above functions to solve question 2
    task_2d()