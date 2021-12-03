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


class Classifier:
    def __init__(self, k, x_train, y_train):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x):
        return np.transpose(np.array([[self.calc_knn(i) for i in x]]))

    def calc_knn(self, x):
        distances_from_train = np.array([np.linalg.norm(x_i - x) for x_i in self.x_train])
        knn_idx = np.argsort(distances_from_train)[:self.k]
        knn_labels = np.array([int(self.y_train[i]) for i in knn_idx])
        majority_label = np.argmax(np.bincount(knn_labels))
        return majority_label


def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    classifier = Classifier(k, x_train, y_train)
    return classifier


def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    return classifier.predict(x_test)


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train1']
    train1 = data['train3']
    train2 = data['train4']
    train3 = data['train6']

    test0 = data['test1']
    test1 = data['test3']
    test2 = data['test4']
    test3 = data['test6']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [1, 3, 4, 6], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [1, 3, 4, 6], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)
    print(calc_error(y_test, preds))
    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def calc_error(y_true, pred):
    return np.mean(y_true != pred.T)


def extract_data():
    data = np.load('mnist_all.npz')
    return data['train1'], data['train3'], data['train4'], data['train6'], data['test1'], data['test3'], data['test4'], \
           data['test6']


def task_2a():
    samples = [i for i in range(10, 110, 10)]
    train1, train3, train4, train6, test1, test3, test4, test6 = extract_data()
    errors, max_errors, min_errors = [], [], []
    for sample in samples:
        max_error = 0
        min_error = 1
        accum_error = 0
        for i in range(10):
            x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6], sample)
            x_test, y_test = gensmallm([test1, test3, test4, test6], [1, 3, 4, 6], 50)
            classifer = learnknn(1, x_train, y_train)

            preds = predictknn(classifer, x_test)
            error = calc_error(y_test, preds)
            accum_error += error
            if error > max_error:
                max_error = error
            if error < min_error:
                min_error = error
        errors.append(accum_error / 10)
        max_errors.append(max_error)
        min_errors.append(min_error)
    plot_results(errors, max_errors, min_errors, "NN", "Sample Size", "Error", samples, 10)


def task_2e():
    ks = [i for i in range(1, 11)]
    train1, train3, train4, train6, test1, test3, test4, test6 = extract_data()
    errors, max_errors, min_errors = [], [], []
    for k in ks:
        max_error = 0
        min_error = 1
        accum_error = 0
        for i in range(10):
            x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6], 100)
            x_test, y_test = gensmallm([test1, test3, test4, test6], [1, 3, 4, 6], 50)
            classifer = learnknn(k, x_train, y_train)

            preds = predictknn(classifer, x_test)
            error = calc_error(y_test, preds)
            accum_error += error
            if error > max_error:
                max_error = error
            if error < min_error:
                min_error = error
        errors.append(accum_error / 10)
        max_errors.append(max_error)
        min_errors.append(min_error)
    plot_results(errors, max_errors, min_errors, "KNN", "K", "Error", ks, 1)


def corrupt_training_set(y_train):
    import random
    size = int(np.floor(y_train.shape[0] * 0.2))
    indexes = random.sample(range(y_train.shape[0]), size)
    for i in indexes:
        current_label = y_train[i]
        y_train[i] = random.sample(list({1, 3, 4, 6} - {current_label}), 1)[0]
    return y_train


def task_2f():
    ks = [i for i in range(1, 11)]
    train1, train3, train4, train6, test1, test3, test4, test6 = extract_data()
    errors, max_errors, min_errors = [], [], []
    for k in ks:
        max_error = 0
        min_error = 1
        accum_error = 0
        for i in range(10):
            x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6], 100)
            y_train = corrupt_training_set(y_train)
            x_test, y_test = gensmallm([test1, test3, test4, test6], [1, 3, 4, 6], 50)
            classifer = learnknn(k, x_train, y_train)

            preds = predictknn(classifer, x_test)
            error = calc_error(y_test, preds)
            accum_error += error
            if error > max_error:
                max_error = error
            if error < min_error:
                min_error = error
        errors.append(accum_error / 10)
        max_errors.append(max_error)
        min_errors.append(min_error)
    plot_results(errors, max_errors, min_errors, "KNN corrupted", "K", "Error", ks, 1)


def plot_results(errors, max_errors, min_errors, title, x_label, y_label, param_change, iter_diff):
    plt.title(title)
    plt.xticks(np.arange(min(param_change), max(param_change) + 1, iter_diff))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(param_change, errors, marker='o', color='b', label='line with marker')
    for i in range(len(param_change)):
        param = param_change[i]
        y = max_errors[i]
        plt.plot(param, y, marker='o', linestyle='None', color='r')
        plt.text(param * (1 + 0.01), y * (1 + 0.01), y, fontsize=9)
        y = min_errors[i]
        plt.plot(param, y, marker='o', linestyle='None', color='g')
        plt.text(param * (1 + 0.01), y * (1 + 0.01), y, fontsize=9)
    plt.legend(["avg err", "max err", "min err"])
    plt.show()


if __name__ == '__main__':
    simple_test()
    task_2a()
    task_2e()
    task_2f()
