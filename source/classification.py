import numpy as np
from sklearn import model_selection
import source.Data as Da
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid, figure, plot, xlabel, ylabel, legend, ylim, show)
import sklearn.linear_model as lm


def logistic_regression(data: Da.Data, k_folds: int):
    X = data.x
    y = data.y

    model = lm.LogisticRegression()
    model = model.fit(X, y)

    # Classify wine as White/Red (0/1) and assess probabilities
    y_est = model.predict(X)
    y_est_white_prob = model.predict_proba(X)[:, 0]

    f = figure()
    class0_ids = np.nonzero(y == 0)[0].tolist()
    plot(class0_ids, y_est_white_prob[class0_ids], '.y')
    class1_ids = np.nonzero(y == 1)[0].tolist()
    plot(class1_ids, y_est_white_prob[class1_ids], '.r')
    xlabel('Data object (Data point)')
    ylabel('Predicted prob. of Class 0')
    legend(['Class 1', 'Class 0'])
    ylim(-0.01, 1)

    show()


def k_nearest_neighbours(data: Da.Data, max_neighbours: int):
    X = data.x2
    y = data.y
    N = data.N

    CV = model_selection.LeaveOneOut()
    errors = np.zeros((N, max_neighbours))
    i = 0
    for train_index, test_index in CV.split(X, y):
        print('Crossvalidation fold: {0}/{1}'.format(i + 1, N))

        # extract training and test set for current CV fold
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]

        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1, max_neighbours + 1):
            knclassifier = KNeighborsClassifier(n_neighbors=l)
            knclassifier.fit(X_train, y_train)
            y_est = knclassifier.predict(X_test)
            errors[i, l - 1] = np.sum(y_est[0] != y_test[0])

        i += 1

    # Plot the classification error rate
    figure()
    plt.plot(100 * sum(errors, 0) / N)
    xlabel('Number of neighbors')
    ylabel('Classification error rate (%)')
    show()


def baseline(data: Da.Data, k_folds: int):
    X = data.x2
    y = data.y
    N = data.N

    CV = model_selection.KFold(k_folds, shuffle=True)
    errors = []

    i = 0
    for train_index, test_index in CV.split(X, y):
        # extract training and test set for current CV fold
        y_train = y[train_index]
        y_test = y[test_index]

        # fit classifier
        c0 = 0
        c1 = 0
        for c in y_train:
            if c == 0:
                c0 = c0 + 1
            else:
                c1 = c1 + 1

        if c0 > c1:
            c = 0
        else:
            c = 1

        err = 0
        for yh in y_test:
            if c != yh:
                err = err + 1

        errors.append(100 * err / N)

        i += 1

    print(errors)
