import string

import matplotlib.pyplot as plt
import numpy as np

import source.Data as Da
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show


def plot_attribute_against(data: Da.Data, attribute_x: int, attribute_y: int, plot_title: string):
    cols = [attribute_x, attribute_y]
    col_data = data.get_columns(cols)

    f = figure()
    title(plot_title)
    plot(col_data[0], col_data[1], 'o', alpha=.1)

    xlabel(data.attributes[attribute_x])
    ylabel(data.attributes[attribute_y])
    show()


def plot_visualized_data(data: Da.Data):
    x = data.x
    x_tilda = x - np.ones((data.data_count, 1)) * x.mean(axis=0)
    x_tilda = x_tilda * (1 / np.std(x_tilda, 0))

    u, s, vh = svd(x_tilda, full_matrices=False)

    rho = (s * s) / (s * s).sum()

    threshold = 0.90

    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, 'x-')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.legend('Individual', 'Cumulative', 'Threshold')
    plt.grid()
    plt.show()
