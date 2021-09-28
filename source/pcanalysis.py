import string

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show
from scipy.linalg import svd

import source.Data as Da


def plot_attribute_against(data: Da.Data, attribute_x: int, attribute_y: int, plot_title: string):
    cols = [attribute_x, attribute_y]
    col_data = data.get_columns(cols)

    f = figure()
    title(plot_title)
    plot(col_data[0], col_data[1], 'o', alpha=.1)

    xlabel(data.attributes[attribute_x])
    ylabel(data.attributes[attribute_y])
    show()


def plot_visualized_data(data: Da.Data, plot_title: string):
    x_tilda = data.y2

    u, s, vh = svd(x_tilda, full_matrices=False)

    rho = (s * s) / (s * s).sum()

    threshold = 0.90

    plt.figure()
    plt.title(plot_title)
    plt.plot(range(1, len(rho) + 1), rho, 'x-')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.show()


def plot_visualized_pca(data: Da.Data, first_pc: int, second_pc: int, plot_title: string):
    x_tilda = data.y2

    u, s, vh = svd(x_tilda, full_matrices=False)
    v = vh.T
    z = data.x @ v

    plt.figure()
    plt.title(plot_title)
    plot(z[:, first_pc], z[:, second_pc], 'o', alpha=0.3)
    xlabel('PC{0}'.format(first_pc + 1))
    ylabel('PC{0}'.format(second_pc + 1))

    show()


def plot_visualized_coefficients(data: Da.Data, pc_count: int, plot_title: string, legend: bool = True):
    pcs = [i for i in range(0, pc_count)]
    legend_strs = ['PC' + str(e + 1) for e in pcs]

    x_tilda = data.y2

    u, s, vh = svd(x_tilda, full_matrices=False)
    v = vh.T

    bw = .2
    r = np.arange(1, data.M + 1)
    for i in pcs:
        plt.bar(r + i * bw, v[:, i], width=bw)

    plt.xticks(r + bw, data.attributes)
    plt.xlabel('Attributes')
    plt.ylabel('Component coefficients')

    if legend:
        plt.legend(legend_strs)

    plt.grid()
    plt.title(plot_title)
    plt.show()
