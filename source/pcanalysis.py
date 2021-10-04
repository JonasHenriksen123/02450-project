import string

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, boxplot, xticks
from scipy.linalg import svd

import source.Data as Da


def plot_attribute_against(data: Da.Data, attribute_x: int, attribute_y: int, plot_title: string):
    f = figure()
    title(plot_title)
    for c in range(data.C):
        class_mask = data.y == c
        plot(data.y2[class_mask,
             attribute_x],
             data.y2[class_mask,
             attribute_y],
             'o',
             alpha=1,
             marker='.',
             markersize=5)

    xlabel(data.attributes[attribute_x])
    ylabel(data.attributes[attribute_y])

    show()


def plot_visualized_data(data: Da.Data, plot_title: string):
    x_tilda = data.y2

    print(data.y2.shape)

    u, s, vh = svd(x_tilda, full_matrices=False)

    rho = (s * s) / (s * s).sum()

    threshold = 0.90

    plt.figure()
    plt.title(plot_title)
    plt.plot(range(1, len(rho) + 1), rho, 'x-')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title(plot_title)
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.show()


def plot_visualized_pca(data: Da.Data, first_pc: int, second_pc: int, plot_title: string):
    x_tilda = data.y2

    u, s, vh = svd(x_tilda, full_matrices=False)
    v = vh.T
    z = data.y2 @ v

    plt.figure()
    plt.title(plot_title)
    for c in range(data.C):
        class_mask = data.y == c
        plot(z[class_mask, first_pc], z[class_mask, second_pc], 'o', alpha=1, marker='.', markersize=5)
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
    r = np.arange(1, data.M)
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


def plot_boxplots(data: Da.Data, plot_title: string):
    boxplot(data.y2)
    xticks(range(1, data.M), data.attributes)
    ylabel('y')
    title(plot_title)
    show()


def plot_boxplot(data: Da.Data, attr: int, plot_title: string):
    boxplot(data.x[:, attr])
    xticks(range(1, 2), [data.attributes[attr]])
    ylabel(data.attribute_units[attr])
    title(plot_title)
    show()
