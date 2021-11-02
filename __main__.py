import source.Data as Da
import source.pcanalysis as pca
import source.regression as re
import numpy as np

if __name__ == '__main__':

    # fetch data through data object
    data = Da.Data()

    # region generate plots
#     pca.plot_attribute_against(data, 4, 7, 'forestfires FFMC v ISI')

#     pca.plot_visualized_data(data, 'forestfires principal component explained variance')

#     pca.plot_visualized_coefficients(data, 3, 'forestfires pca coefficients visualized', legend=True)

#     pca.plot_boxplots(data, 'forestfires boxplots')

#     pca.plot_boxplot(data, 6, 'forestfires boxplot DC')
#     # endregion
    ran = range(0, 10)
    for ra in ran:
        delta_range = np.power(10., range(-2, 8))
        re.least_squares_regression(data, 10, delta_range)
