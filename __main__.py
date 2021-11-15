import source.Data as Da
import source.pcanalysis as pca
import source.regression as re
import source.classification as cl


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
#    ran = range(0, 10)
#    for ra in ran:
#        delta_range = np.power(10., range(-2, 8))
#        re.least_squares_regression(data, 10, delta_range)
#    cl.logistic_regression(data, 10)
#    cl.k_nearest_neighbours(data, 10, 40)
#    cl.baseline(data, 5)
#    cl.two_layer_cross_validation(data, 10, 5, True)
    cl.mcnemera(data, 10, True)
#    cl.train_log_model(data, 5.689866029018293, True)

#    cl.test(data)
