import source.Data as Da
import source.pcanalysis as pca

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

    pca.plot_box_plot(data)
    
    pca.plot_distribution(data)
  
    pca.plot_correlation_matrix(data)    

    pca.plot_PCA(data)

    pca.plot_pca_coeff(data)
    
    pca.plot_cum_variance(data)