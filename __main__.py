import source.Data as Da
import source.pcanalysis as pca

if __name__ == '__main__':

    data = Da.Data()

    #pca.plot_attribute_against(data, 4, 7, '')

    #pca.plot_visualized_data(data, '')

    pca.plot_visualized_coefficients(data, 3, '', legend=True)

    #pca.plot_boxplots(data, 'forestfires boxplots')

    #pca.plot_boxplot(data, 6, 'forestfires boxplot DC')
