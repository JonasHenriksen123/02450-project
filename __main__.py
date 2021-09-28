import source.Data as Da
import source.pcanalysis as pca

if __name__ == '__main__':

    data = Da.Data()

    pca.plot_attribute_against(data, 4, 5, 'forestfires data')

    pca.plot_visualized_data(data, 'forestfires variance visualized')

    pca.plot_visualized_pca(data, 0, 1, 'forestfires pca visualization')

    pca.plot_visualized_coefficients(data, 3, 'forestfires coefficient visualisation', legend=True)
