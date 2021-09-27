import source.Data as Da
import source.pcanalysis as pca

if __name__ == '__main__':

    data = Da.Data()

    pca.plot_attribute_against(data, 0, 1, 'forestfires data')

    print(data.x)

    print(data.mean)
    print(data.std)
    print(data.min)
    print(data.median)
    print(data.max)
