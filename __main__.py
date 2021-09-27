import source.Data as source
import source.pcanalysis as Pca
import numpy as np
if __name__ == '__main__':
    
    data = source.Data()
    
    print(data.X)
    
    mean, std, min, median, max = Pca.summary_stats(data)
    
    print(mean)
    print(std)
    print(min)
    print(median)
    print(max)
    
    
    # print(std)