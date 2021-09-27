import source.Data as source
import numpy as np


def summary_stats(data: source.Data):
    
    mean = np.mean(data.X, axis=0).round(2)
    
    std = np.std(data.X, axis=0).round(2)
    
    min = np.min(data.X, axis=0)
    
    # Q1 = 
    
    median = np.median(data.X, axis=0)
    
    # Q3 = 
    
    max = np.max(data.X, axis=0)
    
    return mean, std, min, median, max

def outputGraph(data: source.Data):
    #do something with data
    print('nice graph')
