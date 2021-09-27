import numpy as np
import pandas as pd
import xlrd
from pathlib import Path
from source.encoding import k_encoding


class Data:
    __filename = Path(__file__).parent / './ressources/forestfires.csv'

    def __init__(self):
        # df = pd.read_csv(self.__filename)
        # raw_data = df.values
        cols = range(0, 12)
        # self.X = raw_data[:,cols]
        # self.attributes = np.asarray(df.columns[cols])
        
        
        doc = xlrd.open_workbook(Path(__file__).parent / 'ressources/forestfires.xls').sheet_by_index(0)
        self.attributes = doc.row_values(0, 0, 12)
        
        self.X = np.empty((517,12))
        month = doc.col_values(2, 1)
        week = doc.col_values(3,1)
        X_second= k_encoding(month, week)
        X_first = np.empty((517, 2))
        for i, col_id in enumerate(range(2)):
            X_first[:, i] = np.asarray(doc.col_values(col_id, 1, 518))
        # for i, col_id in enumerate(cols):
        #     self.X[:, i] = np.asarray(doc.col_values(col_id, 1, 518))
        X_third= np.empty((517, 8))
        for i, col_id in enumerate(range(4, 12)):
            X_third[:, i] = np.asarray(doc.col_values(col_id, 1, 518))
        self.X = np.hstack((X_first,X_second,X_third))       
        # self.X[:,2:4] = k_encoding(self.X[:,2],self.X[:,3])
        
        
        self.N, self.M = self.X.shape
    
        # classlabels = raw_data[:, -1]

        # classnames = np.unique(classlabels)

        # self.class_count = len(classnames)

        # self.class_dict = dict(zip(classnames, range(self.class_count)))

        # self.y = np.array([self.class_dict[cl] for cl in classlabels])
