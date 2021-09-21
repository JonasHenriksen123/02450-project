import numpy as np
import pandas as pd


class Data:
    __filename = './ressources/forestfires.csv'

    def __init__(self):
        df = pd.read_csv(self.__filename)

        raw_data = df.values

        cols = range(0, 12)

        self.X = raw_data[:, cols]

        self.attributes = np.asarray(df.columns[cols])

        classlabels = raw_data[:, -1]

        classnames = np.unique(classlabels)

        self.classdict = dict(zip(classnames, range(len(classnames))))

        self.classes = len(classnames)

        self.y = np.array([self.classdict[cl] for cl in classlabels])
