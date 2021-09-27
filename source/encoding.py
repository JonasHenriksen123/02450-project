#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:14:31 2021

@author: filippo
"""

import numpy as np

def k_encoding(X,Y):
    # convert nominal attributes to ordinal
    # ONE-OUT-OF-k-ENCODINGS
    
    month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    monthDict = dict(zip(month, range(1, len(month) + 1)))
    week = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    weekDict = dict(zip(week, range(1, len(week) + 1)))

    X = np.array([monthDict[m] for m in X])
    Y = np.array([weekDict[d] for d in Y])
    return np.stack((X, Y), axis=1)

# def k_encoding(doc):
#     month = doc.col_values(2, 1)
#     week = doc.col_values(3,1)
#     X_second= k_encoding(month, week)
#     X_first = np.empty((517, 2))
#     for i, col_id in enumerate(range(2)):
#         X_first[:, i] = np.asarray(doc.col_values(col_id, 1, 518))
#     # for i, col_id in enumerate(cols):
#     #     self.X[:, i] = np.asarray(doc.col_values(col_id, 1, 518))
#     X_third= np.empty((517, 8))
#     for i, col_id in enumerate(range(4, 12)):
#         X_third[:, i] = np.asarray(doc.col_values(col_id, 1, 518))
#     return np.hstack((X_first,X_second,X_third))

# def k_encoding(doc):
#     # convert nominal attributes to ordinal
#     # ONE-OUT-OF-k-ENCODINGS
    
#     month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
#     monthDict = dict(zip(month, range(1, len(month) + 1)))
#     week = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
#     weekDict = dict(zip(week, range(1, len(week) + 1)))
    
#     month = doc.col_values(2, 1)
#     week = doc.col_values(3,1)
#     X_second= k_encoding(month, week)
#     X_first = np.empty((517, 2))
#     for i, col_id in enumerate(range(2)):
#         X_first[:, i] = np.asarray(doc.col_values(col_id, 1, 518))
#     # for i, col_id in enumerate(cols):
#     #     self.X[:, i] = np.asarray(doc.col_values(col_id, 1, 518))
#     X_third= np.empty((517, 8))
#     for i, col_id in enumerate(range(4, 12)):
#         X_third[:, i] = np.asarray(doc.col_values(col_id, 1, 518))
#     return np.hstack((X_first,X_second,X_third))