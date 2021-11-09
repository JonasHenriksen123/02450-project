# ========================================================================================================================================
# Importing packages
# ========================================================================================================================================
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
import seaborn as sns
import cufflinks as cf
import plotly as py
import torch
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
py.offline.init_notebook_mode(connected = True)
cf.go_offline()
sns.set()
import sklearn
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from toolbox_02450 import rlr_validate
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary, correlated_ttest, rlr_validate
from datetime import datetime

import source.Data as Da

# import source.Data as Da
def regression():
    
    data = Da.Data()
    ## ========================================================================================================================================
    # Separating the data in X and y format. Also applying K-fold cross validation
    # ========================================================================================================================================
    X = data.df
    y = X['area']
    X = X.drop('area',axis=1)
    # X = np.asarray(X).astype('float64')
    y = np.array(y)
    yTrans = np.log(y+1)
    # y = yTrans.astype('float64')    
      
    # Add offset attribute
    X = np.concatenate((np.ones((X.shape[0],1)),X),1)
    attributeNames = ['Offset'] + data.df_attributes
    
    N, M = X.shape
    K = 10
    random_seed = 0
    CV = model_selection.KFold(K, shuffle=True, random_state=random_seed)
    # CV = model_selection.KFold(K, shuffle=False)
    
    # Values of lambda
    lambdas = np.power(10.,range(-2,10))
    
    # Initialize variables
    #T = len(lambdas)
    Error_train = np.empty((K,1))
    Error_test = np.empty((K,1))
    Error_train_rlr = np.empty((K,1))
    Error_test_rlr = np.empty((K,1))
    Error_train_nofeatures = np.empty((K,1))
    Error_test_nofeatures = np.empty((K,1))
    Error_test_baseline = np.empty((K,1))
    w_rlr = np.empty((M,K))
    mu = np.empty((K, M-1))
    sigma = np.empty((K, M-1))
    w_noreg = np.empty((M,K))
    
    Arr_opt_lambda = np.empty((K,1))
    Arr_mean_w_vs_lambda = []
    Arr_train_err_vs_lambda = []
    Arr_test_err_vs_lambda = []
    
    
    k=0
    y_estimated_no_features = []
    for train_index, test_index in CV.split(X,y):
    
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index] 
        internal_cross_validation = 10
        #  RETURNED VALUE FROM rlr_validate
        # opt_val_err = np.min(np.mean(test_error,axis=0))
        '''The optimal lambda is choosen as the one that give the lowest mean test error'''
        # opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]         
        # train_err_vs_lambda = np.mean(train_error,axis=0)
        # test_err_vs_lambda = np.mean(test_error,axis=0)
        # mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))

        # print(X_train)
        # print(y_train)

        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
        print(opt_lambda)
        Arr_opt_lambda[k] = opt_lambda
        Arr_mean_w_vs_lambda.append(mean_w_vs_lambda)
        Arr_train_err_vs_lambda.append(train_err_vs_lambda)
        Arr_test_err_vs_lambda.append(test_err_vs_lambda)
        
        # Standardize outer fold based on training set, and save the mean and standard
        # deviations since they're part of the model (they would be needed for
        # making new predictions) - for brevity we won't always store these in the scripts
        mu[k, :] = np.mean(X_train[:, 1:], 0)
        sigma[k, :] = np.std(X_train[:, 1:], 0)
        X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
        X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
        
        Xty = X_train.T @ y_train # X.T * y
        XtX = X_train.T @ X_train # X^2
        
        # Compute mean squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
        y_estimated_no_features.append(y_train.mean())
        mean_y = np.mean(y_train)
    
        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Compute mean squared error with regularization with optimal lambda
        Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
        
    
        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
        Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
        
        # yhat_baseline  = np.ones((y_test.shape[0],1))*(np.mean(y_train)).squeeze()
        # Error_test_baseline[k] = np.square(y_test-yhat_baseline.T).sum(axis=1)/y_test.shape[0]
        
        k+=1

    # Display the results for the optimal cross-validation fold
    opt_index = np.argmin(Error_test_rlr)
    k = opt_index
    
    plt.figure(k, figsize=(19,13))
    plt.subplot(1,2,1)
    plt.semilogx(lambdas,Arr_mean_w_vs_lambda[k].T[:,1:],'.-') # Don't plot the bias term
    plt.xlabel('Regularization factor',fontsize=20)
    plt.ylabel('Mean Coefficient Values',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(['X','Y','month','day','FFMC','DMC','DC','ISI',
                      'temp','RH','wind','rain'],fontsize=17)
            
    plt.subplot(1,2,2)
    plt.title('Optimal lambda: 1e{0}'.format(np.log10(Arr_opt_lambda[k][0])),fontsize=17)
    plt.loglog(lambdas,Arr_train_err_vs_lambda[k].T,'b.-',lambdas,Arr_test_err_vs_lambda[k].T,'r.-')
    plt.xlabel('Regularization factor',fontsize=20)
    plt.ylabel('Squared error (crossvalidation)',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(['Train error','Validation error'],fontsize=17)
    plt.show()
    
    # print('Mean squared wihtout features:')
    # print('- Training error: {0}'.format(Error_train_nofeatures.mean()))
    # print('- Test error:     {0}'.format(Error_test_nofeatures.mean()))
    # print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
    # print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
    print('Linear regression without feature selection:')
    print('- Training error: {0}'.format(Error_train.mean()))
    print('- Test error:     {0}'.format(Error_test.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
    print('Regularized linear regression:')
    print('- Training error: {0}'.format(Error_train_rlr.mean()))
    print('- Test error:     {0}'.format(Error_test_rlr.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))
    
    Weights = pd.DataFrame(w_rlr)
    Weights['Features']=pd.Series(['Offset (Maybe Mean)','FFMC','DMC','DC','ISI',
                              'temp','RH','wind','rain'])
    Weights_opt = Weights[['Features',k]]
    print(Weights_opt)
    
    Weights_opt.drop(index = 0, inplace = True)
    ax = Weights_opt[k].plot(kind='bar', figsize=(15, 10), legend=False, fontsize=25)
    ax.set_xticklabels(['FFMC','DMC','DC','ISI','temp','RH','wind','rain'])
    plt.xticks(rotation=-30)
    plt.ylabel('Weights' , fontsize=25)
    
    # data = Da.Data()
    # # ========================================================================================================================================
    # # Separating the data in X and y format. Also applying K-fold cross validation
    # # ========================================================================================================================================
    # X = data.df_data
    # X = np.array(X)
    
    # y = data.df_original['area']
    # y = np.array(y)
    # yTrans = np.log(y+1)
    # y = yTrans    
      
    # # Add offset attribute
    # X = np.concatenate((np.ones((X.shape[0],1)),X),1)
    # attributeNames = ['Offset'] + data.df_attributes
    
    # N, M = X.shape
    # K = 10
    # random_seed = 42
    # CV = model_selection.KFold(K, shuffle=True, random_state=random_seed)
    # # CV = model_selection.KFold(K, shuffle=False)
    
    # # Values of lambda
    # lambdas = np.power(10.,range(-2,10))
    
    # # Initialize variables
    # #T = len(lambdas)
    # Error_train = np.empty((K,1))
    # Error_test = np.empty((K,1))
    # Error_train_rlr = np.empty((K,1))
    # Error_test_rlr = np.empty((K,1))
    # Error_train_nofeatures = np.empty((K,1))
    # Error_test_nofeatures = np.empty((K,1))
    # Error_test_baseline = np.empty((K,1))
    # w_rlr = np.empty((M,K))
    # mu = np.empty((K, M-1))
    # sigma = np.empty((K, M-1))
    # w_noreg = np.empty((M,K))
    
    # k=0
    # y_estimated_no_features = []
    # for train_index, test_index in CV.split(X,y):
    
    #     # extract training and test set for current CV fold
    #     X_train = X[train_index]
    #     y_train = y[train_index]
    #     X_test = X[test_index]
    #     y_test = y[test_index] 
    #     internal_cross_validation = 10
    #     #  RETURNED VALUE FROM rlr_validate
    #     # opt_val_err = np.min(np.mean(test_error,axis=0))
    #     '''The optimal lambda is choosen as the one that give the lowest mean test error'''
    #     # opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]         
    #     # train_err_vs_lambda = np.mean(train_error,axis=0)
    #     # test_err_vs_lambda = np.mean(test_error,axis=0)
    #     # mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    #     opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    #     print(opt_lambda)
        
    #     # Standardize outer fold based on training set, and save the mean and standard
    #     # deviations since they're part of the model (they would be needed for
    #     # making new predictions) - for brevity we won't always store these in the scripts
    #     mu[k, :] = np.mean(X_train[:, 1:], 0)
    #     sigma[k, :] = np.std(X_train[:, 1:], 0)
    #     X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    #     X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
        
    #     Xty = X_train.T @ y_train # X.T * y
    #     XtX = X_train.T @ X_train # X^2
        
    #     # Compute mean squared error without using the input data at all
    #     Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    #     Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    #     y_estimated_no_features.append(y_train.mean())
    #     mean_y = np.mean(y_train)
    
    #     # Estimate weights for the optimal value of lambda, on entire training set
    #     lambdaI = opt_lambda * np.eye(M)
    #     lambdaI[0,0] = 0 # Do no regularize the bias term
    #     w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    #     # Compute mean squared error with regularization with optimal lambda
    #     Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    #     Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
        
    
    #     # Estimate weights for unregularized linear regression, on entire training set
    #     w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    #     Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    #     Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
        
    #     # yhat_baseline  = np.ones((y_test.shape[0],1))*(np.mean(y_train)).squeeze()
    #     # Error_test_baseline[k] = np.square(y_test-yhat_baseline.T).sum(axis=1)/y_test.shape[0]
        
    #     # Display the results for the last cross-validation fold
    #     if k == K-1:
    #         plt.figure(k, figsize=(19,13))
    #         plt.subplot(1,2,1)
    #         plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
    #         plt.xlabel('Regularization factor',fontsize=20)
    #         plt.ylabel('Mean Coefficient Values',fontsize=20)
    #         plt.xticks(fontsize=20)
    #         plt.yticks(fontsize=20)
    #         plt.legend(['X','Y','month','day','FFMC','DMC','DC','ISI',
    #                           'temp','RH','wind','rain'],fontsize=17)
                    
    #         plt.subplot(1,2,2)
    #         plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)),fontsize=17)
    #         plt.loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
    #         plt.xlabel('Regularization factor',fontsize=20)
    #         plt.ylabel('Squared error (crossvalidation)',fontsize=20)
    #         plt.xticks(fontsize=20)
    #         plt.yticks(fontsize=20)
    #         plt.legend(['Train error','Validation error'],fontsize=17)
            
    #     k+=1
    # plt.show()
    
    # # print('Mean squared wihtout features:')
    # # print('- Training error: {0}'.format(Error_train_nofeatures.mean()))
    # # print('- Test error:     {0}'.format(Error_test_nofeatures.mean()))
    # # print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
    # # print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
    # print('Linear regression without feature selection:')
    # print('- Training error: {0}'.format(Error_train.mean()))
    # print('- Test error:     {0}'.format(Error_test.mean()))
    # print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
    # print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
    # print('Regularized linear regression:')
    # print('- Training error: {0}'.format(Error_train_rlr.mean()))
    # print('- Test error:     {0}'.format(Error_test_rlr.mean()))
    # print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
    # print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))
    
    # Weights = pd.DataFrame(w_rlr)
    # Weights['Features']=pd.Series(['Offset (Maybe Mean)','X','Y','month','day','FFMC','DMC','DC','ISI',
    #                           'temp','RH','wind','rain'])
    # print(Weights)
    # Weights_last_fold = Weights[['Features',9]]
    # # print(Weights_last_fold)
    
    # Weights_last_fold.drop(index = 0, inplace = True)
    # ax = Weights_last_fold[9].plot(kind='bar', figsize=(15, 10), legend=False, fontsize=25)
    # ax.set_xticklabels(['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain'])
    # plt.xticks(rotation=-30)
    # plt.ylabel('Weights' , fontsize=25)

