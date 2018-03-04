# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:43:54 2017

@author: Frank
"""
import os
os.chdir("F:/研究生/我爱数据处理/统计软件考试  2017年秋/")
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
def load_data(path):
    """加载数据并切分数据集
    @param path of data
    @return x,y train and test data
    """
    from sklearn.model_selection import train_test_split
    data = pd.read_csv(path)
    y = data['label']
    X = data.drop('label',axis = 1)
    x_train, x_test, y_train, y_test = \
                        train_test_split(X, y, random_state=1,test_size = 0.3)
    return x_train,x_test,y_train,y_test
def print_result(y_test,y_hat):
    acc_score = accuracy_score(y_test,y_hat)
    rec_score = recall_score(y_test,y_hat)
    pre_score = precision_score(y_test,y_hat)
    con_metrx = confusion_matrix(y_test,y_hat)
    print(con_metrx,acc_score,rec_score,pre_score, sep = ',')

def xgboost_model(*args):
    """利用xgboost建模并评估模型质量"""
    import xgboost as xgb
    data_train = xgb.DMatrix(args[0], label=args[2])
    data_test = xgb.DMatrix(args[1], label=args[3])
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}
    bst = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)
    y_hat = bst.predict(data_test)
    y = args[3].flatten()
#    print_result(y,y_hat)
    return y
    
def svm_model(*args):
    """利用svm建模并评估模型质量"""
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    c_can = np.logspace(-2, 2, 10)
    param = {'kernel':['rbf','sigmoid','linear'],'C':c_can}
    clf = svm.SVC()
    Grid_model = GridSearchCV(clf,param_grid=param,cv = 5)
    Grid_model.fit(args[0],args[2])
    y_hat = Grid_model.predict(args[1])
#    print_result(args[4], y_hat)
    return y_hat

def random_forest_model(*args):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=3,max_depth = 5)
    clf.fit(args[1],args[3]) 
    y_hat = clf.predict(args[2])
    print_result(args[4], y_hat)

if __name__ == "__main__":
    data_clf = np.concatenate([data_cat,num_filter_scale ],axis = 1)
   
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = \
                        train_test_split(data_clf, label, random_state=1,test_size = 0.3)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)
    data = [x_train,x_test,y_train,y_test]
    
    y = xgboost_model(*data)
    
    confusion_matrix(y,y_test)
    
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    
    clf = svm.SVC(kernel = 'linear',C=1)
    
    from sklearn.preprocessing import Imputer
    imp = Imputer(strategy='most_frequent',axis = 0)
    #离散化
    x_train = imp.fit_transform(x_train)
    x_test = imp.fit_transform(x_test)
    y_train = imp.fit_transform(y_train)
    clf.fit(x_train,y_train)
    
    y_svm = clf.predict(x_test)
    
    confusion_matrix(y_test,y_svm)
#    print_result(args[4], y_hat)
    return y_hat
    
    y_svm = svm_model(*data)
    
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=3,max_depth = 5)
    clf.fit(x_train,y_train) 
    y_rf = clf.predict(x_test)
    confusion_matrix(y_test,y_rf)