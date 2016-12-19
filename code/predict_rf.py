import os,sys
import numpy as np
from utils import load_data
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from evalute import mapk
import math
import pandas as pd
from predict import showConfusionMatrix_ALL
'''
data file definition
'''
train_file='../data/train_ver2.csv'
train_sample_file='../data/sample_trein.csv'
test_file='../data/test_ver2.csv'
map_file='../result_tmp/map.txt'

fp=open('../data/train_ver2.csv','r')
first_line = fp.readline()
line_title=first_line.strip().split(',')
title = []
for var in line_title:
    title.append(var.strip('\"'))
fp.close()
targets = title[24:]

if __name__=='__main__':
    #fuse_list,id_list,labels_list=load_data(flag=1)
    #label_array=np.array(labels_list)
    fuse_list = pd.read_csv('../data/train_predictors.csv')
    labels_list = pd.read_csv('../data/train_targets.csv')
    len_label=labels_list.shape[1]
    max_depth=int(sys.argv[1])
    random_state=int(sys.argv[2])
    '''
    define several randomfoersts for all labels
    '''

    #global rf_cls = []
    #rf_cls = np.zeros(len_label)
    pred = np.zeros([400000,len_label])
    actual = []
    predicted = []
    for i in range(len_label):
        print i
        rf_cls=RandomForestClassifier(n_estimators=50,n_jobs=4,max_depth=max_depth,random_state=random_state)
    #pred=np.zeros(labels_list.shape)        
        x_train,x_test,y_train,y_test=train_test_split(fuse_list,labels_list[targets[i]],test_size=0.4)
        rf_cls.fit(x_train,y_train)
        #print x_test.shape
        #print pred.shape
        pred[:,i]=rf_cls.predict(x_test)
        actual.append(y_test)
        predicted.append(pred[:,i])
        #print predicted
    #print actual
    #print predicted
    showConfusionMatrix_ALL(actual,predicted)
    '''
    for i in range(len_label):
        rf_cls[i]=RandomForestClassifier(n_estimators=50,n_jobs=4,max_depth=max_depth,random_state=random_state)
    pred=np.zeros(labels_list.shape)
    for i in range(len_label):
        print i
        x_train,x_test,y_train,y_test=train_test_split(fuse_list,labels_list[targets[n]],test_size=0.4)
        rf_cls[i].fit(x_train,y_train)
        pred[:,i]=rf_cls[i].predict(x_test)
    '''

    '''
    get the map@7
    '''
    map_value=mapk(actual,predicted)
    print map_value
    with open(map_file,'a') as fw:
        str_tmp='with max_depth= '+str(max_depth)+' and random_state= '+str(random_state)+': '
        str_tmp+='map@7 is : '+str(map_value)+'\n'
        fw.write(str_tmp)
