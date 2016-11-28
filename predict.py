import numpy as np
import scipy as sp
import math
from sklearn.cross_validation import train_test_split
from read_train_data import load_train_data
from evalute import mapk
import os
import xgboost as xgb
import csv
from IPython import embed

train_file = '../data/train_ver2.csv'
train_sample_file = '../data/sample_train.csv'
test_file = '../data/test_ver2.csv'
model_path = '../model'
map_file = '../result_tmp/map.txt'

def trans2result(i):
    if i>0.5:
        return 1
    else:
        return 0


def error(list1,list2):
    len_list1=len(list1)
    len_list2=len(list2)
    if not len_list1==len_list2:
        print 'the length is wrong!'
        return 0
    sum=0.0
    for i in range(len_list1):
        if trans2result(list1[i]) != list2[i]:
            sum += 1
    result = sum/ len_list1
    return result

if __name__=='__main__':
    fuse_list,id_list,labels_list,len_digit=load_train_data(train_sample_file)
    label_array = np.array(labels_list)
    
    param={
        'eta':0.3,
        'max_depth':4,
        'gamma':1.0,
        'min_child_weight':1,
        'subsample':0.8,
        'save_period':0,
        'booster':'gbtree',
        #'silent':1,
        'nthread':4,
        'objective':'binary:logistic',
        #'eval_metric':'logloss'
    }

    err_list = []
    actual = []
    predicted = []
    for n in range(len(label_array[0])):
        x_train,x_test,y_train,y_test=train_test_split(fuse_list,label_array.T[n],test_size=0.2)
        xg_train=xgb.DMatrix(x_train,label=y_train)
        xg_test=xgb.DMatrix(x_test,label=y_test)
    
        watchlist=[(xg_train,'train'),(xg_test,'test')]
        num_round=100
        bst=xgb.train(param,xg_train,num_boost_round=num_round,evals=watchlist)
        bst.save_model(model_path+'/000'+str(n)+'.model')
        pred=bst.predict(xg_test)
        #print pred
        #print y_test
        print 'predict done! '
        actual.append(y_test)
        predicted.append(pred)

    map_value = mapk(actual,predicted)
    print 'map:',map_value
    fp = open(map_file,'ab+')
    fp.write('map:'+str(map_value)+'\n')
    fp.close()
    

