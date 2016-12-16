import numpy as np
import scipy as sp
import pandas as pd
import math
from sklearn.cross_validation import train_test_split
from utils import load_data
from evalute import mapk
import os
import xgboost as xgb
import csv
from submit import predict_test

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


def showConfusionMatrix(acut,pred):
	a11 = 0
	a12 = 0
	a21 = 0
	a22 = 0
	len_acut = len(acut)
	for i in range(len_acut):
		pred_int = trans2result(pred[i])
		if acut[i] == 0 and pred_int == 0:
			a22 += 1
		elif acut[i] == 1 and pred_int == 1:
			a11 += 1
		elif acut[i] == 1 and pred_int == 0:
			a12 += 1
		elif acut[i] == 0 and pred_int == 1:
			a21 += 1
	print '   |   1  |	  0'
	print '------------------'
	print '1  |  ',a11,' |  ',a12
	print '------------------'
	print '0  |  ',a21,' |  ',a22


def showConfusionMatrix_ALL(acut,pred):
	print len(acut)
	for i in range(len(acut)):
		print 'label ',i
		showConfusionMatrix(acut[i].values,pred[i])
		#showConfusionMatrix(acut[i],pred[i])
		print '\n'


## get the title
fp=open('../data/train_ver2.csv','r')
first_line = fp.readline()
line_title=first_line.strip().split(',')
title = []
for var in line_title:
    title.append(var.strip('\"'))
fp.close()
targets = title[24:]

if __name__=='__main__':
    fuse_list = pd.read_csv('../data/train_predictors.csv')
    labels_list = pd.read_csv('../data/train_targets.csv')
    #fuse_list,id_list,labels_list=load_data(flag = 1)
    #label_array = np.array(labels_list)
    
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
    for n in range(labels_list.shape[1]):
    	#x_train,x_test,y_train,y_test=train_test_split(fuse_list,label_array.T[n],test_size=0.2)
        x_train,x_test,y_train,y_test=train_test_split(fuse_list,labels_list[targets[n]],test_size=0.2)
        xg_train=xgb.DMatrix(x_train,label=y_train)
        xg_test=xgb.DMatrix(x_test,label=y_test)
    
        watchlist=[(xg_train,'train'),(xg_test,'test')]
        num_round=100
        bst=xgb.train(param,xg_train,num_boost_round=num_round,evals=watchlist)
        bst.save_model(model_path+'/002'+str(n)+'.model')
        pred=bst.predict(xg_test)
        #print pred
        #print y_test
        print 'predict done! '
        actual.append(y_test)
        predicted.append(pred)
        #showConfusionMatrix(y_test,pred)
    
    
    showConfusionMatrix_ALL(actual,predicted)
    with open('../data/actual_20w.csv','wb') as csvfile:
    	spamwriter = csv.writer(csvfile, delimiter = ',')
    	spamwriter.writerows(np.array(actual).T)
	with open('../data/predicted_20w.csv','wb') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter = ',')
		spamwriter.writerows(np.array(predicted).T)

    map_value = mapk(actual,predicted)
    print 'map:',map_value
    fp = open(map_file,'ab+')
    fp.write('map:'+str(map_value)+'\n')
    fp.close()
    

