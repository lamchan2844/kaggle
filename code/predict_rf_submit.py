import os,sys
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from evalute import mapk
import math
import pandas as pd
from predict import showConfusionMatrix_ALL
from submit import WriteSubmission
from utils import get_train_title
'''
data file definition
'''
train_file='../data/train_ver2.csv'
train_sample_file='../data/sample_trein.csv'
test_file='../data/test_ver2.csv'
map_file='../result_tmp/map.txt'

title = get_train_title()
targets = title[24:]

def predict_submit_rf():
    #fuse_list,id_list,labels_list=load_data(flag=1)
    #label_array=np.array(labels_list)
    fuse_list = pd.read_csv('../data/train_predictors.csv')
    labels_list = pd.read_csv('../data/train_targets.csv')
    test_list = pd.read_csv('../data/test_predictors.csv')
    id_test_list = pd.read_csv('../data/test_IDs.csv')
    pred_test = []

    len_label=labels_list.shape[1]
    max_depth=48
    random_state=2
    '''
    define several randomfoersts for all labels
    '''

    #global rf_cls = []
    #rf_cls = np.zeros(len_label)
    pred = np.zeros([400000,len_label])
    actual = []
    predicted = []
    x_train,x_test,ys_train,ys_test=train_test_split(fuse_list,labels_list,test_size=0.4)
    for i in range(len_label):
        print 'training the label %d '%i
        rf_cls=RandomForestClassifier(n_estimators=50,n_jobs=4,max_depth=max_depth,random_state=random_state)
    #pred=np.zeros(labels_list.shape)        
        y_train = ys_train[targets[i]]
        y_test = ys_test[targets[i]]
        rf_cls.fit(x_train,y_train)
        #print x_test.shape
        #print pred.shape
        pred[:,i]=rf_cls.predict(x_test)
        actual.append(y_test)
        predicted.append(pred[:,i])
        pred_tmp = rf_cls.predict(test_list)
        pred_test.append(pred_tmp)
    with open('../data/submission/submission_prob_rf.csv','wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter = ',')
        spamwriter.writerows(np.array(predicted).T)

    '''
    get the map@7
    '''
    map_value=mapk(actual,predicted)
    print map_value
    if not os.path.exists('../result_tmp'):
        os.makedirs('../result_tmp')
    with open(map_file,'a') as fw:
        str_tmp='with max_depth= '+str(max_depth)+' and random_state= '+str(random_state)+': '
        str_tmp+='map@7 is : '+str(map_value)+'\n'
        fw.write(str_tmp)
    WriteSubmission(id_test_list,pred_test,'submission_rf')
if __name__=='__main__':
    predict_submit_rf()