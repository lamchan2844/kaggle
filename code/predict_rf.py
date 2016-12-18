import os,sys
import numpy as np
from utils import load_data
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from evalute import mapk
import math

'''
data file definition
'''
train_file='../data/train_ver2.csv'
train_sample_file='../data/sample_trein.csv'
test_file='../data/test_ver2.csv'
map_file='../result_tmp/map.txt'



if __name__=='__main__':
    fuse_list,id_list,labels_list=load_data(flag=1)
    label_array=np.array(labels_list)
    len_label=label_array.shape[1]
    max_depth=int(sys.argv[1])
    random_state=int(sys.argv[2])
    '''
    define several randomfoersts for all labels
    '''

    global rf_cls
    for i in range(len_label):
        rf_cls[i]=RandomForestClassifier(n_estimators=50,n_jobs=4,max_depth=max_depth,random_state=random_state)
    pred=np.zeros(label_array.shape)
    for i in range(len_label):
        x_train,x_test,y_train,y_test=train_test_split(fuse_list,label_array[:,i],test_size=0.4)
        rf_cls[i].fit(x_train,y_train)
        pred[:,i]=rf_cls[i].predict(x_test)

    '''
    get the map@7
    '''
    map_value=mapk(label_array,pred)
    print map_value
    with open(map_file,'a') as fw:
        str_tmp='with max_depth= '+str(max_depth)+' and random_state= '+str(random_state)+': '
        str_tmp+='map@7 is : '+str(map_value)+'\n'
        fw.write(str_tmp)
