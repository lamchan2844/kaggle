#-*- coding:utf-8 -*-
import os
import sys
import csv
source_path='../data/train_ver2.csv'
fp=open(source_path,'r')
line_title=fp.readline().strip().split(',')


lines=[]
for i in range(10000000,20000000):
    if i%100==0:
        line=fp.readline()
        lines.append(line)
        # line_list=line.strip().split(',')
        # print line_list
        # list_train.append(line_list)
target_file='../data/sample_train.csv'
csvfile=file(target_file,'wb')
writer=csv.writer(csvfile)
writer.writerow(line_title)
data=[]

for i in range(100000):
    line=lines[i]
    list_tmp=line.strip().split(',')
    if len(list_tmp)>48:
        list_tmp[20]=list_tmp[20].strip('\"')+list_tmp[21].strip('\"')
        for j in range(21,47):
            list_tmp[j]=list_tmp[j+1]
    init_tuple=(list_tmp[0].strip(' '),)
    for j in range(len(list_tmp)):
        if j==0:
            continue
        else:
            init_tuple+=(list_tmp[j].strip(' \"'),)
    array_tmp=init_tuple
    data.append(array_tmp)
writer.writerows(data)
csvfile.close()
