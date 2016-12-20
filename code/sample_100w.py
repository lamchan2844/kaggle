import os
import sys
import csv
import random
import numpy as np

def sample(length = 1000000):
    '''
    sample from '../data/train_ver2.csv'
    '''
    source_path='../data/train_ver2.csv'
    fp=open(source_path,'r')
    first_line = fp.readline()
    line_title=first_line.strip().split(',')

    title = []
    for var in line_title:
        title.append(var.strip('\"'))

    lines=[]

    #select length numbers from beg to end
    num_sample = length
    beg = 12647300
    end = 13647309
    rand_num = random.sample(range(beg,end),num_sample)
    rand_num.sort()
    #print rand_num
    count = 0;
    for i in range(1,13647309):
    	line=fp.readline()
    	if i == rand_num[count]:
    		lines.append(line)
    		count += 1
    		if count%(num_sample/1000) ==0:
    			print '%d%% completed'%((count+0.0)/num_sample*100.0)
    		if count == num_sample:
    			break
    random.shuffle(lines)
    target_file='../data/sample_train.csv'
    csvfile=file(target_file,'wb')
    writer=csv.writer(csvfile)
    writer.writerow(title)
    data=[]

    for i in range(0,len(rand_num)):
        line=lines[i]
        list_tmp=line.strip().split(',')
        if len(list_tmp)>48:
            list_tmp[20]=list_tmp[20].strip('\"')+list_tmp[21].strip('\"')
            for j in range(21,47):
                list_tmp[j]=list_tmp[j+1]
            list_tmp = list_tmp[:-1]
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
    print 'done!'

if __name__ == '__main__':
    sample(1000000)