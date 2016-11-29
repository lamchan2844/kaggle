# -*- coding:utf-8 -*- #
import csv
import numpy as np
from sklearn import preprocessing

train_file = '../data/train_ver2.csv'
train_sample_file = '../data/sample_train.csv'
test_file = '../data/test_ver2.csv'

number_col = [5,8,22] #values in these cols are number  
discard_col = [0,1,6,10,20] #discard these cols

def isnumber(n):
	if n in number_col:
		return True
	else:
		return False;

def discard(n):
	if (n in discard_col) or (n>23):
		return True
	else:
		return False

def string2int_list(mlist):
	for i in range(len(mlist)):
		try:
			mlist[i] = int(mlist[i])
		except:
			mlist[i] = 0
	return mlist

def data_preprocessing(file = train_sample_file, flag = 1):
	'''
	flag = 1 -- train data (default)
	flag = 0 -- test data

	'''
	digit_list=[] #continuous num
	alpha_list=[] #discontinuous num or string,to be onehotencoded
	id_list=[]	# the customer's id
	labels_list=[]	#the label only exist in train data
	with open(file,'rb') as csvfile:
		spamreader = csv.reader(csvfile,delimiter=',')
		for i,row in enumerate(spamreader):
			if i == 0:
				firstline = row
				continue
			if i == 100001:
				break
			row = '|'.join(row)
			line_list = row.strip().split('|')
			id_list.append(int(line_list[1]))
			
			## transform some features to num  or discard the row
			if line_list[19].strip() == 'NA':# province code transform NA to -1
				line_list[19] = '-999999'
			if (line_list[22].strip() == 'NA') or line_list[22].strip() == '': #renta
				line_list[22] = '-1'
			if line_list[5].strip() == 'NA': #age if no age ,discard this row
				continue
			if line_list[8].strip() == 'NA': #age if no age ,discard this row
				line_list[8] = '-1'
			if flag == 1:
				labels_list.append(string2int_list(line_list[24:]))
			line_digit=[float(digit.strip()) for n,digit in enumerate(line_list) if (isnumber(n)==True) and discard(n) == False ] #digit
			line_alpha=[st for n,st in enumerate(line_list) if isnumber(n)==False and discard(n) == False] # string
			#len_gap=len(line_digit) #length of digit line (may not be used,can be ignored)
			digit_list.append(line_digit)
			alpha_list.append(line_alpha)
	alpha_arr = np.array(alpha_list)

	if flag == 1:
		return id_list,digit_list,alpha_arr,labels_list
	else:
		return id_list,digit_list,alpha_arr


def OneHot(alpha_arr,alpha):
	len_alpha_feature = len(alpha_arr[0])
	for n in range(len_alpha_feature):
		#trans discontinuous value to a number 
		le = preprocessing.LabelEncoder()
		le.fit(alpha_arr.T[n,:])
		arrle = le.transform(alpha_arr.T[n,:])
		arrle_alpha = le.transform(alpha.T[n,:])
		if n == 0:
			alpha_arr_pro = arrle.reshape(1,-1)
			alpha_pro = arrle_alpha.reshape(1,-1)
		else:
			alpha_arr_pro = np.concatenate((alpha_arr_pro,arrle.reshape(1,-1)),axis = 0)
			alpha_pro = np.concatenate((alpha_pro,arrle_alpha.reshape(1,-1)),axis = 0)

	# onehotencoding
	enc = preprocessing.OneHotEncoder()
	enc.fit(alpha_arr_pro.T)
	trans_list = enc.transform(alpha_pro.T).toarray()
	return trans_list

def load_data(file = train_sample_file, flag = 1):
	'''
		file - train data
		
		flag = 1 -- train data (default)
		flag = 0 -- test data
	'''
	id_list_train,digit_list_train,alpha_arr_train,labels_list_train = data_preprocessing(file,1)
	id_list_test,digit_list_test,alpha_arr_test = data_preprocessing(test_file,0)
	alpha_arr = np.concatenate((alpha_arr_train,alpha_arr_test),axis = 0)
	if flag == 1:
		trans_list = OneHot(alpha_arr,alpha_arr_train)
		digit_list = digit_list_train
	else:
		trans_list = OneHot(alpha_arr,alpha_arr_test)
		digit_list = digit_list_test

	trans_list=np.array(trans_list,dtype=int)
	digit_list=np.array(digit_list,dtype=int)
	fuse_list=np.concatenate([digit_list,trans_list],axis=1)

	print 'No. of sample: ',len(trans_list)
	print 'length of encoding: ',len(trans_list[0])
	if flag == 1:
		return fuse_list,id_list_train,labels_list_train
	if flag == 0:
		return fuse_list,id_list_test

if __name__ == '__main__':
	load_data(flag = 1)
