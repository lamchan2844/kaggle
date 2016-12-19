# -*- coding:utf-8 -*-#
import numpy as np
import csv
import os 
import xgboost as xgb
from utils import load_data
import pandas as pd
target_path = '../data/submission'
target_file = '../data/submission/submission.csv'
train_file = '../data/train_ver2.csv'
test_file = '../data/test_ver2.csv'
model_path = '../model'

def FormatSubmission(id_list,pred):
	"""
	transform the test result to the submission form

	"""
	#get title
	fp=open(train_file,'r')
	tmp_lines=fp.readline()
	result_tilte = tmp_lines.strip().split(',')[24:]
	fp.close()
	
	result = []
	pred = np.array(pred)
	pred = pred.T
	for row in pred:
		rst = ''
		count = 0
		row_new = row.tolist()
		while(count!=7):
			ind = row_new.index(max(row_new))
			rst += ' '+result_tilte[ind].strip('"')
			#rst += ' '+str(row_new[ind])			
			row_new[ind] = -1
			count += 1
		'''
		for i,p in enumerate(row):
			if p > 0.5:
				rst += ' '+result_tilte[i].strip('"')
		'''
		rst = rst.strip(' ')
		result.append(rst)
	result = np.concatenate([np.array(id_list).reshape(-1,1), np.array(result).reshape(-1,1)],axis = 1)
	return result

def predict_test():
	#fuse_list,id_list=load_data(flag = 0)
	fuse_list = pd.read_csv('../data/test_predictors.csv')
	id_list = pd.read_csv('../data/test_IDs.csv')
	print 'load completed'
	pred = []
	dtest=xgb.DMatrix(fuse_list)
	for n in range(24):
		bstn = xgb.Booster(model_file = model_path+'/001'+str(n)+'.model')
		predn=bstn.predict(dtest)
		pred.append(predn)
		print 'Predicted' ,str(n+1),'feature'
	print 'Submitting...'
	submit(id_list,pred)

def submit(id_list,pred):
	result = FormatSubmission(id_list,pred)
	if not os.path.exists(target_path):
		os.makedirs(target_path)
	with open(target_file,'wb') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter = ',')
		spamwriter.writerow(['ncodpers','added_products'])
		spamwriter.writerows(result)


if __name__ == '__main__':
	predict_test()
	if os.path.exists(target_file):
		print 'save done!'
	print 'Completed'
