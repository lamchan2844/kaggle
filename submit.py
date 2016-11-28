# -*- coding:utf-8 -*-#
import numpy as np
import csv
import os 
import xgboost as xgb
target_path = '../data/submission'
target_file = '../data/submission/submission.csv'
train_file = '../data/train_ver2.csv'
test_file = '../data/test_ver2.csv'
model_filename = '../model/0001.model'

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
		for i,p in enumerate(row):
			if p > 0.5:
				rst += ' '+result_tilte[i].strip('"')
		rst = rst.strip(' ')
		result.append(rst)
	result = np.concatenate([np.array(id_list).reshape(-1,1), np.array(result).reshape(-1,1)],axis = 1)
	return result

def predict_test(filename = test_file,model = model_filename):
	bst = xgb.Booster(model_file = model_filename)
	#dtest = xgb.DMatrix(xxxx)
	#pred = bst.predict(dtest)
	#submit(id_list,pred)

def submit(id_list,pred):
	result = FormatSubmission(id,pred)
	if not os.path.exists(target_path):
		os.makedirs(target_path)
	with open(target_file,'wb') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter = ',')
		spamwriter.writerows(result)


if __name__ == '__main__':
	id = [1,2,3,4]
	pred = [[0,1,0,1],[0,0,1,1]]
	submit(id,pred)
	if os.path.exists(target_file):
		print 'save done!'
