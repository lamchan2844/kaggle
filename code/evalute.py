# -*- coding:utf-8 -*- #
import numpy as np


def FormatResult(act, pred):
	act = np.array(act)
	act = act.T
	pred = np.array(pred)
	pred = pred.T
	actual = []
	predicted = []
	for row in act:
		actual.append([i for i,p in enumerate(row) if p > 0.5])
	for row in pred:
		predicted.append([i for i,p in enumerate(row) if p > 0.5])
	return actual,predicted

def apk(actual, predicted, k = 7):
	#print actual
	#print predicted
	if len(predicted) >k:
		predicted = predicted[:k]

	score = 0.0
	num_hits = 0.0

	for i,p in enumerate(predicted):
		if p in actual and p not in predicted[:i]:
			num_hits += 1.0
			score += num_hits / (i + 1.0)

	if not actual:
		return 0.0

	#print 'label\'s ap:',score / min(len(actual), k)
	return score / min(len(actual), k)

def mapk(act, pred, k = 7):
	#transform the pred and actual to the ap form
	actual, predicted = FormatResult(act,pred) 
	#actual = act
	#predicted = pred
	return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

if __name__ == '__main__':
	act = [[1,2,3,5],[2,4,7],[5,3]]
	pred = [[1,2,4,6],[3,5,5],[6,3,2,5]]
	#print FormatResult(act,pred)
	print mapk(act,pred)