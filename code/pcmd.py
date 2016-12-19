import os
str_cmd = []
for num1 in range(40,60,2):
	str_cmd = 'python predict_rf.py'+' '+str(num1)+' '+str(2)
	print str_cmd
	os.system(str_cmd)