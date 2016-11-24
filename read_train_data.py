import csv
import numpy as np
from sklearn import preprocessing
from time import clock


train_file = '../data/train_ver2.csv'
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

digit_list=[]
alpha_list=[]
id_list=[]
label_list=[]
with open(train_file,'rb') as csvfile:
	start = clock()
	spamreader = csv.reader(csvfile,delimiter=',')
	for i,row in enumerate(spamreader):
		if i == 0:
			firstline = row
			continue
		if i == 100000:
			break
		row = '|'.join(row)
		line_list = row.strip().split('|')
		id_list.append(int(line_list[1]))
		#print line_list[20]
		#line_list=line_list[2:6]+line_list[7:10]+line_list[11:20]+line_list[21:24]
		#print line_list
		## transform some features to num  or discard the row
		if line_list[19].strip() == 'NA':# province code transform NA to -1
			line_list[19] = '-999999'
		if (line_list[22].strip() == 'NA') or line_list[22].strip() == '': #renta
			line_list[22] = '-1'
		if line_list[5].strip() == 'NA': #age if no age ,discard this row
			continue
		if line_list[8].strip() == 'NA': #age if no age ,discard this row
			line_list[8] = '-1'
		line_digit=[float(digit.strip()) for n,digit in enumerate(line_list) if (isnumber(n)==True) and discard(n) == False ] #digit
		line_alpha=[st for n,st in enumerate(line_list) if isnumber(n)==False and discard(n) == False] # string
		len_gap=len(line_digit) #length of digit line
		digit_list.append(line_digit)
		alpha_list.append(line_alpha)
alpha_arr = np.array(alpha_list)

len_alpha_feature = len(alpha_arr[0])
for n in range(len_alpha_feature):
	le = preprocessing.LabelEncoder()
	le.fit(alpha_arr.T[n,:])
	arrle = le.transform(alpha_arr.T[n,:])
	if n == 0:
		alpha_arr_pro = arrle.reshape(1,-1)
	else:
		alpha_arr_pro = np.concatenate((alpha_arr_pro,arrle.reshape(1,-1)),axis = 0)

num_code = np.concatenate([digit_list, alpha_arr_pro.T],axis = 1)

feature_name = []
#change the position of features' name to fit the data
for col in number_col:
	feature_name.append(firstline[col])
for col in range(24):
	if col not in number_col and col not in discard_col:
		feature_name.append(firstline[col])
with open('../data/train_processed.csv','wb') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter = ',')
	spamwriter.writerow(feature_name)
	spamwriter.writerows(num_code)	


enc = preprocessing.OneHotEncoder()
enc.fit(alpha_arr_pro.T)
trans_list = enc.transform(alpha_arr_pro.T).toarray()


trans_list=np.array(trans_list,dtype=int)
digit_list=np.array(digit_list,dtype=int)
fuse_list=np.concatenate([digit_list,trans_list],axis=1)

#print fuse_list
print 'No. of sample: ',len(trans_list)
print 'length of encoding: ',len(trans_list[0])
#return fuse_list,id_list,len_gap # no label_list


finish = clock()
print 'time: ',(finish - start),' s'