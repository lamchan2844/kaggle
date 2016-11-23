import csv
import numpy as np
from sklearn import preprocessing

def isnumber(num):
	try:
		float(num)
		return True
	except:
		return False

train_file = '../data/train_ver2.csv'
test_file = '../data/test_ver2.csv'

digit_list=[]
alpha_list=[]
id_list=[]
label_list=[]
with open(train_file,'rb') as csvfile:
	spamreader = csv.reader(csvfile,delimiter=',')
	for i,row in enumerate(spamreader):
		if i <= 0:
			continue
		if i == 10000:
			break
		row = '|'.join(row)
		line_list = row.strip().split('|')
		id_list.append(int(line_list[1]))
		#print line_list[20]
		line_list=line_list[2:6]+line_list[7:10]+line_list[11:20]+line_list[21:24]
		#print line_list
		## transform some features to num  or discard the row
		if line_list[15].strip() == 'NA':# province code transform NA to -1
			line_list[15] = '-1'
		if line_list[17].strip() == '' or 'NA': #renta
			line_list[17] = '-1'
		if line_list[3].strip() == 'NA': #age if no age ,discard this row
			continue
		line_digit=[float(digit) for digit in line_list if isnumber(digit.strip())==True] #digit
		line_alpha=[st for st in line_list if isnumber(st.strip())==False] # string
		len_gap=len(line_digit) #length of digit line
		digit_list.append(line_digit)
		alpha_list.append(line_alpha)

testarr = np.array(alpha_list)
#print testarr
#print (testarr.T)

len_alpha_feature = len(testarr[0])
for n in range(len_alpha_feature):
	le = preprocessing.LabelEncoder()
	le.fit(testarr.T[n,:])
	arrle = le.transform(testarr.T[n,:])
	#print arrle
	enc = preprocessing.OneHotEncoder()
	enc.fit(arrle.reshape(-1,1))
	arrone = enc.transform(arrle.reshape(-1,1))
	print 'feature ',n
	print arrone.toarray()
	
#print id_list

