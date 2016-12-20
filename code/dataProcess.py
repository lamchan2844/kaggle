import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from utils import get_train_title

train_file = '../data/sample_train.csv'
test_file = '../data/test_ver2.csv'

def dataProcess():
	'''
	processing the raw data ,reduce the length of code
	'''
	##load data:
	
	train = pd.read_csv(train_file)
	test = pd.read_csv(test_file)
	print 'dataLoadCompleted!'

	## get the title
	title = get_train_title()
	targets = title[24:]
	features_object = ['ind_empleado','pais_residencia','sexo',
	'ind_nuevo','indrel','indrel_1mes','tiprel_1mes',
	'indresi','indext','conyuemp','canal_entrada',
	'indfall','tipodom','cod_prov','ind_actividad_cliente',
	'segmento']

	## Combine into data:
	train['source'] = 'train'
	test['source'] = 'test'
	data = pd.concat([train,test],ignore_index = True)

	data.drop('nomprov',axis = 1,inplace = True)
	data.drop(['fecha_alta','fecha_dato','ult_fec_cli_1t'],axis = 1,inplace = True)

	#indrel_1mes
	data['indrel_1mes'].fillna('NAN',inplace = True)
	indrel_1mes_features_pnan = ['P','NAN']
	data['indrel_1mes'] = data['indrel_1mes'].apply(lambda x: int(float(x)) if x not in indrel_1mes_features_pnan else x)

	#pais_residencia
	data['pais_residencia'] = data['pais_residencia'].apply(lambda x: 'others' if x!= 'ES' else x)

	## renta
	renta_median = train['renta'].median()
	data['renta'] = data['renta'].apply(lambda x: renta_median if str(x).strip()== 'NA' or pd.isnull(x) else float(x))

	## sexo
	sexo_majority = ['V','H']
	data['sexo'].fillna(sexo_majority[random.randint(0,1)],inplace = True)
	#tiprel_1mes
	tiprel_1mes_majority = ['I','A']
	data['tiprel_1mes'].fillna(tiprel_1mes_majority[random.randint(0,1)],inplace = True)

	## canal_entrada missing 10232
	cannal_entrada_primary = ['KHE','KAT','KFC','KHQ','KHM','KFA','KHN','KHK','KHD','RED','KAS']
	data['canal_entrada'] = data['canal_entrada'].apply(lambda x : 'others' if x not in cannal_entrada_primary else x)

	#data['cod_prov'].fillna(28,inplace = True)
	data.drop('cod_prov',axis = 1,inplace = True)
	features_object.remove('cod_prov')

	## conyuemp
	data['conyuemp_missing'] = data['conyuemp'].apply(lambda x:1 if pd.isnull(x) else 0)
	#print data[['conyuemp','conyuemp_missing']].head(10)
	data.drop('conyuemp',axis = 1, inplace = True)
	features_object.remove('conyuemp')
	features_object.append('conyuemp_missing')

	data['segmento'].fillna('others',inplace = True)

	'''
	for v in features_object:
	    print '\nFrequency count for variable %s'%v
	    print data[v].value_counts()
	'''
	#print data.apply(lambda x:sum(x.isnull()))

	## Numerical Coding
	le = LabelEncoder()
	for col in features_object:
	    data[col] = le.fit_transform(data[col])

	## One-Hot Coding
	data = pd.get_dummies(data, columns=features_object)

	## Separate train & test:
	train = data.loc[data['source']=='train']
	test = data.loc[data['source']=='test']

	train.drop('source',axis=1,inplace=True)
	test.drop('source',axis=1,inplace=True)
	test.drop(targets,axis = 1,inplace = True)

	IDcol = 'ncodpers'
	predictors = [x for x in train.columns if x not in targets and x not in IDcol]
	#print predictors
	#print train[targets].values

	train[predictors].to_csv('../data/train_predictors.csv',index = False)
	train[targets].to_csv('../data/train_targets.csv',index = False)
	test[predictors].to_csv('../data/test_predictors.csv',index = False)
	open('../data/test_IDs.csv','w').write(IDcol+'\n')
	test[IDcol].to_csv('../data/test_IDs.csv',mode = 'ab+',index = False)

	print 'save done!'

if __name__ == '__main__':
	dataProcess()
