import dataProcess as dp
from submit import submit
from sample_100w import sample
from predict import predict
from predict_rf-submit import predict_submit_rf

def run_ALL():
	print 'ready to sample...'
	sample(1000000)
	print 'sample done!\n ready to processing data'
	dp.dataProcess()
	print 'processing data done!\nready to train data'
	predict()
	print 'train done'
	submit('submission_2')
	print 'All done'
def run_rf():
	predict_submit_rf()
	
if __name__ == '__main__':
	run_ALL()