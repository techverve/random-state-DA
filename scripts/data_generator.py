'''------- EXTERNAL IMPORTS -------'''
import numpy as np
import matplotlib.pyplot as plt


PATH_TO_MODULE = '../data/'
sys.path.append(PATH_TO_MODULE)

def data_gen(num_samples):
	# The desired mean values of the sample ~ 4 attributes .
	mu = np.array([10.0, 10.0, 10.0, 50])

	# The desired covariance matrix.
	r = np.array([
		[  3.40, -2.75, -2.00, 3],
		[ -2.75,  5.50,  1.50, 1.75],
		[ -2.00,  1.50,  1.25, 1.60],
		[2.75, 3.00, 2.55, 2.75],
	    ])
	y = np.random.multivariate_normal(mu, r, size=num_samples)
	df = pd.DataFrame()
	df['col'] = y.tolist()
	df = pd.DataFrame([pd.Series(x) for x in df.col])
	df.columns = ['col_{}'.format(x+1) for x in df.columns]
	df.to_csv(f'../data/{num_samples}rows_dataset.csv', index = False)
	
	
if __name__ == '__main__':
	data_gen(100)
	data_gen(500)
	data_gen(1000)
	data_gen(10000)
	data_gen(100000)
