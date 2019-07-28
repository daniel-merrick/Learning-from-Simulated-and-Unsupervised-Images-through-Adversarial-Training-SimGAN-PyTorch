import sys
sys.path.insert(0, 'lib/')


from train_estimator import TrainEstimator
from test_estimator import TestEstimator

import train_config
import test_config

if __name__ == '__main__':
	
	obj = TrainEstimator(train_config)	
	obj.train()
	
	# obj = TestEstimator(test_config)
	# obj.test()

