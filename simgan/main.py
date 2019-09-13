import sys

sys.path.insert(0, 'lib/')

# Import class and config file for training 
from train_simgan import TrainSimGAN
import train_config

# Import class and config file for testing
from test_simgan import TestSimGAN
import test_config

if __name__ == '__main__':
	# If you want to train, uncomment the following two line
	#trainer = TrainSimGAN(train_config)
	#trainer.train()
	
	# If you want to test, uncomment the following two lines
	#tester = TestSimGAN(test_config)
	#tester.refine()
