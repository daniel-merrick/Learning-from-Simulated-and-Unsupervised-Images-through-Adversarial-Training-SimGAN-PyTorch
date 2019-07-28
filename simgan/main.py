import sys

sys.path.insert(0, 'lib/')

from train_simgan import TrainSimGAN
import train_config

''' Still need to implement '''
from test_simgan import TestSimGAN
import test_config

if __name__ == '__main__':
	trainer = TrainSimGAN(train_config)
	trainer.train()

	#tester = TestSimGAN(test_config)
	#tester.refine()
