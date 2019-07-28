"""
	This file only contains the Gaze Estimator Network Class 
	
	It contains network architecture, forward function,
	and a function to set gradient flag for training.
"""


from torch import nn

class GazeEstimatorNetwork(nn.Module):
	'''
	Estimator --- Network used for gaze prediction
	
		Notes*
			1. Input is a batch of grayscale images of shape == (1, 35, 55)
			2. Outputs a batch of 3D tensors (predicted gazes)
	'''
	def __init__(self):
		super(GazeEstimatorNetwork, self).__init__()
		
		# self.conv is a convolutional block,
		# it outputs a shape of (batch_size, 192, 10, 15)
		self.conv = nn.Sequential(
			nn.Conv2d(1, 32, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(32, 64, 3, 1, 1),
			nn.LeakyReLU(),
			nn.MaxPool2d(3, 2, 1),
			nn.Conv2d(64, 80, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(80, 192, 3, 1, 1),
			nn.LeakyReLU(),
			nn.MaxPool2d(2, 2, 1),
		)
		
		# self.fc is a fully connected block
		# it outputs a 3-dim tensor (the predicted gaze vector)
		self.fc = nn.Sequential(
			nn.Linear(192 * 10 * 15, 9600),
			nn.ReLU(),
			nn.Linear(9600, 1000),
			nn.ReLU(),
			nn.Linear(1000, 3)
		)	
	
	# this function is called externally 
	# when we want to set the gradients 
	# on the network params for training
	def train_mode(self, flag=True):
		for p in self.parameters():
			p.requires_grad=flag
	
	# this function is called for forward prop
	def forward(self, input_images):
		conv_output = self.conv(input_images)

		# flatten the output of the conv block so we
		# can feed it into the fully connected layers
		fc_input = conv_output.view(-1, 192 * 10 * 15)
		gaze_vector = self.fc(fc_input)
		return gaze_vector

