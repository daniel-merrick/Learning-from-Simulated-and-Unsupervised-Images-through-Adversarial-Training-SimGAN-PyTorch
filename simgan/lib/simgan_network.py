'''
	This file hold the network architecture for SimGAN including the 
	Refiner and Discriminator
'''

from torch import nn

class ResnetBlock(nn.Module):
	# Resnet block used for building the Refiner
	# Implements a skip connection
	# 64 output filters with 3x3 convolutions

	def __init__(self, input_features, nb_features=64, filter_size=3):
		super(ResnetBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(input_features, nb_features, filter_size, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(nb_features, nb_features, filter_size, 1, 1)
		)
		self.relu = nn.LeakyReLU()

	def forward(self, x):
		convs = self.conv(x)
		sum_ = convs + x
		output = self.relu(sum_)
		return output

class Refiner(nn.Module):
	'''
	Refiner --- class used to refine inputed synthetic data

		Notes*
			1. input should be a batch of synthetic, grayscale images with shape equal to [1, 35, 55]

			2. Output would be a batch of refined imaged (more realistic)

	'''

	def __init__(self, block_num=4, nb_features=64):
		super(Refiner, self).__init__()
		
		''' Image size is [1, 35, 55] '''
		self.conv_1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=nb_features, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU()
		)

		blocks = []
		for i in range(block_num):
			blocks.append(ResnetBlock(nb_features, nb_features, filter_size=3))
			
				
		self.resnet_blocks = nn.Sequential(*blocks)
		self.conv_2 = nn.Sequential(
			nn.Conv2d(nb_features, 1, kernel_size=1, stride=1, padding=0),
			nn.Tanh()
		)

	# used externally
	# used to set the gradient flags
	def train_mode(self, flag=True):
		for p in self.parameters():
			p.requires_grad = flag

	# used externally 
	# used for forward prop
	def forward(self, x):
		conv_1 = self.conv_1(x)
		res_block = self.resnet_blocks(conv_1)
		output = self.conv_2(res_block)
		return output


class Discriminator(nn.Module):
	''' 
	Discriminator --- class used to discriminate between refined and real data
		
		Notes*
			1. Input is a set of refined or real grayscale images of shape == [1, 35, 55]
			2. Output is a 2D conv map which is a map of probabilities between refined or real
	'''

	def __init__(self):
		super(Discriminator, self).__init__()
		
		''' Image size is [1, 55, 35] '''
		self.convs = nn.Sequential(
			nn.Conv2d(1, 96, 3, 2, 1),
			nn.LeakyReLU(),
			nn.Conv2d(96, 64, 3, 2, 1),
			nn.LeakyReLU(),
			nn.MaxPool2d(3, 1, 1),
			nn.Conv2d(64, 32, 3, 1, 1),
			nn.LeakyReLU(),
			nn.Conv2d(32, 32, 1, 1, 0),
			nn.LeakyReLU(),
			nn.Conv2d(32, 2, 1, 1, 0)
		)
		
	# used externally
	# used to set the gradient flags
	def train_mode(self, flag=True):
		for p in self.parameters():
			p.requires_grad = flag

	# used externally
	# used to set the gradient flags
	def forward(self, x):
		convs = self.convs(x)
		output = convs.permute(0, 2, 3, 1).contiguous().view(-1, 2)
		return output

