from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch

class DataLoader(Dataset):
	'''
		DataLoader: class to load the data. Passed into torch.utils.data.dataloader

			Notes*
				1. Input is the path to the file with the image files and gaze vectors

					For Example:
							file 'path' should be formatted like the following:
							imgFilePath1,x1,y1,z1,0
							imgFilePath2,x2,y2,z2,0
							...
							...
							imgFilePathN,xN,yN,zN,0

					where x,y,z are the gaze vector values
		
				2. This class is overriding the torch.utils.data.dataset.Dataset class
				3. This class is passed to torch.utils.data.dataloader
	'''
	
	# used internally
	# load image files, get # of images, load transforms
	def __init__(self, path, mpii=False):
		
		self.mpii = mpii
		with open(path, 'r') as fin:
			lines = [x.strip().split(',') for x in fin]

		self.imageFiles = [x[0] for x in lines]
		self.gazes 		= [x[1:] if len(x[1:]) == 3 else x[1:-1] for x in lines]
		

		
		self.data_len = len(self.imageFiles)

		self.transform = transforms.Compose([
							transforms.Grayscale(),
							transforms.ToTensor(),
							transforms.Normalize([0],[1])])
		self.path = path	
		
		self.badFile = True
		self.indexes = list(range(self.data_len))

	# used externally
	# prepares the data by opening image / applying the transform
	def __getitem__(self, index):
		
		# sometimes files get removed and 
		# an img path in the text file may
		# have been removed.
		# This finds an image path that exists
		# from the training/testing file
		while self.badFile:
			try :
				image_file = self.imageFiles[index]
				img_as_img = Image.open(image_file)
				if not self.mpii:
					image_gaze = np.array(self.gazes[index], dtype=np.float64)
				else:
					image_gaze = np.array(list([self.gazes[index][1], self.gazes[index][0], self.gazes[index][2]]), dtype=np.float64)
				self.badFile = False
			except IOError:
				print("Could not open filename : {}. Choosing random index.\n".format(self.imageFiles[index]))
				index = np.random.choice(self.indexes, 1)[0]
				self.badFile = True
		
		self.badFile = True


		img_as_tensor = self.transform(img_as_img)
		gaze_as_tensor = torch.tensor(image_gaze, dtype=torch.float)
		return img_as_tensor, gaze_as_tensor, image_file
	
	# used internally / externally
	# get the number of data points
	def __len__(self):
		return self.data_len
