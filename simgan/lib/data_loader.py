'''
	DataLoader over rides the torch.utils.data.dataset.Dataset class
	This should be changed dependent on the format of your data sets.
'''

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import os

class DataLoader(Dataset):
	'''
		DataLoader: class to load the data. Passed into torch.utils.data.dataloader

			Notes*
				1. Input is the path to the directory with all the images
				2. This class is overriding the torch.utils.data.dataset.Dataset class
				3. This class is passed to torch.utils.data.dataloader
	'''
	# used internally
	# load image files, get # of images, load transforms
	def __init__(self, path):
		self.imageFiles = [img for img in os.listdir(path)]
		self.data_len = len(self.imageFiles)

		self.transform = transforms.Compose([transforms.ToTensor()])
		self.path = path	
	
	# used externally
	# prepares the data by oppening image / applying the transform
	def __getitem__(self, index):
		image_file = self.imageFiles[index]
		img_as_img = Image.open(self.path + image_file)
		img_as_tensor = self.transform(img_as_img)

		return img_as_tensor, image_file
	
	# used internally / externally
	# get the number of data points
	def __len__(self):
		return self.data_len
