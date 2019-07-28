'''
	This file has functions that are used to structure
	image plots to check how well the refiner is
	refining the images. It isn't terrible important.
	
	This is also taken from the repository linked at 
	the bottom of this README
'''
import numpy as np

# Move a Variable to CPU
# and convert to NUMPY
def var_to_np(img_var):
	return img_var.data.cpu().numpy()


# Get axes values
# This is a helper function 
# for stack_images
def get_transpose_axes(n):
	if n % 2 == 0:
		y_axes = list(range(1, n - 1, 2))
		x_axes = list(range(0, n - 1, 2))
	else:
		y_axes = list(range(0, n - 1, 2))
		x_axes = list(range(1, n - 1, 2))
	return y_axes, x_axes, [n - 1]

# this function just stacks an array of images
# into a 2d grid. This is what gets saved or displayed
# during training
def stack_images(images):
	images_shape = np.array(images.shape)
	new_axes = get_transpose_axes(len(images_shape))
	new_shape = [np.prod(images_shape[x]) for x in new_axes]
	return np.transpose(images, axes=np.concatenate(new_axes)).reshape(new_shape)

