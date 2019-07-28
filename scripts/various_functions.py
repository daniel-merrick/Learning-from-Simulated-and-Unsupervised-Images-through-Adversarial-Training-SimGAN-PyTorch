''' 
	This file is meant to hold various scripts I used to
	clean / move the data around. 

	I will document it, later.
'''



import os
import shutil
import random
import cv2
import numpy as np
import json

'''
Get a random permutation of a certain number of files (numFiles)
    from a source path (path)
'''

def getImageFiles(path, numFiles):
	files = [file_ for file_ in os.listdir(path) if file_[-3:] == 'png' or file_[-3:] == 'jpg']
	random.shuffle(files)
	return(files[:numFiles])

'''
Copy image files from some source directory (srcPath)
		 to some destination directory (dstPath).
'''

def copyImageFiles(srcPath, imgFiles, dstPath):
	for imgFile in imgFiles:
		shutil.copy(srcPath + imgFile, dstPath + imgFile)		

'''
Change the illumination of an image
'''

def adjustGamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
	  for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image,table)

''' Open image using OpenCV '''

def openImage(img_path):
	return(cv2.imread(img_path))

''' Use the following coordinates for the synthetic data :
	x1, x2, y1, y2 = 450, 800, 120, 450 '''

def cropImage(x1, x2, y1, y2, img):
	return(img[y1:y1+(y2-y1), x1:x1+(x2-x1)])

''' Save Image using OpenCV '''
def saveImage(dstPath, imgFile, img, type_='jpg'):
	cv2.imwrite(dstPath + imgFile[:-3] + type_, img)

''' Resize Image '''
def resizeImage(img, size=256):
	return cv2.resize(img, (size, size))

''' Crop and Resize images in a srcDirectory ''' 
def cropFolder(srcPath, dstPath, x1, x2, y1, y2, size=256):
	for imgFile in os.listdir(srcPath):
		img = openImage(srcPath + imgFile)
		img = cropImage(x1, x2, y1, y2, img)
		img = resizeImage(img, size)
		img = saveImage(dstPath, imgFile, img)


''' Split UnityEyes img folder into training and testing 
	data set splits. '''
def splitUnityEyes(srcPath, cropPath, trainFile, testFile, trainPcnt=.8):
	
	''' srcPath is the path to the img folder '''
	
	JSONfiles = np.array([x for x in os.listdir(srcPath) if x[-4:] == 'json'])
	np.random.shuffle(JSONfiles)
	
	JSONfiles_train = JSONfiles[:int(len(JSONfiles) * trainPcnt)]
	JSONfiles_test = JSONfiles[int(len(JSONfiles) * trainPcnt):]

	def process_json_dict(json_dict):
		return(json_dict[1:-1].split(', '))
	
	with open(trainFile, 'w') as fout:
		for json_f in JSONfiles_train:
			with open(srcPath + json_f) as json_ff:
				j = json.loads(json_ff.read())
				key = 'eye_details'
				gaze = process_json_dict(j[key]['look_vec'])
			fout.write(cropPath + json_f[:-4] + 'jpg' + ',' + ','.join(gaze) + '\n'))
	
	with open(testFile, 'w') as fout:
		for json_f in JSONfiles_test:
			with open(srcPath + json_f) as json_ff:
				j = json.loads(json_ff.read())
				key = 'eye_details'
				gaze = process_json_dict(j[key]['look_vec'])
			fout.write(cropPath + json_f[:-4] + 'jpg' + ',' + ','.join(gaze) + '\n'))

''' Split MPIIGaze img folder into training and testing
	data splits '''
def splitMPIIFolder(srcDirectory, trainFile, testFile, trainPcnt=.8):
	imgFiles = np.array([x for x in os.listdir(srcDirectory)])
	np.random.shuffle(imgFiles)
	
	gazes = np.array([x[:-4].split('_')[5:] for x in imgFiles])
	
	
	trainImgs = imgFiles[:int(len(imgFiles) * trainPcnt)]
	trainGazes = gazes[:int(len(gazes) * trainPcnt)]

	testImgs = imgFiles[int(len(imgFiles) * trainPcnt):]
	testGazes = gazes[int(len(imgFiles) * trainPcnt):]

	with open(trainFile, 'w') as fout:
		for imgFile, gaze in zip(trainImgs, trainGazes):
			fout.write(srcDirectory + imgFile + ',' + ','.join(gaze)+'\n')

	with open(testFile, 'w') as fout:
		for imgFile, gaze in zip(testImgs, testGazes):
			fout.write(srcDirectory + imgFile + ',' + ','.join(gaze)+'\n')
	
def to_grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def convertFFHQFolderToGrayScale(srcPath, dstPath):
	for imgFile in os.listdir(srcPath):
		print(srcPath + imgFile)
		img = openImage(srcPath + imgFile)
		saveImage(dstPath, imgFile, to_grayscale(img))

if __name__ == '__main__':
	srcPath = '/raid/dmerric5/ffhq-dataset/images1024x1024_croppedEyes/'
	dstPath = '/raid/dmerric5/ffhq-dataset/images1024x1024_croppedEyes_grayScale/'
	

	#convertFFHQFolderToGrayScale(srcPath, dstPath)
	splitMPIIFolder('/raid/dmerric5/MPIIGaze/cropped_eyes_with_gaze/', '/raid/dmerric5/MPIIGaze/MPIIGaze_TrainFile.txt', '/raid/dmerric5/MPIIGaze/MPIIGaze_TestFile.txt')
	
	"""
	path = '/raid/dmerric5/synthetic_data/run20/'
	numFiles = 10000
	imgFiles = getImageFiles(path, numFiles)	
	copyImageFiles(path, imgFiles, '/raid/dmerric5/deepfake-synthetic-more/')

	
	srcPath = '/raid/dmerric5/deepfake-synthetic-more/'
	dstPath = '/raid/dmerric5/deepfake-synthetic-cropped-more/'
	x1, x2, y1, y2 = 450, 800, 100, 500 #synthetic

	cropFolder(srcPath, dstPath, x1, x2, y1, y2)	
	"""

	#splitUnityEyes('/raid/dmerric5/UnityEyes/imgs/', '/raid/dmerric5/UnityEyes/imgs_cropped/', 'UnityEyes_Train.txt', 'UnityEyes_Test.txt')
