from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
import torch
from skimage.transform import resize
import utils
import params
from PIL import Image

class baiduDataset(Dataset):
	def __init__(self, img_root, label_path, alphabet, isBaidu, resize, transforms=None):
		super(baiduDataset, self).__init__()
		self.img_root = img_root
		self.isBaidu = isBaidu
		self.labels = self.get_labels(label_path)
		# print(self.labels[:10])
		self.alphabet = alphabet
		self.transforms = transforms
		self.width, self.height = resize
		# print(list(self.labels[1].values())[0])
	def get_labels(self, label_path):
		# return text labels in a list
		if self.isBaidu:
			with open(label_path, 'r', encoding='utf-8') as file:
				# {"image_name":"chinese_text"}
				content = [[{c.split('\t')[2]:c.split('\t')[3][:-1]},{"w":c.split('\t')[0]}] for c in file.readlines()];
			labels = [c[0] for c in content]
			# self.max_len = max([int(list(c[1].values())[0]) for c in content])
		else:
			with open(label_path, 'r', encoding='utf-8') as file:
				labels = [ {c.split(' ')[0]:c.split(' ')[-1][:-1]}for c in file.readlines()]	
		return labels


	def __len__(self):
		return len(self.labels)

	# def compensation(self, image):
	# 	h, w = image.shape # (48,260)
	# 	image = cv2.resize(image, (0,0), fx=280/w, fy=32/h, interpolation=cv2.INTER_CUBIC)
	# 	# if w>=self.max_len:
	# 	# 	image = cv2.resize(image, (0,0), fx=280/w, fy=32/h, interpolation=cv2.INTER_CUBIC)
	# 	# else:
	# 	# 	npi = -1*np.ones(self.max_len-)

	# 	return image
	def preprocessing(self, image):

		## already have been computed
		image = image.astype(np.float32) / 255.
		image = torch.from_numpy(image).type(torch.FloatTensor)
		image.sub_(params.mean).div_(params.std)

		return image

	def __getitem__(self, index):
		image_name = list(self.labels[index].keys())[0]
		# label = list(self.labels[index].values())[0]
		image = cv2.imread(self.img_root+'/'+image_name)
		# print(self.img_root+'/'+image_name)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		h, w = image.shape
		# Data augmentation
		# width > len ==> resize width to len
		# width < len ==> padding width to len 
		# if self.isBaidu:
		# 	# image = self.compensation(image)
		# 	image = cv2.resize(image, (0,0), fx=160/w, fy=32/h, interpolation=cv2.INTER_CUBIC)
		image = cv2.resize(image, (0,0), fx=self.width/w, fy=self.height/h, interpolation=cv2.INTER_CUBIC)
		image = (np.reshape(image, (32, self.width, 1))).transpose(2, 0, 1)
		image = self.preprocessing(image)

		return image, index

		


if __name__ == '__main__':
	dataset = baiduDataset("H:/DL-DATASET/BaiduTextR/train_images/train_images", "H:/DL-DATASET/BaiduTextR/train.list", params.alphabet, True)
	# dataset = baiduDataset("H:/DL-DATASET/360M/images", "E:/08-Github-resources/00-MY-GitHub-Entries/crnn_chinese_characters_rec-master/crnn_chinese_characters_rec-master/label/test.txt", params.alphabet, False)
	dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
	# alphabet = utils.to_alphabet("H:/DL-DATASET/BaiduTextR/train.list")
	
	for i_batch, (image, index) in enumerate(dataloader):
		print(image.shape)
