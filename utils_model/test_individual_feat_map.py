#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import copy
import torchvision.transforms as T



########################################################
##               DATASET CAT_DOG                      ##	
########################################################

class MyDataset(Dataset):
	def __init__(self,base,base_image,input_path, num_images = None, transform=None, idx_change = None,image_modify = None):
		
		
		self.input_samples = pd.read_csv(base+input_path)
		self.input_samples = self.input_samples.loc[self.input_samples['label'] != -1].reset_index()
		self.input_samples_param = self.input_samples
		self.transform = transform
		self.base = base
		self.base_image = base_image
		self.num_images = num_images
		self.image_modify = image_modify
		self.idx_change = idx_change

	def __getitem__(self, idx):
		param = 0
		image_path = self.input_samples['path'][idx]
		label = self.input_samples['label'][idx]

		sample =Image.open(self.base_image+image_path, mode='r').convert("RGB")	
		
		sample = sample.resize((224,224))
	
		train_mean = np.array([[[106.20628246,115.9285463,124.40483277]]], dtype=np.float32)
		train_std = np.array([[[65.59749505,64.94964833,66.61112731]]], dtype=np.float32)

		
		sample = (sample - train_mean) / train_std
		if (self.image_modify != None) and (self.idx_change== idx):

			sample = self.image_modify.permute(1,2,0).detach().cpu().numpy()
			
		
		label = int(label)
		if label==1:
			
			label =  [1,0]
		else:
			
			label =  [0,1]

		if self.transform:
			sample = self.transform(sample)
			
		return torch.Tensor(sample), self.base_image+image_path,np.asarray(label).astype(np.float32).reshape(-1,)	

	def __len__(self):
		if self.num_images == None:
			return len(self.input_samples_param)
		else:
			return self.num_images


########################################################
##                   DATASET CUB                      ##	
########################################################


class MyDataset_CUB(Dataset):
	
	def __init__(self,base,base_image,input_path, num_images = None, transform=None, idx_change = None,image_modify = None):

		self.class1 = np.arange(113,134) #sparrow
		self.class2 = np.arange(158,179) #warbler vai até 182
		self.input_samples = pd.read_csv(base+input_path)
		self.input_samples = self.input_samples[np.logical_or(self.input_samples['class'].isin(self.class1),self.input_samples['class'].isin(self.class2))]
		self.input_samples = self.input_samples.loc[self.input_samples['class'] != -1].reset_index()	
		self.input_samples_param = self.input_samples
		self.transform = transform
		self.base = base
		self.base_image = base_image
		self.num_images = num_images
		self.image_modify = image_modify
		self.idx_change = idx_change
		
		

	def __getitem__(self, idx):
		
		image_path = self.input_samples['path'][idx]
		label = self.input_samples['class'][idx]
		sample =Image.open(self.base_image+image_path, mode='r').convert("RGB")	
		sample = sample.resize((224,224))
		train_mean = np.array([[[106.20628246,115.9285463,124.40483277]]], dtype=np.float32)
		train_std = np.array([[[65.59749505,64.94964833,66.61112731]]], dtype=np.float32)

		# zero mean and unit variance
		sample = (sample - train_mean) / train_std


		if (self.image_modify != None) and (self.idx_change== idx):
			sample = self.image_modify.permute(1,2,0).detach().cpu().numpy()

		
		label = int(label)
		if label in self.class1:
			
			label =  [1,0]
			#print('class1')
		else:
			
			label =  [0,1]  
			#print('class2')

		if self.transform:
			sample = self.transform(sample)

		return torch.Tensor(sample), self.base_image+image_path,np.asarray(label).astype(np.float32).reshape(-1,)	

	def __len__(self):
		return len(self.input_samples_param)



#********************************************************************************************************************#

def test(net, dataloader,device='cpu'):

	'''
	Input: model and dataloader with images to be tested
	Output: total accuracy and loss
	'''

	correct = 0
	total = 0
	correct_cat = 0
	dog_label = 0
	cat_label = 0
	correct_dog = 0
	loss = 0
	prob = nn.Softmax(dim = 1)
	net.eval()
	with torch.no_grad():
		for i,data in enumerate(dataloader):
			print('Batch:', i)
			images,_, labels = data
			images, labels = images.to(device), labels.to(device)

			outputs = prob(net(images))
			loss +=  torch.nn.functional.binary_cross_entropy(outputs, labels)
			_, predicted = torch.max(outputs.data, 1)
			_, labels_aux = torch.max(labels, 1)
			total += labels.size(0)
			correct += (predicted == labels_aux).sum().item()
			dog_label += labels_aux.sum()
			cat_label += len(labels_aux) - labels_aux.sum()
			for t in range(len(predicted)):
			
				correct_dog += ((predicted[t] == labels_aux[t]) and (labels_aux[t] == 1)).sum().item()
				correct_cat += ((predicted[t] == labels_aux[t]) and (labels_aux[t] == 0)).sum().item()


	
	print('Loss val:', loss/i)
	print('Accrracy: %f %%' % (100 * correct / total))
	print('Dog: %f %%' % (100 * correct_dog / dog_label))
	print('Cat: %f %%' % (100 * correct_cat / cat_label))
	print(correct_dog,correct_cat, dog_label, cat_label)

	return (100 * correct / total), (loss.detach().cpu().numpy()/i)


def extract_probs(net, dataloader, idx=None,device='cpu'):

	'''
	Input: model and dataloader with images to extract activations
	Output: matrix of extracted final probabilities (before softmax)
	'''

	activations = []
	
	net.eval()
	with torch.no_grad():
		if idx==None:
			for i,data in enumerate(dataloader):
				print('Batch:', i)
				images,_, labels = data
				images, labels = images.to(device), labels.to(device)

				outputs = net(images)
				activations.extend(np.array(outputs.detach().cpu()))
		else:
			image,_, labels = dataloader.dataset[idx]
			outputs = net(torch.Tensor(image).unsqueeze(0).cuda())
			activations = np.array(outputs.detach().cpu())

	
	return np.array(activations)




def zera(feat_maps, num_feat,analyzed_layer, model):

	'''
	Input: list of feature maps ids to maintain, number of feature maps, analyzed layer to filter the file, inverted top with the chosen number of smallest correlations to be deleted, and the copy of the model to be manipulated
	Output: the modified copy of the model
	'''

	## testar isso aqui, nao sei se ta certo
		
	string ='model.'+analyzed_layer

	with torch.no_grad():
		#print(m['layer'])
		module = eval(string)

		for feat_map in range(num_feat):
			#print(module)
			if feat_map not in feat_maps:
				module.weight[feat_map,:,:,:] = torch.nn.Parameter(torch.zeros_like(module.weight[feat_map,:,:,:]))
			

	return(model)




