#!/usr/bin/env python
# coding: utf-8

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import copy
import collections
from functools import partial
import causal_viz.occlude_images_model as occlude_images_model


#--------------------------------------------------------------



def save_tiles_as_images(max_cols, max_rows, data, path_save, size_tile, orders, top, filter_label):


	number_images = max_cols * max_rows

	
	data['order'] = orders[1,:]
	
	data['label_single'] = orders[0,:]

	pd.options.display.max_colwidth = 1000


	if filter_label != None:
		data = data[data['label_single'] == filter_label].reset_index()

	data_ordered = data.sort_values(by=['order','label_single'], ascending=True).reset_index()


	fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20*max_cols,20*max_rows))

	
	for idx in range(top):

		
		image =Image.open(data_ordered.iloc[idx]['filename'], mode='r').convert("RGB")	
		image = image.resize((224,224))

		

		image = occlude_images_model.crop_image(image, size_tile, data_ordered.iloc[idx]['tile_x'], data_ordered.iloc[idx]['tile_y'])

		row = idx // max_cols
		col = idx % max_cols

		ax = fig.add_subplot(max_rows, max_cols, 1 + idx)


		ax.imshow(image, aspect="auto") #, cmap="gray"

	plt.subplots_adjust(wspace=.05, hspace=.05)
	plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
	#plt.tick_params(axis='both', which='both',  right=False, left=False, labelleft = False, top=False, bottom=False,labelbottom = False)
	plt.savefig(path_save)



########################################################
##               DATASET CAT_DOG                      ##	
########################################################

class MyDataset_tile(Dataset):
	def __init__(self,base,base_image,input_path,size_tile,num_images = None, transform=None):
		
		
		self.input_samples = pd.read_csv(base+input_path)
		self.input_samples = self.input_samples.loc[self.input_samples['label'] != -1].reset_index()
		

		self.input_samples_param = self.input_samples
		self.size_tile = size_tile
		self.num_tiles_side = (224// self.size_tile)
		
		self.transform = transform
		self.base = base
		self.base_image = base_image
		self.num_images = num_images
		
		
		

	def __getitem__(self, idx):

	
		real_idx, tile_x, tile_y = occlude_images_model.find_image_position(idx, self.num_tiles_side)
		image_path = self.input_samples['path'][real_idx]
		label = self.input_samples['label'][real_idx]


		sample =Image.open(self.base_image+image_path, mode='r').convert("RGB")	
		sample = sample.resize((224,224))

		sample = occlude_images_model.crop_image(sample, self.size_tile, tile_x, tile_y)

		sample = sample.resize((224,224))
		
		train_mean = np.array([[[106.20628246,115.9285463,124.40483277]]], dtype=np.float32)
		train_std = np.array([[[65.59749505,64.94964833,66.61112731]]], dtype=np.float32)
		# zero mean and unit variance
		sample = (sample - train_mean) / train_std

		
		label = int(label)
		if label==1:
			
			label =  [1,0]
		else:
			
			label =  [0,1]


		if self.transform:
			sample = self.transform(sample)


		return torch.Tensor(sample), self.base_image+image_path,np.asarray(label).astype(np.float32).reshape(-1,), tile_x, tile_y	

	def __len__(self):

		if self.num_images == None:
			return len(self.input_samples_param)*self.num_tiles_side*self.num_tiles_side
		else:
			return self.num_images*self.num_tiles_side*self.num_tiles_side



########################################################
##                   DATASET CUB                      ##	
########################################################


class MyDataset_tile_CUB(Dataset):
	
	def __init__(self,base,base_image,input_path,size_tile,num_images = None, transform=None):

		self.class1 = np.arange(113,134) #sparrow
		self.class2 = np.arange(158,179) #warbler vai até 182
		self.input_samples = pd.read_csv(base+input_path)
		self.input_samples = self.input_samples[np.logical_or(self.input_samples['class'].isin(self.class1),self.input_samples['class'].isin(self.class2))]
		self.input_samples = self.input_samples.loc[self.input_samples['class'] != -1].reset_index()
		self.input_samples_param = self.input_samples
		self.input_samples_param = self.input_samples
		self.size_tile = size_tile
		self.num_tiles_side = (224// self.size_tile)
		
		self.transform = transform
		self.base = base
		self.base_image = base_image
		self.num_images = num_images
		
		

	def __getitem__(self, idx):
		
		real_idx, tile_x, tile_y = occlude_images_model.find_image_position(idx, self.num_tiles_side)

		#print(real_idx, tile_x, tile_y)

		image_path = self.input_samples['path'][real_idx]
		label = self.input_samples['class'][real_idx]


		sample =Image.open(self.base_image+image_path, mode='r').convert("RGB")	
		sample = sample.resize((224,224))

		sample = occlude_images_model.crop_image(sample, self.size_tile, tile_x, tile_y)
		sample = sample.resize((224,224))

		
		train_mean = np.array([[[106.20628246,115.9285463,124.40483277]]], dtype=np.float32)
		train_std = np.array([[[65.59749505,64.94964833,66.61112731]]], dtype=np.float32)
		# zero mean and unit variance
		sample = (sample - train_mean) / train_std

		
		label = int(label)
		if label in self.class1:
			
			label =  [1,0]
			#print('class1')
		else:
			
			label =  [0,1]  
			#print('class2')

		if self.transform:
			sample = self.transform(sample)


		return torch.Tensor(sample), self.base_image+image_path,np.asarray(label).astype(np.float32).reshape(-1,), tile_x, tile_y	

	def __len__(self):

		if self.num_images == None:
			return len(self.input_samples_param)*self.num_tiles_side*self.num_tiles_side
		else:
			return self.num_images*self.num_tiles_side*self.num_tiles_side



def extract_probs(net, dataloader,device='cpu'):

	'''
	Input: model and dataloader with images to extract activations
	Output: matrix of extracted final probabilities (before softmax)
	'''

	activations = []
	
	net.eval()
	with torch.no_grad():
		for i,data in enumerate(dataloader):
			print('Batch:', i)
			images,_, labels,_,_ = data
			images, labels = images.to(device), labels.to(device)

			outputs = net(images)

			activations.extend(outputs)

	

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

def obtain_activations(model,data,name,layer,device='cpu'):

	'''
	Input: model, dataloader and object layer to extract activations 
	Output: vector of summed positive activations for the specified layer and all layer's feature maps, finenames of the original image tiles 
	'''

	print('Begin activation')

	activations = collections.defaultdict(list)
	relu = nn.ReLU()

	def save_activation(name, mod, inp, out):
		activations[name].append(relu(out.detach()).cpu().sum(axis=(2,3)))
	
	params = []
	filename = []
	labels = []
	tiles_x = []
	tiles_y = []

	layer.register_forward_hook(partial(save_activation, name))
	
	for batch_idx, batch in enumerate(tqdm(data)):
		image,filen,label, tile_x, tile_y=batch
		image = image.to(device)
		out = model(image)
		
		filename.extend(filen)
		labels.extend(label.detach().cpu().numpy())
		tiles_x.extend(tile_x.detach().cpu().numpy())
		tiles_y.extend(tile_y.detach().cpu().numpy())

		
		
	activations_aux = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}
	labels = np.array(labels)

	
	print('End activation')
	
	return np.array(activations_aux[name]), pd.DataFrame({'filename': filename, 'labels': labels[:,1],'tile_x': tiles_x,'tile_y': tiles_y})


def find_best_per_feat(activations_sum,filenames, num_feat_map, top, size_tile):

	'''
	Input: vector of summed positive activations for the specified layer and all layer's feature maps, index of feature map to be analyzed, number of most activated tiles to be selected, size tile
	Output: positions of the most activated tiles and vector containing original image filenames, tile coordinates and labels 
	'''

	num_tiles_side = (224// size_tile)

	chosen_tiles = []
	corresponding_paths = []

	

	for i in tqdm(range(0,len(activations_sum), num_tiles_side*num_tiles_side)):
		
		order = np.argsort(-activations_sum[i:(i+ num_tiles_side*num_tiles_side),num_feat_map])

		for k in range(top):
			chosen_tiles.append(i+order[k])
			corresponding_paths.append(filenames.iloc[i+order[k]]) #order[k]
		

	chosen_tiles = np.array(chosen_tiles)
	
	return chosen_tiles,pd.DataFrame(data = corresponding_paths, columns = ['filename', 'labels', 'tile_x', 'tile_y'])


def find_top_ranking(rankings_image,filenames, top, size_tile):

	'''
	Input: vector containing aggregated rankings of tiles for image,number of most activated tiles to be selected, size tile
	Output: positions of the most activated tiles and vector containing original image filenames, tile coordinates and labels 
	'''

	num_tiles_side = (224// size_tile)

	chosen_tiles = []
	corresponding_paths = []

	

	for i in tqdm(range(len(rankings_image))):
		
		for k in range(top):
			chosen_tiles.append(i*num_tiles_side*num_tiles_side+rankings_image[i][k])
			
			corresponding_paths.append(filenames.iloc[i*num_tiles_side*num_tiles_side+rankings_image[i][k]]) #order[k]
		

	chosen_tiles = np.array(chosen_tiles)
	
	return chosen_tiles,pd.DataFrame(data = corresponding_paths, columns = ['filename', 'labels', 'tile_x', 'tile_y'])
	



	


		
